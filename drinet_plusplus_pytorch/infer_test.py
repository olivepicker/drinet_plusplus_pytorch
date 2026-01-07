import os
import glob
import pandas as pd
import numpy as np
import spconv.pytorch as spconv
import torch

from tqdm.auto import tqdm

from utils import voxelize, build_semantickitti_index
from dataset import SemanticKITTIDataset
from drinet_plusplus_pytorch import DRINetPlusPlus

NUM_CLASSES = 20
LEARNING_MAP_INV = {
    0: 0,     # unlabeled
    1: 10,    # car
    2: 11,    # bicycle
    3: 15,    # motorcycle
    4: 18,    # truck
    5: 20,    # other-vehicle
    6: 30,    # person
    7: 31,    # bicyclist
    8: 32,    # motorcyclist
    9: 40,    # road
    10: 44,   # parking
    11: 48,   # sidewalk
    12: 49,   # other-ground
    13: 50,   # building
    14: 51,   # fence
    15: 70,   # vegetation
    16: 71,   # trunk
    17: 72,   # terrain
    18: 80,   # pole
    19: 81,   # traffic-sign
}

def build_inv_lut(device="cpu"):
    lut = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    for tid, rid in LEARNING_MAP_INV.items():
        lut[tid] = rid
    return lut

SEMANTICKITTI_SPLIT = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    "valid": [8],
    "test":  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


def get_tta_transforms(device):
    """
    standard TTA:
      - flip_y: False / True
      - yaw: 0, 90, 180, 270 deg
    """
    angles = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    transforms = []

    for flip_y in [False, True]:
        for a in angles:
            c = np.cos(a)
            s = np.sin(a)
            R = torch.tensor([
                [ c, -s, 0.0],
                [ s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32, device=device)
            transforms.append((flip_y, R))
    return transforms


def apply_tta_transform(points, feats, flip_y, R):
    """
    points: (N, 3)
    feats : (N, 4)  # [x,y,z,intensity] 라고 가정
    """
    pts = points
    f   = feats

    # flip_y: train augmentation에서 y축 flip 쓰던 것과 동일하게
    if flip_y:
        pts = pts.clone()
        pts[:, 1] = -pts[:, 1]

        f = f.clone()
        f[:, 1] = -f[:, 1]        # xyz 부분도 같이 뒤집기

    # z축 회전
    pts_rot = pts @ R.T          # (N,3)

    # feats도 xyz 부분을 동일하게 회전
    f_rot = f.clone()
    f_rot[:, :3] = pts_rot       # [x,y,z] 부분 교체, intensity는 그대로 유지

    return pts_rot, f_rot

def run_inference_and_save(
    root_dir: str,
    ckpt_path: str,
    out_root: str,
    voxel_size = [0.2, 0.2, 0.2],
    point_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    device = "cuda",
    use_tta = True,
):
    os.makedirs(out_root, exist_ok=True)

    index_df = build_semantickitti_index(root_dir, SEMANTICKITTI_SPLIT)
    test_df = index_df[index_df["split"] == "test"].reset_index(drop=True)
    print("Test samples:", len(test_df))

    test_dataset = SemanticKITTIDataset(
        test_df,
        remap_labels=True,
        augment=False,
    )

    model = DRINetPlusPlus(
        in_channels=4,
        out_channels=64,
        num_blocks=4,
        num_classes=NUM_CLASSES,
        scales=[2,4,8,16],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    print(model.load_state_dict(ckpt["model_state"]))
    print(f"Loaded checkpoint from {ckpt_path}, epoch = {ckpt.get('epoch','?')}")

    model.eval()

    inv_lut = build_inv_lut(device=device)
    tta_transforms = get_tta_transforms(device) if use_tta else None

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Inference on test"):
            sample = test_dataset[idx]
            seq    = int(sample["sequence"])
            frame  = int(sample["frame"])

            points = sample["points"].to(device)   # (N,3)
            feats  = sample["feats"].to(device)    # (N,C)
            N      = points.size(0)

            if not use_tta:
                vox = voxelize(
                    points=points,
                    feats=feats,
                    voxel_size=voxel_size,
                    point_range=point_range,
                    batch_idx=0,
                )

                v_feats       = vox["v_feats"]
                v_coords      = vox["v_coords"]
                spatial_shape = vox["spatial_shape"]
                point2voxel   = vox["point2voxel"]   # (N_kept,)
                point_mask    = vox["point_mask"]    # (N_raw,)

                sp_tensor = spconv.SparseConvTensor(
                    features      = v_feats,
                    indices       = v_coords,
                    spatial_shape = spatial_shape,
                    batch_size    = 1,
                )

                logits = model(sp_tensor, point2voxel)   # (N, num_classes)
                preds_train = logits.argmax(dim=1)       # (N,)
                preds_full = torch.zeros(N, dtype=torch.long, device=device)
                preds_full[point_mask] = preds_train
                preds_train = preds_full

            else:
                logits_accum = torch.zeros(
                    (N, NUM_CLASSES),
                    dtype=torch.float32,
                    device=device,
                )
                num_tta = 0

                for flip_y, R in tta_transforms:
                    pts_aug, feats_aug = apply_tta_transform(points, feats, flip_y, R)

                    vox = voxelize(
                        points=pts_aug,
                        feats=feats_aug,
                        voxel_size=voxel_size,
                        point_range=point_range,
                        batch_idx=0,
                    )

                    v_feats       = vox["v_feats"]
                    v_coords      = vox["v_coords"]
                    spatial_shape = vox["spatial_shape"]
                    point2voxel   = vox["point2voxel"]   # (N_kept,)
                    point_mask    = vox["point_mask"]    # (N_raw,)

                    sp_tensor = spconv.SparseConvTensor(
                        features      = v_feats,
                        indices       = v_coords,
                        spatial_shape = spatial_shape,
                        batch_size    = 1,
                    )

                    logits_t = model(sp_tensor, point2voxel)  # (N, num_classes)
                    logits_accum += logits_t
                    num_tta += 1

                logits = logits_accum / max(num_tta, 1)
                preds_train = logits.argmax(dim=1)   # (N,)

            preds_full_raw = inv_lut[preds_train]  # (N,)

            sem = preds_full_raw.cpu().numpy().astype(np.uint32)
            inst = np.zeros_like(sem, dtype=np.uint32)
            final_label = (inst << 16) | sem

            seq_str = f"{seq:02d}"
            frame_str = f"{frame:06d}"
            out_dir = os.path.join(out_root, "sequences", seq_str, "predictions")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{frame_str}.label")

            final_label.tofile(out_path)

    print(f"\nDone! Predictions saved under: {out_root}")

if __name__ == '__main__':
    root_dir  = "data/dataset/"
    ckpt_path = "drinet_plusplus_pytorch/drinetpp_best_mIoU.pth"
    out_root  = "data/output/"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference_and_save(
        root_dir=root_dir,
        ckpt_path=ckpt_path,
        out_root=out_root,
        voxel_size=[0.2, 0.2, 0.2],
        point_range=[-48., -48., -3.0, 48.,  48.,  1.8],
        device=device,
        use_tta=False
    )