import os
import glob
import pandas as pd
import numpy as np
import spconv.pytorch as spconv
import torch

from tqdm.auto import tqdm

from utils import voxelize, voxelize_full, build_semantickitti_index
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

def run_inference_and_save(
    root_dir: str,
    ckpt_path: str,
    out_root: str,
    voxel_size = [0.2, 0.2, 0.2],
    point_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    device = "cuda",
):
    os.makedirs(out_root, exist_ok=True)

    index_df = build_semantickitti_index(root_dir, SEMANTICKITTI_SPLIT)
    test_df = index_df[index_df["split"] == "test"].reset_index(drop=True)#.sample(500)
    print("Test samples:", len(test_df))

    test_dataset = SemanticKITTIDataset(
        test_df,
        remap_labels=True,
    )

    model = DRINetPlusPlus(
        in_channels=4,
        out_channels=64,
        num_blocks=4,
        num_classes=NUM_CLASSES,
        scales=[2,4,8,16],
    )
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    print(model.load_state_dict(ckpt["model_state"]))
    print(f"Loaded checkpoint from {ckpt_path}, epoch = {ckpt.get('epoch','?')}")

    inv_lut = build_inv_lut(device=device)

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Inference on test"):
            sample = test_dataset[idx]
            seq    = int(sample["sequence"])
            frame  = int(sample["frame"])

            points = sample["points"].to(device)   # (N,3)
            feats  = sample["feats"].to(device)    # (N,C)
            N = points.size(0)

            vox = voxelize_full(
                points=points,
                feats=feats,
                voxel_size=voxel_size,
                batch_idx=0,
            )

            v_feats       = vox["v_feats"]          # (M,C)
            v_coords      = vox["v_coords"]         # (M,4)
            spatial_shape = vox["spatial_shape"]    # [Z,Y,X]
            point2voxel   = vox["point2voxel"]      # (N_kept,)
            point_mask    = vox["point_mask"]       # (N,)

            point2voxel_kept = point2voxel[point_mask] 
            
            sp_tensor = spconv.SparseConvTensor(
                features      = v_feats,
                indices       = v_coords,
                spatial_shape = spatial_shape,
                batch_size    = 1,
            )

            # forward
            logits = model(sp_tensor, point2voxel_kept)   # (N_kept, num_classes)
            preds_train = logits.argmax(dim=1)       # (N_kept,)

            preds_full_train = torch.zeros(N, dtype=torch.long, device=device)
            preds_full_train[point_mask] = preds_train

            preds_full_raw = inv_lut[preds_full_train]  # (N,)

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
    ckpt_path = "weight/drinetpp_best_mIoU.pth"
    out_root  = "data/output/"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference_and_save(
        root_dir=root_dir,
        ckpt_path=ckpt_path,
        out_root=out_root,
        voxel_size=[0.2, 0.2, 0.2],
        point_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        device=device,
    )