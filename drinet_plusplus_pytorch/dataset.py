import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import voxelize

LEARNING_MAP = {
    0:   0,  1:   0,
    10:  1,  11:  2,  13:  5,
    15:  3,  16:  5,  18:  4,  20:  5,
    30:  6,  31:  7,  32:  8,
    40:  9,  44: 10,  48: 11,  49: 12,
    50: 13,  51: 14,
    52:  0,  60:  9,
    70: 15,  71: 16,  72: 17,
    80: 18,  81: 19,
    99:  0,
    252: 1, 253: 7, 254: 6, 255: 8,
    256: 5, 257: 5, 258: 4, 259: 5,
}

def build_label_lut(device='cpu'):
    max_key = max(LEARNING_MAP.keys())
    lut = torch.zeros(max_key + 1, dtype=torch.long, device=device)
    for raw_id, train_id in LEARNING_MAP.items():
        lut[raw_id] = train_id
    return lut

class SemanticKITTIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, remap_labels=True):
        self.df = df.reset_index(drop=True)
        self.remap_labels = remap_labels

        if remap_labels:
            self.remap_lut = build_label_lut(device='cpu')
        else:
            self.remap_lut = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        velo_path  = row["velodyne_path"]
        label_path = row["label_path"]

        scan = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        xyz  = scan[:, :3]
        inten = scan[:, 3:4]
        feats = inten  # or ones

        if label_path is not None and os.path.isfile(label_path):
            labels_u32 = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            semantic_raw = labels_u32 & 0xFFFF
        else:
            semantic_raw = np.zeros((scan.shape[0],), dtype=np.uint32)

        xyz_t   = torch.from_numpy(xyz).float()
        feats_t = torch.from_numpy(feats).float()
        sem_t = torch.from_numpy(semantic_raw).long()  # raw label (0~260...)

        if self.remap_labels and self.remap_lut is not None:
            # lut_device 에 올려둔 lut으로 인덱싱
            sem_t = sem_t.to(self.remap_lut.device)
            sem_t = self.remap_lut[sem_t]  # raw -> train (0~19)
            #sem_t = sem_t.cpu()

        return {
            "points": xyz_t,        # (N,3)
            "feats": feats_t,       # (N,1)
            "labels": sem_t, 
            "sequence": int(row["sequence"]),
            "frame": int(row["frame"]),
        }

def make_collate_fn(voxel_size, point_range, device="cpu"):
    def collate_fn(batch):
        batch_size = len(batch)

        all_v_feats      = []
        all_v_coords     = []
        all_point2voxel  = []
        all_point_labels = []
        all_point_masks  = []
        batch_metas      = []

        voxel_offset = 0

        spatial_shape = None

        for b_idx, sample in enumerate(batch):
            points = sample["points"].to(device)   # (N,3)
            feats  = sample["feats"].to(device)    # (N,C)
            labels = sample["labels"].to(device)   # (N,)

            vox = voxelize(
                points=points,
                feats=feats,
                voxel_size=voxel_size,
                point_range=point_range,
                batch_idx=b_idx,
            )
            v_feats      = vox["v_feats"]          # (M_i, C)
            v_coords     = vox["v_coords"]         # (M_i, 4)
            spatial_shape_i = vox["spatial_shape"] # [Z,Y,X]
            point2voxel_i   = vox["point2voxel"]   # (N_kept_i,)
            point_mask_i    = vox["point_mask"]    # (N,)

            if spatial_shape is None:
                spatial_shape = spatial_shape_i
            else:
                assert spatial_shape == spatial_shape_i, "spatial_shape mismatch between samples"

            labels_valid_i = labels[point_mask_i]  # (N_kept_i,)
            point2voxel_global = point2voxel_i + voxel_offset

            all_v_feats.append(v_feats)
            all_v_coords.append(v_coords)
            all_point2voxel.append(point2voxel_global)
            all_point_labels.append(labels_valid_i)
            all_point_masks.append(point_mask_i)

            voxel_offset += v_feats.size(0)

            batch_metas.append({
                "sequence": sample["sequence"],
                "frame": sample["frame"],
                "num_points_raw": points.size(0),
                "num_points_kept": labels_valid_i.size(0),
            })

        v_feats_batch  = torch.cat(all_v_feats, dim=0)      # (sum_i M_i, C)
        v_coords_batch = torch.cat(all_v_coords, dim=0)     # (sum_i M_i, 4)

        point2voxel_batch = torch.cat(all_point2voxel, dim=0)   # (sum_i N_kept_i,)
        labels_batch      = torch.cat(all_point_labels, dim=0)  # (sum_i N_kept_i,)

        batch_out = {
            "v_feats": v_feats_batch,
            "v_coords": v_coords_batch,
            "spatial_shape": spatial_shape,
            "point2voxel": point2voxel_batch,
            "point_labels": labels_batch,
            "batch_size": batch_size,
            "point_masks": all_point_masks,
            "metas": batch_metas,
        }
        return batch_out

    return collate_fn


def collate_fn_full(batch, voxel_size, device="cpu"):
    """
    batch: list of samples from SemanticKITTIDataset
           each sample: {"points", "feats", "labels", "sequence", "frame"}
    """
    all_v_feats = []
    all_v_coords = []
    all_point_labels = []
    all_point2voxel = []
    spatial_shapes = []
    batch_offsets = []

    voxel_offset = 0
    batch_size = len(batch)

    for b_idx, sample in enumerate(batch):
        points = sample["points"].to(device)  # (N,3)
        feats  = sample["feats"].to(device)   # (N,C)
        labels = sample["labels"].to(device)  # (N,)

        vox = voxelize_full(
            points=points,
            feats=feats,
            voxel_size=voxel_size,
            batch_idx=b_idx,
        )

        v_feats       = vox["v_feats"]       # (M,C)
        v_coords      = vox["v_coords"]      # (M,4)
        spatial_shape = vox["spatial_shape"] # [Z,Y,X]
        point2voxel   = vox["point2voxel"]   # (N,)
        point_mask    = vox["point_mask"]    # (N,)

        # SparseConvTensor용
        all_v_feats.append(v_feats)
        all_v_coords.append(v_coords)

        spatial_shapes.append(spatial_shape)

        # point label / mapping (전체 N 기준, 그냥 concat)
        all_point_labels.append(labels)          # (N,)
        all_point2voxel.append(point2voxel + voxel_offset)  # voxel index에 offset 부여

        voxel_offset += v_feats.size(0)

    v_feats_batch  = torch.cat(all_v_feats, dim=0)  # (sum_M, C)
    v_coords_batch = torch.cat(all_v_coords, dim=0) # (sum_M, 4)
    point_labels_batch = torch.cat(all_point_labels, dim=0)   # (sum_N,)
    point2voxel_batch  = torch.cat(all_point2voxel, dim=0)    # (sum_N,)

    batch_dict = {
        "v_feats": v_feats_batch,
        "v_coords": v_coords_batch,
        "spatial_shape": spatial_shapes[0],  # batch_size=1 가정
        "batch_size": batch_size,
        "point_labels": point_labels_batch,
        "point2voxel": point2voxel_batch,
    }
    return batch_dict