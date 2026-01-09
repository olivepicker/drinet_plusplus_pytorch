import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import voxelize_full

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
    def __init__(
        self,
        df,
        remap_labels: bool = True,
        augment: bool = False,

        rot_range = (-np.pi, np.pi),
        scale_range = (0.95, 1.05),
        max_dropout_ratio: float = 0.1,
        enable_flip: bool = True,

        enable_translation: bool = True,
        trans_p: float = 0.5,
        trans_max_xy: float = 1.0,
        trans_max_z: float = 0.1,

        enable_xyz_jitter: bool = True,
        xyz_jitter_p: float = 0.5,
        xyz_jitter_sigma: float = 0.015,

        enable_intensity_aug: bool = True,
        intensity_p: float = 0.5,
        inten_scale_min: float = 0.9,
        inten_scale_max: float = 1.1,
        inten_noise_std: float = 0.02,
        inten_clamp_min = None,
        inten_clamp_max = None,
    ):
        self.df = df.reset_index(drop=True)
        self.remap_labels = remap_labels
        self.augment = augment

        self.rot_range = rot_range
        self.scale_range = scale_range
        self.max_dropout_ratio = max_dropout_ratio
        self.enable_flip = enable_flip

        self.enable_translation = enable_translation
        self.trans_p = trans_p
        self.trans_max_xy = trans_max_xy
        self.trans_max_z = trans_max_z

        self.enable_xyz_jitter = enable_xyz_jitter
        self.xyz_jitter_p = xyz_jitter_p
        self.xyz_jitter_sigma = xyz_jitter_sigma

        self.enable_intensity_aug = enable_intensity_aug
        self.intensity_p = intensity_p
        self.inten_scale_min = inten_scale_min
        self.inten_scale_max = inten_scale_max
        self.inten_noise_std = inten_noise_std
        self.inten_clamp_min = inten_clamp_min
        self.inten_clamp_max = inten_clamp_max

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
        xyz  = scan[:, :3]       # (N,3)
        feats = scan             # (N,4) = [x,y,z,intensity]

        if label_path is not None and os.path.isfile(label_path):
            labels_u32 = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            semantic_raw = labels_u32 & 0xFFFF
        else:
            semantic_raw = np.zeros((scan.shape[0],), dtype=np.uint32)

        xyz_t   = torch.from_numpy(xyz).float()
        feats_t = torch.from_numpy(feats).float()
        sem_t   = torch.from_numpy(semantic_raw).long()

        if self.augment:
            xyz_t, feats_t, sem_t = self._augment_points(xyz_t, feats_t, sem_t)

        if self.remap_labels and self.remap_lut is not None:
            sem_t = sem_t.to(self.remap_lut.device)
            sem_t = self.remap_lut[sem_t]

        return {
            "points": xyz_t,
            "feats": feats_t,
            "labels": sem_t,
            "sequence": int(row["sequence"]),
            "frame": int(row["frame"]),
        }

    def _augment_points(self, points, feats, labels):
        N, C = feats.shape
        
        # Clone for safety
        points = points.clone()
        feats  = feats.clone()
        labels = labels.clone()

        # Flip y
        if self.enable_flip and np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]
            feats[:, 1]  = -feats[:, 1]

        # Scaling
        if self.scale_range is not None:
            s_min, s_max = self.scale_range
            scale = np.random.uniform(s_min, s_max)
            points = points * scale
            feats[:, :3] = feats[:, :3] * scale

        # Rotation
        if self.rot_range is not None:
            rot_min, rot_max = self.rot_range
            angle = np.random.uniform(rot_min, rot_max)
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            rot_mat = points.new_tensor([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0],
            ])
            points = points @ rot_mat.T
            feats[:, :3] = feats[:, :3] @ rot_mat.T

        # XYZ Jitter
        if self.enable_xyz_jitter and np.random.rand() < self.xyz_jitter_p:
            noise = torch.randn_like(points) * float(self.xyz_jitter_sigma)
            points = points + noise
            feats[:, :3] = feats[:, :3] + noise

        # Translation
        if self.enable_translation and np.random.rand() < self.trans_p:
            dx = np.random.uniform(-self.trans_max_xy, self.trans_max_xy)
            dy = np.random.uniform(-self.trans_max_xy, self.trans_max_xy)
            dz = np.random.uniform(-self.trans_max_z,  self.trans_max_z)
            shift = points.new_tensor([dx, dy, dz])
            
            points = points + shift
            feats[:, :3] = feats[:, :3] + shift

        # Intensity
        if C >= 4 and self.enable_intensity_aug and np.random.rand() < self.intensity_p:
            inten = feats[:, 3]
            scale = np.random.uniform(self.inten_scale_min, self.inten_scale_max)
            inten = inten * float(scale)
            inten = inten + torch.randn_like(inten) * float(self.inten_noise_std)
            
            # Clamp
            cmin = float(self.inten_clamp_min) if self.inten_clamp_min is not None else float("-inf")
            cmax = float(self.inten_clamp_max) if self.inten_clamp_max is not None else float("inf")
            inten = inten.clamp(min=cmin, max=cmax)
            
            feats[:, 3] = inten

        # Dropout
        if self.max_dropout_ratio is not None and self.max_dropout_ratio > 0.0:
            drop_ratio = np.random.uniform(0.0, self.max_dropout_ratio)
            if drop_ratio > 0:
                keep_mask = torch.rand(N, device=points.device) > drop_ratio
                if keep_mask.sum() < 10:
                    keep_mask = torch.ones(N, dtype=torch.bool, device=points.device)
                
                points = points[keep_mask]
                feats  = feats[keep_mask]
                labels = labels[keep_mask]

        return points, feats, labels

def collate_fn_full(batch, voxel_size, device="cpu"):
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

        all_v_feats.append(v_feats)
        all_v_coords.append(v_coords)

        spatial_shapes.append(spatial_shape)

        all_point_labels.append(labels)
        all_point2voxel.append(point2voxel + voxel_offset)

        voxel_offset += v_feats.size(0)

    v_feats_batch  = torch.cat(all_v_feats, dim=0)  # (sum_M, C)
    v_coords_batch = torch.cat(all_v_coords, dim=0) # (sum_M, 4)
    point_labels_batch = torch.cat(all_point_labels, dim=0)   # (sum_N,)
    point2voxel_batch  = torch.cat(all_point2voxel, dim=0)    # (sum_N,)

    batch_dict = {
        "v_feats": v_feats_batch,
        "v_coords": v_coords_batch,
        "spatial_shape": spatial_shapes[0],
        "batch_size": batch_size,
        "point_labels": point_labels_batch,
        "point2voxel": point2voxel_batch,
    }
    return batch_dict