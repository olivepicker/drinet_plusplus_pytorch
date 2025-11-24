import os
import glob
import torch
import pandas as pd

def voxelize(points, feats, voxel_size, point_range, batch_idx: int = 0):
    device = points.device
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)  # (3,)
    xyz_min = torch.tensor(point_range[0:3], dtype=torch.float32, device=device)  # (3,)
    xyz_max = torch.tensor(point_range[3:6], dtype=torch.float32, device=device)  # (3,)

    grid_size = torch.floor((xyz_max - xyz_min) / voxel_size).long()  # (3,)
    nx, ny, nz = grid_size[0].item(), grid_size[1].item(), grid_size[2].item()
    spatial_shape = [nz, ny, nx]

    coors_xyz = torch.floor((points - xyz_min) / voxel_size).long()  # (N,3)

    valid_mask = ((coors_xyz >= 0) & (coors_xyz < grid_size)).all(dim=1)  # (N,)
    if valid_mask.sum() == 0:
        raise ValueError("No points fall inside the given point_range.")

    points_kept = points[valid_mask]      # (N_kept,3)
    feats_kept = feats[valid_mask]        # (N_kept,C)
    coors_xyz = coors_xyz[valid_mask]     # (N_kept,3)
    coors_zxy = torch.stack(
        [coors_xyz[:, 2], coors_xyz[:, 1], coors_xyz[:, 0]], dim=1
    )  # (N_kept,3)

    unique_coords, point2voxel = torch.unique(
        coors_zxy, dim=0, return_inverse=True
    )

    M = unique_coords.size(0)
    C = feats_kept.size(1)

    v_feats = torch.zeros((M, C), dtype=feats_kept.dtype, device=device)

    index_expand = point2voxel.view(-1, 1).expand(-1, C)  # (N_kept, C)
    v_feats.scatter_add_(0, index_expand, feats_kept)     # (M,C)

    counts = torch.bincount(point2voxel, minlength=M).float().view(-1, 1)  # (M,1)
    v_feats = v_feats / counts

    batch_col = torch.full(
        (M, 1), batch_idx, dtype=torch.int32, device=device
    )
    v_coords = torch.cat(
        [batch_col, unique_coords.to(torch.int32)], dim=1
    )  # (M,4)

    return {
        "v_feats": v_feats,             # (M,C)
        "v_coords": v_coords,           # (M,4) [b,z,y,x]
        "spatial_shape": spatial_shape, # [Z,Y,X]
        "point2voxel": point2voxel,     # (N_kept,)
        "point_mask": valid_mask,       # (N,)
        "points_kept": points_kept,     # (N_kept,3) optional
    }

def build_semantickitti_index(root_dir: str, split_dict: dict) -> pd.DataFrame:
    records = []

    for split_name, seq_list in split_dict.items():
        for seq in seq_list:
            seq_str = f"{seq:02d}"
            velo_dir  = os.path.join(root_dir, "sequences", seq_str, "velodyne")
            label_dir = os.path.join(root_dir, "sequences", seq_str, "labels")

            velo_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
            if len(velo_files) == 0:
                print(f"[WARN] No velodyne files in {velo_dir}")
                continue

            has_label_dir = os.path.isdir(label_dir)

            for velo_path in velo_files:
                fname = os.path.basename(velo_path)  # '000123.bin'
                frame_id = int(os.path.splitext(fname)[0])

                if has_label_dir:
                    label_path = os.path.join(label_dir, f"{frame_id:06d}.label")
                    if not os.path.isfile(label_path):
                        label_path = None
                else:
                    label_path = None

                records.append({
                    "split": split_name,
                    "sequence": seq,
                    "frame": frame_id,
                    "velodyne_path": velo_path,
                    "label_path": label_path,
                })

    df = pd.DataFrame.from_records(records)
    return df