import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch.nn as nn

# class SparseResBlock(nn.Module):
#     def __init__(self, channels=64,kernel_size=3, bn_momentum=0.1):
#         super().__init__()
#         self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
#         self.bn1   = nn.BatchNorm1d(channels, momentum=bn_momentum)
#         self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
#         self.bn2   = nn.BatchNorm1d(channels, momentum=bn_momentum)
#         self.act   = nn.LeakyReLU(0.01, inplace=True)

#     def forward(self, x: spconv.SparseConvTensor):
#         identity = x.features

#         out = self.conv1(x)
#         out = out.replace_feature(self.act(self.bn1(out.features)))
#         out = self.conv2(out)
#         out = out.replace_feature(self.bn2(out.features))
#         out = out.replace_feature(self.act(out.features + identity))

#         return out

class SparseConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        bn_momentum=0.1
    ):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        self.act   = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: spconv.SparseConvTensor):
        out = self.conv1(x)
        out = out.replace_feature(self.act(self.bn1(out.features)))
        return out

class SFEBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, bn_momentum=0.1):
        super().__init__()
        self.block0 = nn.Sequential(
            SparseConvBlock(channels, channels, 1),
            SparseConvBlock(channels, channels, 3),
            SparseConvBlock(channels, channels, 1),
        )
        self.block1 = nn.Sequential(
            SparseConvBlock(channels, channels, 1),
            SparseConvBlock(channels, channels, 3),
            SparseConvBlock(channels, channels, 1),
        )
        self.block2 = nn.Sequential(
            SparseConvBlock(channels, channels, 1),
            SparseConvBlock(channels, channels, 3),
            SparseConvBlock(channels, channels, 1),
        )    
        self.block3 = nn.Sequential(
            SparseConvBlock(channels, channels, 1),
            SparseConvBlock(channels, channels, 3),
            SparseConvBlock(channels, channels, 1),
        )    

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: spconv.SparseConvTensor):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = out.replace_feature(out.features + x.features)
        out = out.replace_feature(self.lrelu(out.features))

        return out

class MultiScaleSparseProjection(nn.Module):
    def __init__(
        self, 
        channels, 
        scales=[2,4,8,16]
    ):
        super().__init__()
        self.scales = scales

        self.gates = nn.ModuleList()
        self.projs = nn.ModuleList()

        for _ in self.scales:
            self.gates.append(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(channels, channels),
                    nn.Sigmoid(),
                )
            )
            self.projs.append(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.LeakyReLU(inplace=True),
                )
            )

    def forward(self, x: spconv.SparseConvTensor):
        feats   = x.features      # (M,C)
        indices = x.indices      # (M,4) [b,z,y,x]
        device  = feats.device
        M, C    = feats.shape

        batch = indices[:, 0:1]    # (M,1)
        xyz   = indices[:, 1:]     # (M,3) [z,y,x]

        ms_outputs = []

        for i, s in enumerate(self.scales):
            coarse = torch.div(xyz, s, rounding_mode='floor')  # (M,3)
            key = torch.cat([batch, coarse], dim=1)            # (M,4)

            unique_key, inv = torch.unique(
                key, dim=0, return_inverse=True
            )
            K = unique_key.size(0)

            V = feats.new_zeros((K, C))      # (K,C)
            index_expand = inv.view(-1, 1).expand(-1, C)
            V.scatter_add_(0, index_expand, feats)

            counts = torch.bincount(inv, minlength=K).float().to(device)
            counts = counts.view(-1, 1)
            V = V / counts
            V_up = V[inv]
            Os = feats - V_up
            Os = Os * feats

            Os_proj = self.projs[i](Os)      # (M,C_out)
            ms_outputs.append(Os_proj)

        ms_feat = torch.stack(ms_outputs, dim=1)  # (M,S,C_out)
        return ms_feat

class AttentiveMultiScaleFusion(nn.Module):
    def __init__(self, channels, num_scales=4):
        super().__init__()
        self.attn = nn.ModuleList()
        for _ in range(num_scales):
            self.attn.append(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.Sigmoid()
                )
            )
        self.head = nn.Linear(channels, channels)

    def forward(self, x):
        sum_x = torch.sum(x, dim=1)
        B, C = sum_x.size()
        
        L = []
        for idx, attn in enumerate(self.attn):
            o = attn(sum_x)
            o = o * x[:,idx,:]
            L.append(o)

        out = torch.sum(torch.stack(L, dim=-1), dim=-1)

        return out

class SparseGeometryFeatureEnhancement(nn.Module):
    def __init__(self, channels, scales):
        super().__init__()
        self.msp = MultiScaleSparseProjection(channels, scales)
        self.amf = AttentiveMultiScaleFusion(channels, len(scales))
    
    def forward(self, x):
        x = self.msp(x)
        x = self.amf(x)

        return x

class DRINetBlock(nn.Module):
    def __init__(
            self, 
            channels=64,
            num_classes=20,
            scales=[2, 4, 8, 16],
        ):
        super().__init__()
        self.sfe        = SFEBlock(channels)
        self.sgfe       = SparseGeometryFeatureEnhancement(channels, scales)
        self.aux_head   = nn.Linear(channels, num_classes)
        self.block_head = nn.Linear(channels, num_classes)
        
    def forward(self, x: spconv.SparseConvTensor):
        V = self.sfe(x)
        aux_voxel_logits = self.aux_head(V.features)  # (M, num_classes)
        
        fused  = self.sgfe(V)                         # (M, out_channels)
        x_new  = V.replace_feature(fused)
        
        voxel_logits = self.block_head(x_new.features)  # (M, num_classes)

        return x_new, voxel_logits, aux_voxel_logits
        
class DRINetPlusPlus(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=64, 
        num_blocks=4, 
        num_classes=20, 
        scales=[2,4,8,16],
        aux_loss_ratio=0.4
    ):
        super().__init__()
        self.num_blocks  = num_blocks
        self.num_classes = num_classes

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            DRINetBlock(
                out_channels,
                num_classes,
                scales
            )
            for _ in range(num_blocks)
        ])

        self.final_mlp = nn.Linear(num_blocks * num_classes, num_classes)
        self.aux_loss_ratio = aux_loss_ratio

    def forward(self, x: spconv.SparseConvTensor, point2voxel, point_labels=None):
        F_sparse = self.stem(x)
        N_valid = point2voxel.size(0)

        if point_labels is not None:
            assert point_labels.size(0) == N_valid, \
                f"point_labels({point_labels.size(0)}) and point2voxel({N_valid}) must have same length."

        block_outputs = []
        aux_loss = 0.0
        F_cur = F_sparse
        
        for b, block in enumerate(self.blocks):
            F_cur, voxel_logits_b, aux_voxel_logits_b = block(F_cur)  # voxel_logits_b: (M,C)

            voxel_idx = point2voxel                   # (N_valid,)
            point_logits_b = voxel_logits_b[voxel_idx]   # (N_valid, num_classes)
            block_outputs.append(point_logits_b)

            if self.training and point_labels is not None:
                aux_point_logits_b = aux_voxel_logits_b[voxel_idx]   # (N_valid,C)
                aux_loss += F.cross_entropy(aux_point_logits_b, point_labels, ignore_index=0)

        L = torch.stack(block_outputs, dim=1)   # (N_valid, B, num_classes)
        N_valid, B, C = L.shape
        L_flat = L.reshape(N_valid, B * C)
        final_logits_valid = self.final_mlp(L_flat)  # (N_valid, num_classes)

        if point_labels is not None:
            final_loss = F.cross_entropy(final_logits_valid, point_labels, ignore_index=0)
            total_loss = final_loss + self.aux_loss_ratio * aux_loss
            return final_logits_valid, total_loss

        return final_logits_valid