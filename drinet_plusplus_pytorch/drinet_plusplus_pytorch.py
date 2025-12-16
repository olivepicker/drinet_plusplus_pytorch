import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch.nn as nn

from utils import lovasz_softmax, make_voxel_labels_majority

class SparseConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        bn_momentum=0.1
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        self.act   = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: spconv.SparseConvTensor):
        out = self.conv1(x)
        out = out.replace_feature(self.act(self.bn1(out.features)))
        return out

class SparseResNetBottleneck(nn.Module):
    def __init__(self, channels, kernel_size=3, bn_momentum=0.1):
        super().__init__()
        self.conv1 = SparseConvBlock(channels, channels, 1, bn_momentum)
        self.conv2 = SparseConvBlock(channels, channels, kernel_size, bn_momentum)
        self.conv3 = SparseConvBlock(channels, channels, 1, bn_momentum)
        
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.act(out.features))
        
        return out

class SFEBlock(nn.Module):
    def __init__(self, channels=64, num_layers=4, bn_momentum=0.1):
        super().__init__()
        self.layers = nn.Sequential(*[
            SparseResNetBottleneck(channels, 3, bn_momentum)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        return self.layers(x)

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

            gate = self.gates[i](Os)
            gated = gate * feats

            Os_proj = self.projs[i](gated)    # (M,C)
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
        out = self.head(out)
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
        
    def forward(self, x: spconv.SparseConvTensor):
        V = self.sfe(x)
        aux_voxel_logits = self.aux_head(V.features)  # (M, num_classes)
        
        fused  = self.sgfe(V)                         # (M, out_channels)
        x_new  = V.replace_feature(fused)
        
        return x_new, aux_voxel_logits
        
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

        self.final_mlp = nn.Linear(num_blocks * out_channels, num_classes)
        self.aux_loss_ratio = aux_loss_ratio

    def forward(self, x: spconv.SparseConvTensor, point2voxel, point_labels=None):
        F_sparse = self.stem(x)
        N_valid = point2voxel.size(0)

        if point_labels is not None:
            assert point_labels.size(0) == N_valid, \
                f"point_labels({point_labels.size(0)}) and point2voxel({N_valid}) must have same length."

        feature_list = []
        aux_loss = 0.0
        F_cur = F_sparse
        
        for b, block in enumerate(self.blocks):
            F_cur, aux_voxel_logits_b = block(F_cur)

            point_features_b = F_cur.features[point2voxel]
            feature_list.append(point_features_b)

            if self.training and point_labels is not None:
                voxel_labels = make_voxel_labels_majority(
                    point2voxel=point2voxel,
                    point_labels=point_labels,
                    num_voxels=aux_voxel_logits_b.size(0),
                    ignore_index=0,
                )

                aux_loss += F.cross_entropy(
                    aux_voxel_logits_b,
                    voxel_labels,
                    ignore_index=0,
                )
                
        L = torch.cat(feature_list, dim=1)
        final_logits_valid = self.final_mlp(L)

        if point_labels is not None:
            ce_loss = F.cross_entropy(
                final_logits_valid,
                point_labels,
                ignore_index=0,
            )

            probs = F.softmax(final_logits_valid, dim=1)
            lovasz_loss = lovasz_softmax(
                probs,
                point_labels,
                ignore_index=0,
            )

            final_loss = ce_loss + lovasz_loss
            total_loss = final_loss + self.aux_loss_ratio * aux_loss
            return final_logits_valid, total_loss

        return final_logits_valid