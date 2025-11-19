import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch.nn as nn

class SFEBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, bn_momentum=0.1):
        super().__init__()
        assert in_channels == out_channels, "SpconvSFEBlock assumes in_channels == out_channels for residual connection."

        mid_channels = in_channels
        
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, mid_channels, kernel_size=1, padding=0, bias=False, indice_key="subm1"
            ),
            nn.BatchNorm1d(mid_channels, momentum=bn_momentum),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            spconv.SubMConv3d(
                mid_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False, indice_key="subm2"
            ),
            nn.BatchNorm1d(mid_channels, momentum=bn_momentum),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            spconv.SubMConv3d(
                mid_channels, out_channels, kernel_size=1, padding=0, bias=False, indice_key="subm3"
            ),
            nn.BatchNorm1d(out_channels, momentum=bn_momentum),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: spconv.SparseConvTensor):
        out = self.net(x)
        out = out.replace_feature(out.features + x.features)
        out = out.replace_feature(self.lrelu(out.features))
        return out

class MultiScaleSparseProjection(nn.Module):
    def __init__(self, in_channels, out_channels=None, scales=[2,4,8,16]):
        super().__init__()
        self.scales = scales
        C = in_channels
        if out_channels is None:
            out_channels = C
        self.out_channels = out_channels

        self.gates = nn.ModuleList()
        self.projs = nn.ModuleList()

        for _ in self.scales:
            self.gates.append(
                nn.Sequential(
                    nn.Linear(C, C),
                    nn.BatchNorm1d(C),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(C, C),
                    nn.Sigmoid(),
                )
            )
            self.projs.append(
                nn.Sequential(
                    nn.Linear(C, out_channels),
                    nn.BatchNorm1d(out_channels),
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
    def __init__(self, in_channels, num_scales=4):
        super().__init__()
        self.attn = nn.ModuleList()
        for _ in range(num_scales):
            self.attn.append(
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.BatchNorm1d(in_channels),
                    nn.Sigmoid()
                )
            )
        self.head = nn.Linear(in_channels, in_channels)

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
    def __init__(self, in_channels, out_channels, scales):
        super().__init__()
        self.msp = MultiScaleSparseProjection(in_channels, out_channels, scales)
        self.amf = AttentiveMultiScaleFusion(in_channels, len(scales))
    
    def forward(self, x):
        x = self.msp(x)
        x = self.amf(x)

        return x

#TODO
class DRINetPlusPlus(nn.Module):
    def __init__(
            self, 
            in_channels=64,
            out_channels=64,
            scales = [2,4,8,16]
        ):
        super().__init__()
        self.sfe = SFEBlock(in_channels=in_channels, out_channels=out_channels)
        self.sgfe = SparseGeometryFeatureEnhancement(in_channels, out_channels, scales)

    def forward(self, x):
        pass