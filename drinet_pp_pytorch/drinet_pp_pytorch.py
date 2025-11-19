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



#TODO
class DRINetPlusPlus(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        pass