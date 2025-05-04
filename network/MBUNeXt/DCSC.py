import torch
import torch.nn as nn
import torch.nn.functional as F

class DCSC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid_channels=int(dim/1)
        mid_groups=int(dim/2)

        self.conv0 = nn.Conv3d(dim, mid_channels, kernel_size=1)
        self.conv1_1 = nn.Conv3d(mid_channels, mid_channels, (1, 1, 7), padding=(0, 0, 3), groups=mid_groups)
        self.conv1_2 = nn.Conv3d(mid_channels, mid_channels, (1, 7, 1), padding=(0, 3, 0), groups=mid_groups)
        self.conv1_3 = nn.Conv3d(mid_channels, mid_channels, (7, 1, 1), padding=(3, 0, 0), groups=mid_groups)

        self.conv2_1 = nn.Conv3d(mid_channels, mid_channels, (1, 1, 11), padding=(0, 0, 5), groups=mid_groups)
        self.conv2_2 = nn.Conv3d(mid_channels, mid_channels, (1, 11, 1), padding=(0, 5, 0), groups=mid_groups)
        self.conv2_3 = nn.Conv3d(mid_channels, mid_channels, (11, 1, 1), padding=(5, 0, 0), groups=mid_groups)

        self.conv3_1 = nn.Conv3d(
            mid_channels, mid_channels, (1, 1, 21), padding=(0, 0, 10), groups=mid_groups)
        self.conv3_2 = nn.Conv3d(
            mid_channels, mid_channels, (1, 21, 1), padding=(0, 10, 0), groups=mid_groups)
        self.conv3_3 = nn.Conv3d(
            mid_channels, mid_channels, (21, 1, 1), padding=(10, 0, 0), groups=mid_groups)
        
        self.Conv_Out = nn.Sequential(
            nn.Conv3d(mid_channels, dim, (3, 3, 3), padding=1, groups=mid_groups),
            nn.InstanceNorm3d(dim),
            nn.PReLU(dim)
        )


    def forward(self, x):
        dwc_0 = self.conv0(x)

        dwc_1 = self.conv1_1(dwc_0)
        dwc_1 = self.conv1_2(dwc_1)
        dwc_1 = self.conv1_3(dwc_1)

        dwc_2 = self.conv2_1(dwc_1)
        dwc_2 = self.conv2_2(dwc_2)
        dwc_2 = self.conv2_3(dwc_2)

        dwc_3 = self.conv3_1(dwc_2)
        dwc_3 = self.conv3_2(dwc_3)
        dwc_3 = self.conv3_3(dwc_3)
        dwc = dwc_0 + dwc_1 + dwc_2 + dwc_3

        dwc = self.Conv_Out(dwc)

        return dwc + x