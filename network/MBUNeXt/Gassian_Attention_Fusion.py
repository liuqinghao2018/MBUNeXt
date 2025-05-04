import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class Basic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Basic_conv3d, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm3d(out_planes, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate_Fusion(nn.Module):
    def __init__(self, gate_channels, enhancement_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate_Fusion, self).__init__()
        self.gate_channels = gate_channels
        self.fusion_mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels * 4, gate_channels * 4 * enhancement_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels * 4 * enhancement_ratio, gate_channels*2),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels*2, gate_channels * 2 * enhancement_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels * 2 * enhancement_ratio, gate_channels)
        )
        self.branch_mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels*2, gate_channels*2 * enhancement_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels*2 * enhancement_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, model_1, model_2, model_3, model_4):
        fusion_channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                fusion_model = torch.cat((model_1, model_2, model_3, model_4), dim=1)
                fusion_avg_pool = F.avg_pool3d(fusion_model,
                                               (fusion_model.size(2), fusion_model.size(3), fusion_model.size(4)),
                                               stride=(
                                                fusion_model.size(2), fusion_model.size(3), fusion_model.size(4)))
                fusion_avg_pool = self.fusion_mlp(fusion_avg_pool).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                fusion_channel_att_raw = fusion_avg_pool
            elif pool_type == 'max':
                fusion_model = torch.cat((model_1, model_2, model_3, model_4), dim=1)
                fusion_max_pool = F.max_pool3d(fusion_model,
                                               (fusion_model.size(2), fusion_model.size(3), fusion_model.size(4)),
                                               stride=(
                                                fusion_model.size(2), fusion_model.size(3), fusion_model.size(4)))
                fusion_max_pool = self.fusion_mlp(fusion_max_pool).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                fusion_channel_att_raw = fusion_max_pool
            if fusion_channel_att_sum is None:
                fusion_channel_att_sum = fusion_channel_att_raw
            else:
                fusion_channel_att_sum = fusion_channel_att_sum + fusion_channel_att_raw

        # Gauss modulation
        mean = torch.mean(fusion_channel_att_sum).detach()
        std = torch.std(fusion_channel_att_sum).detach()
        fusion_scale = GaussProjection(fusion_channel_att_sum, mean, std).expand_as(model_1)

        return model_1 * fusion_scale, model_2 * fusion_scale, model_3 * fusion_scale, model_4 * fusion_scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate_Fusion(nn.Module):
    def __init__(self):
        super(SpatialGate_Fusion, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.fusion_spatial = Basic_conv3d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, model_1, model_2, model_3, model_4):
        fusion_model = torch.cat((model_1, model_2, model_3, model_4), dim=1)
        # Spatial Max Pooling
        fusion_max_pool = torch.max(fusion_model, 1)[0].unsqueeze(1)

        # Spatial Avg Pooling
        fusion_avg_pool = torch.mean(fusion_model, 1).unsqueeze(1)

        # Spatial Fusion Pooling
        fusion_pool = self.fusion_spatial(torch.cat((fusion_max_pool, fusion_avg_pool), dim=1))

        # Gauss modulation
        mean = torch.mean(fusion_pool).detach()
        std = torch.std(fusion_pool).detach()
        fusion_scale = GaussProjection(fusion_pool, mean, std)

        return model_1 * fusion_scale, model_2 * fusion_scale, model_3 * fusion_scale, model_4 * fusion_scale


class Gassian_Attention(nn.Module):
    def __init__(self, gate_channels, enhancement_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Gassian_Attention, self).__init__()
        self.ChannelGate_Fusion = ChannelGate_Fusion(gate_channels, enhancement_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate_Fusion = SpatialGate_Fusion()

    def forward(self, model_1, model_2, model_3, model_4):
        model_1_out, model_2_out, model_3_out, model_4_out = self.ChannelGate_Fusion(model_1, model_2, model_3, model_4)
        if not self.no_spatial:
            model_1_out, model_2_out, model_3_out, model_4_out = self.SpatialGate_Fusion(model_1_out, model_2_out, model_3_out, model_4_out)
        return model_1_out, model_2_out, model_3_out, model_4_out
