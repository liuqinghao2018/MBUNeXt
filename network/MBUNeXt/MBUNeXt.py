import torch
import torch.nn as nn
import torch.nn.functional as F

from MBUNeXt.network.MBUNeXt.Transformer import TransformerModel
from MBUNeXt.network.MBUNeXt.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from MBUNeXt.network.MBUNeXt.Gassian_Attention_Fusion import Gassian_Attention
from MBUNeXt.network.MBUNeXt.DCSC import DCSC

class MBUNeXt(nn.Module):
    def __init__(
            self,
            in_channel=1,
            out_channel=3,
            training=True,
            img_dim=128,
            patch_dim=8,
            num_channels=4,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=4096,
            dropout_rate=0.2,
            attn_dropout_rate=0.0,
            positional_encoding_type="learned"):
        super(MBUNeXt, self).__init__()
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.training = training

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dorp_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches  
        self.flatten_dim = 128 * num_channels
        # Encoder_Stage
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.PReLU(16),
        )
        self.attention_fusion1 = Gassian_Attention(gate_channels=16, enhancement_ratio=16, pool_types=['avg', 'max'], no_spatial=True)

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.PReLU(32),
        )
        self.attention_fusion2 = Gassian_Attention(gate_channels=32, enhancement_ratio=8, pool_types=['avg', 'max'], no_spatial=True)

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.PReLU(64),
        )
        self.attention_fusion3 = Gassian_Attention(gate_channels=64, enhancement_ratio=4, pool_types=['avg', 'max'], no_spatial=True)

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.PReLU(128),
        )
        self.attention_fusion4 = Gassian_Attention(gate_channels=128, enhancement_ratio=2, pool_types=['avg', 'max'], no_spatial=True)

        # Transformer_Stage
        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dorp_rate)
        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dorp_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        # Decoder_Stage
        self.decoder_stage1 = nn.Sequential(
            nn.ConvTranspose3d(128, 256, 2, 2, groups=4),
            nn.InstanceNorm3d(256, affine=True),
            nn.PReLU(256),
        )

        self.sc2 = DCSC(256)

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(256 + 256, 256, 3, 1, padding=1, groups=4),
            nn.InstanceNorm3d(256, affine=True),
            nn.PReLU(256),
        )

        self.sc3 = DCSC(128)

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(128 + 128, 128, 3, 1, padding=1, groups=4),
            nn.InstanceNorm3d(128, affine=True),
            nn.PReLU(128),
        )

        self.sc4 = DCSC(64)
        # self.sc4 = DCSC(32)

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, 1, padding=1 , groups=4),
            nn.InstanceNorm3d(64, affine=True),
            nn.PReLU(64),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32, affine=True),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64, affine=True),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128, affine=True),
            nn.PReLU(128)
        )

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.InstanceNorm3d(128, affine=True),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64, affine=True),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(

            nn.Conv3d(64, 32, 3, 1, padding=1, groups=2),
            nn.InstanceNorm3d(32, affine=True),
            nn.PReLU(32),
            nn.Conv3d(32, 16, 3, 1, padding=1, groups=2),
            nn.InstanceNorm3d(16, affine=True),
            nn.PReLU(16)
        )

        self.map = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
        )


    def forward(self, inputs):
        modal_1 = inputs[:, 0:1, :, :, :]
        modal_2 = inputs[:, 1:2, :, :, :]
        modal_3 = inputs[:, 2:3, :, :, :]
        modal_4 = inputs[:, 3:4, :, :, :]
        modal_1_long_range1 = self.encoder_stage1(modal_1)
        modal_2_long_range1 = self.encoder_stage1(modal_2)
        modal_3_long_range1 = self.encoder_stage1(modal_3)
        modal_4_long_range1 = self.encoder_stage1(modal_4)
        modal_1_fusion1, modal_2_fusion1, modal_3_fusion1, modal_4_fusion1 = self.attention_fusion1(modal_1_long_range1,
                                                                                                    modal_2_long_range1,
                                                                                                    modal_3_long_range1,
                                                                                                    modal_4_long_range1)
        modal_1_long_range1 = modal_1_fusion1 + modal_1_long_range1
        modal_2_long_range1 = modal_2_fusion1 + modal_2_long_range1
        modal_3_long_range1 = modal_3_fusion1 + modal_3_long_range1
        modal_4_long_range1 = modal_4_fusion1 + modal_4_long_range1
        modal_1_long_range1 = F.dropout(modal_1_long_range1, self.dorp_rate, self.training)
        modal_2_long_range1 = F.dropout(modal_2_long_range1, self.dorp_rate, self.training)
        modal_3_long_range1 = F.dropout(modal_3_long_range1, self.dorp_rate, self.training)
        modal_4_long_range1 = F.dropout(modal_4_long_range1, self.dorp_rate, self.training)
        modal_1_short_range1 = self.down_conv1(modal_1_long_range1)
        modal_2_short_range1 = self.down_conv1(modal_2_long_range1)
        modal_3_short_range1 = self.down_conv1(modal_3_long_range1)
        modal_4_short_range1 = self.down_conv1(modal_4_long_range1)

        # encode_stage_2
        modal_1_long_range2 = self.encoder_stage2(modal_1_short_range1)
        modal_2_long_range2 = self.encoder_stage2(modal_2_short_range1)
        modal_3_long_range2 = self.encoder_stage2(modal_3_short_range1)
        modal_4_long_range2 = self.encoder_stage2(modal_4_short_range1)
        modal_1_fusion2, modal_2_fusion2, modal_3_fusion2, modal_4_fusion2 = self.attention_fusion2(modal_1_long_range2,
                                                                                                    modal_2_long_range2,
                                                                                                    modal_3_long_range2,
                                                                                                    modal_4_long_range2)
        modal_1_long_range2 = modal_1_fusion2 + modal_1_short_range1
        modal_2_long_range2 = modal_2_fusion2 + modal_2_short_range1
        modal_3_long_range2 = modal_3_fusion2 + modal_3_short_range1
        modal_4_long_range2 = modal_4_fusion2 + modal_4_short_range1
        modal_1_long_range2 = F.dropout(modal_1_long_range2, self.dorp_rate, self.training)
        modal_2_long_range2 = F.dropout(modal_2_long_range2, self.dorp_rate, self.training)
        modal_3_long_range2 = F.dropout(modal_3_long_range2, self.dorp_rate, self.training)
        modal_4_long_range2 = F.dropout(modal_4_long_range2, self.dorp_rate, self.training)

        modal_1_short_range2 = self.down_conv2(modal_1_long_range2)
        modal_2_short_range2 = self.down_conv2(modal_2_long_range2)
        modal_3_short_range2 = self.down_conv2(modal_3_long_range2)
        modal_4_short_range2 = self.down_conv2(modal_4_long_range2)

        # encode_stage_3
        modal_1_long_range3 = self.encoder_stage3(modal_1_short_range2)
        modal_2_long_range3 = self.encoder_stage3(modal_2_short_range2)
        modal_3_long_range3 = self.encoder_stage3(modal_3_short_range2)
        modal_4_long_range3 = self.encoder_stage3(modal_4_short_range2)
        modal_1_fusion3, modal_2_fusion3, modal_3_fusion3, modal_4_fusion3 = self.attention_fusion3(modal_1_long_range3,
                                                                                                    modal_2_long_range3,
                                                                                                    modal_3_long_range3,
                                                                                                    modal_4_long_range3)
        modal_1_long_range3 = modal_1_fusion3 + modal_1_short_range2
        modal_2_long_range3 = modal_2_fusion3 + modal_2_short_range2
        modal_3_long_range3 = modal_3_fusion3 + modal_3_short_range2
        modal_4_long_range3 = modal_4_fusion3 + modal_4_short_range2
        modal_1_long_range3 = F.dropout(modal_1_long_range3, self.dorp_rate, self.training)
        modal_2_long_range3 = F.dropout(modal_2_long_range3, self.dorp_rate, self.training)
        modal_3_long_range3 = F.dropout(modal_3_long_range3, self.dorp_rate, self.training)
        modal_4_long_range3 = F.dropout(modal_4_long_range3, self.dorp_rate, self.training)

        modal_1_short_range3 = self.down_conv3(modal_1_long_range3)
        modal_2_short_range3 = self.down_conv3(modal_2_long_range3)
        modal_3_short_range3 = self.down_conv3(modal_3_long_range3)
        modal_4_short_range3 = self.down_conv3(modal_4_long_range3)

        # encode_stage_4
        modal_1_long_range4 = self.encoder_stage4(modal_1_short_range3)
        modal_2_long_range4 = self.encoder_stage4(modal_2_short_range3)
        modal_3_long_range4 = self.encoder_stage4(modal_3_short_range3)
        modal_4_long_range4 = self.encoder_stage4(modal_4_short_range3)
        modal_1_fusion4, modal_2_fusion4, modal_3_fusion4, modal_4_fusion4 = self.attention_fusion4(modal_1_long_range4,
                                                                                                    modal_2_long_range4,
                                                                                                    modal_3_long_range4,
                                                                                                    modal_4_long_range4)
        modal_1_long_range4 = modal_1_fusion4 + modal_1_short_range3
        modal_2_long_range4 = modal_2_fusion4 + modal_2_short_range3
        modal_3_long_range4 = modal_3_fusion4 + modal_3_short_range3
        modal_4_long_range4 = modal_4_fusion4 + modal_4_short_range3
        modal_1_long_range4 = F.dropout(modal_1_long_range4, self.dorp_rate, self.training)
        modal_2_long_range4 = F.dropout(modal_2_long_range4, self.dorp_rate, self.training)
        modal_3_long_range4 = F.dropout(modal_3_long_range4, self.dorp_rate, self.training)
        modal_4_long_range4 = F.dropout(modal_4_long_range4, self.dorp_rate, self.training)

        # transformer_stage
        multimodal_transformer_1 = torch.cat([modal_1_long_range4, modal_2_long_range4,
                                    modal_3_long_range4, modal_4_long_range4], dim=1)
        x = multimodal_transformer_1.permute(0, 2, 3, 4, 1).contiguous()
        multimodal_transformer_2 = x.view(x.size(0), -1, self.embedding_dim)
        multimodal_transformer_3 = self.position_encoding(multimodal_transformer_2)
        multimodal_transformer_3 = self.pe_dropout(multimodal_transformer_3)
        multimodal_transformer_4, intmd_x = self.transformer(multimodal_transformer_3)
        multimodal_transformer_4 = self.pre_head_ln(multimodal_transformer_4)
        multimodal_transformer = self._reshape_output(multimodal_transformer_4)
        multimodal_transformer = multimodal_transformer_1 + multimodal_transformer  
        multimodal_transformer = self.Enblock8_1(multimodal_transformer)
        multimodal_transformer = self.Enblock8_2(multimodal_transformer)

        ###
        # decode_stage_1
        multimodal_short_range4 = self.decoder_stage1(multimodal_transformer)

        ###
        # decode_stage_2
        multimodal_sc_2 = torch.cat([modal_1_long_range3, modal_2_long_range3,
                                    modal_3_long_range3, modal_4_long_range3], dim=1)
        multimodal_sc_2 = self.sc2(multimodal_sc_2)
        # skip_connection_2
        multimodal_long_range5 = self.decoder_stage2(
            torch.cat([multimodal_short_range4, multimodal_sc_2], dim=1)) + multimodal_short_range4

        multimodal_short_range5 = self.up_conv1(multimodal_long_range5)

        ###
        # decode_stage_3
        multimodal_sc_3 = torch.cat([modal_1_long_range2, modal_2_long_range2,
                                    modal_3_long_range2, modal_4_long_range2], dim=1)
        multimodal_sc_3 = self.sc3(multimodal_sc_3)
        # skip_connection_3
        multimodal_long_range6 = self.decoder_stage3(
            torch.cat([multimodal_short_range5, multimodal_sc_3], dim=1)) + multimodal_short_range5

        multimodal_short_range6 = self.up_conv2(multimodal_long_range6)

        ###
        # decode_stage_4
        multimodal_sc_4 = torch.cat([modal_1_long_range1, modal_2_long_range1,
                                    modal_3_long_range1, modal_4_long_range1], dim=1)
        multimodal_sc_4 = self.sc4(multimodal_sc_4)
        # skip_connection_4
        multimodal_long_range7 = self.decoder_stage4(
            torch.cat([multimodal_short_range6, multimodal_sc_4], dim=1)) + multimodal_short_range6

        multimodal_short_range7 = self.up_conv3(multimodal_long_range7)

        # map
        output = self.map(multimodal_short_range7)

        return output

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(512 // 4, affine=True)
        self.relu1 = nn.PReLU(512 // 4)

        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(512 // 4, affine=True)
        self.relu2 = nn.PReLU(512 // 4)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(512 // 4, affine=True)
        self.relu1 = nn.PReLU(512 // 4)
        self.bn2 = nn.InstanceNorm3d(512 // 4, affine=True)
        self.relu2 = nn.PReLU(512 // 4)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1