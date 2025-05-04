import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, 512))  # 8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings


class ConVPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(ConVPositionalEncoding, self).__init__()
        self.nc = 512
        self.proj_q = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.rpe_table = nn.Conv3d(self.nc, self.nc,
                                   kernel_size=3, stride=1, padding=1, groups=self.nc)

    def forward(self, x, position_ids=None):
        B, C, D, H, W = x.size()
        q = self.proj_q(x)
        # q[B, 512, 16, 16, 16]
        residual_lepe = self.rpe_table(q.reshape(B, C, D, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                         D * H * W)

        return x + residual_lepe
