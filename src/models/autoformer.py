import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.positional import PositionalEncoding
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
    def forward(self, x):
        trend = F.avg_pool1d(
            x.transpose(1,2),
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        ).transpose(1,2)

        seasonal = x - trend
        return seasonal, trend


class AutoformerEncoderOnly(nn.Module):
    def __init__(self,
                 input_dim,
                 pred_len,
                 kernel_size,
                 d_model,
                 n_heads,
                 enc_layers,
                 dropout):

        super().__init__()

        assert d_model % n_heads == 0

        self.pred_len = pred_len
        self.input_dim = input_dim

        self.decomp = SeriesDecomposition(kernel_size)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=enc_layers
        )

        self.output_proj = nn.Linear(d_model,
                                     pred_len * input_dim)

    def forward(self, x):

        seasonal, _ = self.decomp(x)

        x = self.input_proj(seasonal)
        x=self.pos_enc(x)
        x = self.encoder(x)

        x = x[:, -1, :]

        out = self.output_proj(x)
        out = out.view(-1,
                       self.pred_len,
                       self.input_dim)

        return out
