import torch
import torch.nn as nn
from src.utils.positional import PositionalEncoding


class VanillaTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 pred_len,
                 d_model,
                 n_heads,
                 enc_layers,
                 dropout):

        super().__init__()

        self.pred_len = pred_len
        self.input_dim = input_dim

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

        self.output_proj = nn.Linear(d_model, pred_len * input_dim)

    def forward(self, x):

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        x = x[:, -1, :]
        out = self.output_proj(x)
        out = out.view(-1, self.pred_len, self.input_dim)

        return out
