import torch
import torch.nn as nn
from src.utils.positional import PositionalEncoding
class InformerEncoderOnly(nn.Module):
    def __init__(self, input_dim,pred_len,d_model,n_heads,enc_layers,dropout):
        super().__init__()

        self.pred_len=pred_len
        self.input_dim=input_dim


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

        # 🔹 Distillation layer (reduces sequence length)
        self.distill = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=2
        )

        self.output_proj = nn.Linear(d_model, pred_len * input_dim)

    def forward(self, x):

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        # Distillation
        x = x.transpose(1, 2)      # (B, D, T)
        x = self.distill(x)
        x = x.transpose(1, 2)      # (B, T/2, D)

        x = x[:, -1, :]            # Last compressed timestep

        out = self.output_proj(x)
        out = out.view(-1, self.pred_len, self.input_dim)

        return out
