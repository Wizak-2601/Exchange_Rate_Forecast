from .informer import InformerEncoderOnly
from .autoformer import AutoformerEncoderOnly
from.transformer import VanillaTransformer
def build_model(config, device):

    if config["model_type"] == "informer":
        model = InformerEncoderOnly(
            input_dim=config["input_dim"],
            pred_len=config["pred_len"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            enc_layers=config["enc_layers"],
            dropout=config["dropout"]
        )

    elif config["model_type"] == "autoformer":
        model = AutoformerEncoderOnly(
            input_dim=config["input_dim"],
            pred_len=config["pred_len"],
            kernel_size=config["kernel_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            enc_layers=config["enc_layers"],
            dropout=config["dropout"]
        )
    elif config["model_type"]== "transformer":
        model = VanillaTransformer(
            input_dim=config["input_dim"],
            pred_len=config["pred_len"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            enc_layers=config["enc_layers"],
            dropout=config["dropout"]
        )

    else:
        raise ValueError("Unknown model type")

    return model.to(device)
