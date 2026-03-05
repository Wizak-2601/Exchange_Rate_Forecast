import os
import yaml

# Get project root automatically
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


def load_config(path, overrides=None):

    # Convert config path to absolute
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"{path} is empty or invalid YAML.")

    flat_config = {}

    for section in ["data", "model", "training", "misc"]:
        if section in config:
            flat_config.update(config[section])

    flat_config["model_type"] = config.get("model_type")

    if overrides:
        flat_config.update(overrides)

    # 🔥 AUTO-GENERATE SAVE NAME

    model_name = flat_config["model_type"]
    seq_len = flat_config["seq_len"]
    pred_len = flat_config["pred_len"]

    filename = f"{model_name}_s{seq_len}_p{pred_len}"

    # Optional components
    if "n_heads" in flat_config:
        filename += f"_h{flat_config['n_heads']}"

    if "kernel_size" in flat_config:
        filename += f"_k{flat_config['kernel_size']}"

    if flat_config.get("univariate"):
        filename += "_uni"

    if flat_config.get("use_lags"):
        filename += "_lags"

    filename += ".pth"

    flat_config["save_path"] = os.path.join(
        PROJECT_ROOT,
        "results",
        "saved_models",
        filename
    )


    return flat_config
