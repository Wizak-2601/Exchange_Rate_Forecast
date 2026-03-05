import torch
from src.data.load_data import load_exchange_data
from src.data.window import create_windows
from src.data.dataloader import create_dataloaders
from src.models.build import build_model
from src.training.train import train_model
from src.evaluation.naive import compute_naive
from src.evaluation.arima import arima_multivariate
def run_experiment(config):

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    train_data, val_data,_ = load_exchange_data(
        univariate=config["univariate"]
    )


    X_train, Y_train = create_windows(
        train_data,
        config["seq_len"],
        config["pred_len"]
    )
    

    X_val, Y_val = create_windows(
        val_data,
        config["seq_len"],
        config["pred_len"]
    )

    if config["residual"]:
        Y_train = Y_train - X_train[:, -1:, :]
        Y_val   = Y_val - X_val[:, -1:, :]

    train_loader, val_loader = create_dataloaders(
        X_train, Y_train,
        X_val, Y_val,
        config["batch_size"]
    )

    config["input_dim"] = X_train.shape[-1]
    # print("SAVE PATH USED:", config["save_path"])

    model = build_model(config, device)
    # print("SAVE PATH USED:", config["save_path"])

    model_smape = train_model(
        model,
        train_loader,
        val_loader,
        config,
        device
    )
  
    naive_smape = compute_naive(X_val, Y_val, device)
    # arima_smape=arima_multivariate(train_data,val_data,order=(1,1,1))



    experiment_name = (
    f"{config['model_type']}"
    f"_s{config['seq_len']}"
    f"_p{config['pred_len']}"
    f"_h{config['n_heads']}"
    f"_k{config.get('kernel_size', 'NA')}"
    f"_d{config['dropout']}"
)




    return {
        "model_type": config["model_type"],
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
        "n_heads":config["n_heads"],
        "dropout":config["dropout"],
        "d_model":config["d_model"],
        "enc_layers":config["enc_layers"],
        "name":experiment_name,
        "model_smape": model_smape,
        "naive_smape": naive_smape,
        # "arima_smape": arima_smape,
        "beats_naive": model_smape < naive_smape
        # "beats_arima":bool(model_smape<arima_smape)
        
    }
