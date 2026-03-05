import torch
from src.evaluation.metrics import smape
from src.training.early_stopping import EarlyStopping

def train_model(model,
                train_loader,
                val_loader,
                config,
                device):
    
    import os
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0

    early_stopper=EarlyStopping(
        patience=config["patience"],
        save_path=config["save_path"]

    )


    for epoch in range(config["epochs"]):

        model.train()
        train_loss = 0

        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)

            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_smape = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:

                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                output = model(X_batch)

                if config["residual"]:
                    last = X_batch[:, -1:, :]
                    output = last + output
                    Y_batch = last + Y_batch

                val_smape += smape(Y_batch, output).item()

        val_smape /= len(val_loader)

        print(f"Epoch {epoch+1} | "
              f"Train Loss {train_loss/len(train_loader):.4f} | "
              f"Val sMAPE {val_smape:.4f}")
        

        early_stopper(val_smape, model)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    if os.path.exists(config["save_path"]):
        model.load_state_dict(torch.load(config["save_path"]))
        best_score = early_stopper.best_score
    else:
        # Fallback if no model was ever saved
        best_score = val_smape
    
    
    return best_score
