from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset

def create_dataloaders(X_train, Y_train,
                       X_val, Y_val,
                       batch_size):

    train_dataset = TimeSeriesDataset(X_train, Y_train)
    val_dataset   = TimeSeriesDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, val_loader
