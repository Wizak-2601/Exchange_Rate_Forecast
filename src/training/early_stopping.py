import os
import torch


class EarlyStopping:
    def __init__(self, patience=5, mode="min", save_path="results/saved_models/model.pth", verbose=False):
        self.patience = patience
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self._save_model(model)

        elif (self.mode == "min" and score < self.best_score) or \
             (self.mode == "max" and score > self.best_score):

            self.best_score = score
            self.counter = 0
            self._save_model(model)

        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

    def _save_model(self, model):

    # Convert to absolute path based on current working directory
        abs_path = os.path.abspath(self.save_path)

        dir_path = os.path.dirname(abs_path)

        if dir_path != "":
            os.makedirs(dir_path, exist_ok=True)

        torch.save(model.state_dict(), abs_path)

        self.save_path = abs_path
