import os
import pandas as pd


def save_results(result_dict, filepath="results/experiment_results.csv"):
    """
    Appends experiment result to CSV file.
    Creates file if it doesn't exist.
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df_new = pd.DataFrame([result_dict])

    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(filepath, index=False)

    print(f"Results saved to {filepath}")
