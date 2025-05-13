import os
import glob
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Creating a data frame containing 2BCNN models info"
)
parser.add_argument(
    "--model_dir", type=str, required=True,
    help="Absolute path to the directory where models are stored."
)

args = parser.parse_args()
model_dir = args.model_dir

# Directory where the checkpoint files are saved.
CHECKPOINTS_DIR = model_dir

# Use glob to get all checkpoint files matching the pattern.
# trial checkpoints are named like "checkpoint_trial_*.pth"
checkpoint_pattern = os.path.join(CHECKPOINTS_DIR, "checkpoint_trial_*.pth")
checkpoint_files = glob.glob(checkpoint_pattern)

# List to hold information from each checkpoint.
trial_info_list = []

for file in checkpoint_files:
    try:
        # Load the checkpoint on CPU.
        checkpoint = torch.load(file, map_location="cpu")
        
        # Extract basic information.
        trial_info = {
            "checkpoint_file": os.path.basename(file),
            "trial_number": checkpoint.get('trial_number', None),
            "best_epoch": checkpoint.get('epoch', None),
            "val_loss": checkpoint.get('val_loss', None),
            "RMSE": checkpoint.get('RMSE', None),
            "train_loss_history": ','.join(map(str, checkpoint.get('train_loss_history', []))),
            "val_loss_history": ','.join(map(str, checkpoint.get('val_loss_history', [])))
        }
        
        # Extract hyperparameters (which is a dictionary) and update trial_info.
        hyperparams = checkpoint.get('hyperparameters', {})
        trial_info.update(hyperparams)
        
        trial_info_list.append(trial_info)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Create a DataFrame from the collected trial information.
df_trials = pd.DataFrame(trial_info_list)

# Sort the DataFrame in ascending order of validation loss (lowest first).
df_trials.sort_values(by="val_loss", inplace=True)

# Save the DataFrame as a CSV file.
output_csv = os.path.join(CHECKPOINTS_DIR, "trial_results.csv")
df_trials.to_csv(output_csv, index=False)

print(f"Trial results saved to {output_csv}")

