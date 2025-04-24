#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import (
    set_random_seeds,
    get_device,
    get_indices,
    DNADualDataset,
    TwoBranchCNN,
    DummyTrial,
    evaluate_model,
)

# ============================================================================
# 1. Set Random Seeds and Global Settings
# ============================================================================
set_random_seeds(42)
device = get_device()
print("device:", device)

# ============================================================================
# 2. Parse Command-Line Arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="Test top-X 2BCNN models on the test set and report losses"
)
parser.add_argument(
    "--val_group", type=str, required=True,
    help="Validation group ID (e.g. '4')"
)
parser.add_argument(
    "--test_group", type=str, required=True,
    help="Test group ID (e.g. '5')"
)
parser.add_argument(
    "--top_x", type=int, default=1,
    help="Number of top models to evaluate"
)
args = parser.parse_args()

# ============================================================================
# 3. Define Paths and Directories
# ============================================================================
DATA_DIR = "/home/behrooz/WP2/Datasets/PC_Embeddings/npy_files/Bdi_Osa"
base_output_dir = (f"/home/behrooz/WP2/Models/2BCNN/Allspecies_PCembed/val{args.val_group}_test{args.test_group}/round8_Bdi_Osa/"
)
CHECKPOINTS_DIR = base_output_dir
TRIALS_CSV = os.path.join(CHECKPOINTS_DIR, "trial_results.csv")

# ============================================================================
# 4. Load Data and Compute Indices
# ============================================================================
tss = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tss.npy"), mmap_mode="r", allow_pickle=True)
tts = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tts.npy"), mmap_mode="r", allow_pickle=True)
TPM = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_TPM.npy"), mmap_mode="r", allow_pickle=True)
groups = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_group_for_cross_validation.npy"), mmap_mode="r", allow_pickle=True)

# Log-transform targets
TPM = np.log10(1 + TPM)

# Get train/val/test indices (we only need test_idx here)
train_idx, _, test_idx = get_indices(args.val_group, args.test_group, groups)
print("Test set size:", len(test_idx))

# ============================================================================
# 5. Load Global Statistics for Standardization
# ============================================================================
train_groups = np.unique(groups[train_idx])
train_groups_str = "_".join(map(str, np.sort(train_groups)))
stats_path = os.path.join(
    DATA_DIR, f"global_stats_train_{train_groups_str}.npz"
)
stats = np.load(stats_path)
tss_mean, tss_std = stats["tss_mean"], stats["tss_std"]
tts_mean, tts_std = stats["tts_mean"], stats["tts_std"]
stats.close()
print("Loaded global stats from", stats_path)

# ============================================================================
# 6. Create Test Dataset & DataLoader
# ============================================================================
test_dataset = DNADualDataset(test_idx, tss, tts, TPM,tss_mean, tss_std, tts_mean, tts_std)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ============================================================================
# 7. Select Top-X Trials
# ============================================================================
df_trials = pd.read_csv(TRIALS_CSV)
top_trials = df_trials.head(args.top_x).reset_index(drop=True)
print("Top models:")
print(top_trials[["checkpoint_file", "val_loss", "batch_size"]])

# ============================================================================
# 8. Evaluate Each Model on the Test Set
# ============================================================================
all_predictions = []
test_loss_list = []

for idx, row in top_trials.iterrows():
    ckpt_file = row["checkpoint_file"].strip()
    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}")
        continue

    print(f"\nLoading model from: {ckpt_path}")
    # Build hyperparameter dict from CSV row
    hp = {
        "n_conv_layers":       int(row["n_conv_layers"]),
        "n_filters":           int(row["n_filters"]),
        "kernel_size":         int(row["kernel_size"]),
        "n_dense_layers":      int(row["n_dense_layers"]),
        "dense_units":         int(row["dense_units"]),
        "n_post_dense_layers": int(row["n_post_dense_layers"]),
        "dropout_rate":        float(row["dropout_rate"]),
        "batch_norm":          True
    }
    dummy = DummyTrial(hp)

    # Instantiate and load the model
    model = TwoBranchCNN(dummy).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate on test set
    test_loss, predictions = evaluate_model(model, test_loader, device, torch.nn.MSELoss())
    print(f"Model {idx+1} test loss: {test_loss:.6f}")

    test_loss_list.append(test_loss)
    all_predictions.append(predictions.squeeze())

# ============================================================================
# 9. Ensemble Predictions
# ============================================================================
if not all_predictions:
    print("No predictions were generated. Exiting.")
    exit(1)

pred_stack = np.stack(all_predictions, axis=0)    # shape: (top_x, N_test)
ensemble_pred = pred_stack.mean(axis=0)           # shape: (N_test,)

# compute ensemble loss using pytorch MSELoss
observed = torch.tensor(TPM[test_idx], dtype=torch.float32, device=device)
ensemble = torch.tensor(ensemble_pred, dtype=torch.float32, device=device)
ensemble_loss = torch.nn.functional.mse_loss(ensemble, observed).item()
print(f"\nEnsemble test loss: {ensemble_loss:.6f}")

if __name__ == "__main__":
    # all logic runs at import time above, so nothing here
    pass
