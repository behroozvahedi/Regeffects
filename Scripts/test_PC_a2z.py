import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils_PC_a2z import (
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
    "--data_dir", type=str, required=True,
    help="Directory where input .npy/.npz files are located"
)
parser.add_argument(
    "--out_dir", type=str, required=True,
    help="Base output directory (must match train.py --out_dir)"
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
parser.add_argument(
    "--use_extra", action="store_true",
    help="Include extra channels from a2z_tss_preds.npy & a2z_tts_preds.npy"
)
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir
val_group = args.val_group
test_group = args.test_group
top_x = args.top_x
use_extra = args.use_extra
print(f"Reading input data from: {data_dir}")
print(f"\nUsing validation group: {val_group}, test group: {test_group}, use_extra: {use_extra}")

# ============================================================================
# 3. Define Paths and Directories
# ============================================================================
# Directory where input data files and standardization statistics files are stored.
DATA_DIR = data_dir

# ============================================================================
# 4. Load Data and Compute Indices
# ============================================================================
tss = np.load(os.path.join(DATA_DIR, "tss.npy"), mmap_mode = "r", allow_pickle = True)
tts = np.load(os.path.join(DATA_DIR, "tts.npy"), mmap_mode = "r", allow_pickle = True)
TPM = np.load(os.path.join(DATA_DIR, "TPM.npy"), mmap_mode = "r", allow_pickle = True)
groups = np.load(os.path.join(DATA_DIR, "group_for_cross_validation.npy"), mmap_mode = "r", allow_pickle = True)

print("Loaded shapes:")
print("tss:", tss.shape)
print("tts:", tts.shape)
print("TPM:", TPM.shape)
print("groups:", groups.shape)

# Log-transform targets
TPM = np.log10(1 + TPM)

if use_extra:
    extra_tss = np.load(os.path.join(data_dir, "a2z_tss_preds.npy"), mmap_mode = "r", allow_pickle = True)
    extra_tts = np.load(os.path.join(data_dir, "a2z_tts_preds.npy"), mmap_mode = "r", allow_pickle = True)

    # extra_tss = np.load(os.path.join(data_dir, "a2z_tss_embeds.npy"), mmap_mode = "r", allow_pickle = True)
    # extra_tts = np.load(os.path.join(data_dir, "a2z_tts_embeds.npy"), mmap_mode = "r", allow_pickle = True)

    print("Loaded extra TSS channels:", extra_tss.shape)
    print("Loaded extra TTS channels:", extra_tts.shape)
else:
    extra_tss = None
    extra_tts = None

# ============================================================================
# 5. Cross-Validation Splitting and Output Directories
# ============================================================================
train_idx, val_idx, test_idx = get_indices(val_group, test_group, groups)
print("Fold split:")
print("  Validation group:", val_group)
print("  Test group:      ", test_group)
train_groups = np.unique(groups[train_idx])
print("  Training groups: ", train_groups)

# Print datasets size
print("\nTrain set size:", len(train_idx))
print("Validation set size:", len(val_idx))
print("Test set size:", len(test_idx))

# Building run‐specific directories under out_dir by joining out_dir + validation and test group numbers + base/full models
run_dir = os.path.join(
    out_dir,
    f"val{val_group}_test{test_group}",
    "full_models" if use_extra else "base_models",
)

os.makedirs(run_dir, exist_ok=True)

CHECKPOINTS_DIR = run_dir
PLOTS_DIR = os.path.join(run_dir, "plots")
TRIALS_CSV = os.path.join(CHECKPOINTS_DIR, "trial_results.csv")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# 6. Load Global Statistics for Standardization
# ============================================================================
train_groups_sorted = np.sort(train_groups)
train_groups_str    = "_".join(map(str, train_groups_sorted))
global_stats_file   = os.path.join(DATA_DIR,f"global_stats_train_{train_groups_str}.npz")
stats = np.load(global_stats_file)
tss_mean = stats['tss_mean']
tss_std  = stats['tss_std']
tts_mean = stats['tts_mean']
tts_std  = stats['tts_std']
if use_extra:
    extra_tss_mean = stats['extra_tss_mean']
    extra_tss_std  = stats['extra_tss_std']
    extra_tts_mean = stats['extra_tts_mean']
    extra_tts_std  = stats['extra_tts_std']
else:
    extra_tss_mean = extra_tss_std = None
    extra_tts_mean = extra_tts_std = None
stats.close()
print("Loaded global stats from", global_stats_file)

# Determine model input channels
base_channels = tss_mean.shape[1]  # Expected: 384
extra_channels = extra_tss.shape[1] if use_extra else 0
in_channels = base_channels + extra_channels  # Expected: 384 + extra_channels

# ============================================================================
# 7. Create Test Dataset & DataLoader
# ============================================================================
test_dataset = DNADualDataset(
    test_idx,
    tss, tts, TPM,
    tss_mean, tss_std,
    tts_mean, tts_std,
    extra_tss = extra_tss,
    extra_tss_mean = extra_tss_mean,
    extra_tss_std = extra_tss_std,
    extra_tts = extra_tts,
    extra_tts_mean = extra_tts_mean,
    extra_tts_std = extra_tts_std,
)

test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False)

val_dataset = DNADualDataset(
    val_idx,
    tss, tts, TPM,
    tss_mean, tss_std,
    tts_mean, tts_std,
    extra_tss = extra_tss,
    extra_tss_mean = extra_tss_mean,
    extra_tss_std = extra_tss_std,
    extra_tts = extra_tts,
    extra_tts_mean = extra_tts_mean,
    extra_tts_std = extra_tts_std,
)

val_loader = DataLoader(val_dataset, batch_size = 256, shuffle = False)
# ============================================================================
# 8. Select Top-X Trials
# ============================================================================
df_trials = pd.read_csv(TRIALS_CSV)
top_trials = df_trials.head(top_x).reset_index(drop=True)
print("Top models:")
print(top_trials[["checkpoint_file", "val_loss", "batch_size"]])

# ============================================================================
# 9. Evaluate Each Model on the Test Set
# ============================================================================
all_predictions = []
test_loss_list = []
criterion = torch.nn.MSELoss()

for idx, row in top_trials.iterrows():
    checkpoint_file = row["checkpoint_file"].strip()
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        continue

    print(f"\nLoading model from: {checkpoint_path}")
    # Build hyperparameter dict from CSV row
    hp = {
        "n_conv_layers":       int(row["n_conv_layers"]),
        "n_filters":           int(row["n_filters"]),
        "kernel_size":         int(row["kernel_size"]),
        "n_dense_layers":      int(row["n_dense_layers"]),
        "dense_units":         int(row["dense_units"]),
        "n_post_dense_layers": int(row["n_post_dense_layers"]),
        "dropout_rate":        float(row["dropout_rate"]),
        "batch_norm":          True,
    }
    dummy = DummyTrial(hp)

    # Instantiate and load the model with correct in_channels
    model = TwoBranchCNN(dummy, in_channels=in_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on validation set to replicate val loss
    val_loss, predictions = evaluate_model(
        model, val_loader, device, criterion
    )
    print(f"Model {idx+1} replicated val loss: {val_loss:.6f}")

    # Evaluate on test set
    test_loss, predictions = evaluate_model(
        model, test_loader, device, criterion
    )
    print(f"Model {idx+1} test loss: {test_loss:.6f}")

    test_loss_list.append(test_loss)
    all_predictions.append(predictions.squeeze())

# ============================================================================
# 10. Ensemble Predictions
# ============================================================================
if not all_predictions:
    print("No predictions were generated. Exiting.")
    exit(1)

pred_stack = np.stack(all_predictions, axis=0)
ensemble_pred = pred_stack.mean(axis=0)

observed = torch.tensor(TPM[test_idx], dtype=torch.float32, device=device)
ensemble = torch.tensor(ensemble_pred, dtype=torch.float32, device=device)
ensemble_loss = torch.nn.functional.mse_loss(ensemble, observed).item()
print(f"\nEnsemble test loss: {ensemble_loss:.6f}")

if __name__ == "__main__":
    pass
