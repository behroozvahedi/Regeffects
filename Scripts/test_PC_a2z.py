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
    TwoBranchCNN_OHE,
    DummyTrial,
    evaluate_model,
    EMPRES_CONFIG,
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
    description="Test top-X EMPRES models on the test set and report losses"
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
    "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
    help="EMPRES model type to test (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)"
)
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir
val_group = args.val_group
test_group = args.test_group
top_x = args.top_x
EMPRES_type = args.EMPRES_type
cfg = EMPRES_CONFIG[EMPRES_type]
print(f"Reading input data from: {data_dir}")
print(f"\nUsing validation group: {val_group}, test group: {test_group}, EMPRES_type: {EMPRES_type}")

# Canonical CV fold numbering (same as train_PC_a2z.py): fold 1: val1_test2, ..., fold 5: val5_test1
FOLD_NUMBER = {(1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4, (5, 1): 5}
fold_num = FOLD_NUMBER.get((int(val_group), int(test_group)))

# ============================================================================
# 3. Define Paths and Directories
# ============================================================================
# Directory where input data files and standardization statistics files are stored.
DATA_DIR = data_dir

# ============================================================================
# 4. Load Data and Compute Indices
# ============================================================================
tss = np.load(os.path.join(DATA_DIR, cfg["base_tss_file"]), mmap_mode = "r", allow_pickle = True)
tts = np.load(os.path.join(DATA_DIR, cfg["base_tts_file"]), mmap_mode = "r", allow_pickle = True)
TPM = np.load(os.path.join(DATA_DIR, "TPM.npy"), mmap_mode = "r", allow_pickle = True)
groups = np.load(os.path.join(DATA_DIR, "group_for_cross_validation.npy"), mmap_mode = "r", allow_pickle = True)

print("Loaded shapes:")
print("tss:", tss.shape)
print("tts:", tts.shape)
print("TPM:", TPM.shape)
print("groups:", groups.shape)

# Log-transform targets
TPM = np.log10(1 + TPM)

# Optionally load extra channels (a2z_preds or a2z_embeddings)
if cfg["extra_tss_file"] is not None:
    extra_tss = np.load(os.path.join(DATA_DIR, cfg["extra_tss_file"]), mmap_mode = 'r', allow_pickle = True)  # Expected: (N, 1, 20) for pred, (N, 925, 20) for emb
    extra_tts = np.load(os.path.join(DATA_DIR, cfg["extra_tts_file"]), mmap_mode = 'r', allow_pickle = True)  # Expected: (N, 1, 20) for pred, (N, 925, 20) for emb
    print("Loaded extra tss channels:", extra_tss.shape)
    print("Loaded extra tts channels:", extra_tts.shape)

else:
    extra_tss = extra_tts = None

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

# Building run‐specific directories under out_dir by joining out_dir + validation and test group numbers + EMPRES-type-specific subdir
run_dir = os.path.join(
    out_dir,
    f"val{val_group}_test{test_group}",
    cfg["subdir"],
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
if cfg["standardize"]:
    train_groups_sorted = np.sort(train_groups)
    train_groups_str    = "_".join(map(str, train_groups_sorted))
    global_stats_file   = os.path.join(DATA_DIR,f"global_stats_train_{train_groups_str}.npz")
    stats = np.load(global_stats_file)
    base_keys = cfg["base_stats_keys"]
    tss_mean = stats[base_keys[0]]
    tss_std  = stats[base_keys[1]]
    tts_mean = stats[base_keys[2]]
    tts_std  = stats[base_keys[3]]

    if cfg["extra_stats_keys"] is not None:
        # loading mean and std of extra channels from stats file
        extra_keys = cfg["extra_stats_keys"]
        extra_tss_mean = stats[extra_keys[0]]
        extra_tss_std  = stats[extra_keys[1]]
        extra_tts_mean = stats[extra_keys[2]]
        extra_tts_std  = stats[extra_keys[3]]
    else:
        extra_tss_mean = extra_tss_std = extra_tts_mean = extra_tts_std = None
    stats.close()
    print("Loaded global stats from", global_stats_file)
else:
    # For EMPRES 0 (OHE input): skip .npz load and use identity standardization
    # (mean=0, std=1) so DNADualDataset passes OHE values through unchanged.
    base_C = tss.shape[1]
    base_L = tss.shape[2]
    tss_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
    tss_std  = np.ones((1, base_C, base_L), dtype=np.float32)
    tts_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
    tts_std  = np.ones((1, base_C, base_L), dtype=np.float32)
    extra_tss_mean = extra_tss_std = extra_tts_mean = extra_tts_std = None
    print("Skipped global stats load (identity standardization for OHE input).")

# Determine model input channels
base_channels = tss_mean.shape[1]  # Expected: 384 (EMPRES 1-3), 925 (EMPRES 4), 4 (EMPRES 0)
extra_channels = extra_tss.shape[1] if extra_tss is not None else 0
in_channels = base_channels + extra_channels  # Expected: base_channels + extra_channels

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

# One row per top-X model plus one ensemble row; written to a TSV file in the run directory
results_rows = []
# e.g. test_results_EMPRES_1_fold_1.tsv (fold 1 = val1_test2); a non-standard val/test pair is named by its groups.
# top_x is not in the name (it is a column in the file): a rerun with another --top_x overwrites the file.
fold_tag = f"fold_{fold_num}" if fold_num is not None else f"val{val_group}_test{test_group}"
TEST_RESULTS_TSV = os.path.join(CHECKPOINTS_DIR, f"test_results_EMPRES_{EMPRES_type}_{fold_tag}.tsv")

def result_row(rank, checkpoint_file, trial_number, status, recorded_val_loss=None,
               replicated_val_loss=None, test_loss=None, n_models_in_ensemble=None):
    return {
        "EMPRES_type": EMPRES_type, "fold": fold_num, "val_group": val_group, "test_group": test_group,
        "top_x": top_x, "rank": rank, "checkpoint_file": checkpoint_file, "trial_number": trial_number,
        "status": status, "recorded_val_loss": recorded_val_loss, "replicated_val_loss": replicated_val_loss,
        "test_loss": test_loss, "n_val": len(val_idx), "n_test": len(test_idx),
        "n_models_in_ensemble": n_models_in_ensemble,
    }

def save_test_results(rows):
    # Write to a temporary name first, then rename, so a crash never leaves a half-written file
    tmp_path = TEST_RESULTS_TSV + ".tmp"
    df_out = pd.DataFrame(rows)
    for col in ("fold", "trial_number", "n_models_in_ensemble"):   # integers, empty where not applicable
        df_out[col] = df_out[col].astype("Int64")
    df_out.to_csv(tmp_path, sep="\t", index=False)
    os.replace(tmp_path, TEST_RESULTS_TSV)
    print(f"Test results saved to {TEST_RESULTS_TSV}")

for idx, row in top_trials.iterrows():
    checkpoint_file = row["checkpoint_file"].strip()
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        results_rows.append(result_row(idx + 1, checkpoint_file, int(row["trial_number"]), "checkpoint not found",
                                       recorded_val_loss=float(row["val_loss"])))
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
    # EMPRES 0 has two extra Optuna-searched categorical hyperparameters
    if EMPRES_type == 0:
        hp["dilation"]  = int(row["dilation"])
        hp["pool_size"] = int(row["pool_size"])
    dummy = DummyTrial(hp)

    # Instantiate and load the model with correct in_channels
    model = cfg["model_class"](dummy, in_channels=in_channels, **cfg["model_kwargs"]).to(device)
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
    results_rows.append(result_row(idx + 1, checkpoint_file, int(row["trial_number"]), "evaluated",
                                   recorded_val_loss=float(row["val_loss"]),
                                   replicated_val_loss=val_loss, test_loss=test_loss))

# ============================================================================
# 10. Ensemble Predictions
# ============================================================================
if not all_predictions:
    print("No predictions were generated. Exiting.")
    save_test_results(results_rows)
    exit(1)

pred_stack = np.stack(all_predictions, axis=0)
ensemble_pred = pred_stack.mean(axis=0)

observed = torch.tensor(TPM[test_idx], dtype=torch.float32, device=device)
ensemble = torch.tensor(ensemble_pred, dtype=torch.float32, device=device)
ensemble_loss = torch.nn.functional.mse_loss(ensemble, observed).item()
print(f"\nEnsemble test loss: {ensemble_loss:.6f}")

evaluated_files = [r["checkpoint_file"] for r in results_rows if r["status"] == "evaluated"]
results_rows.append(result_row("ensemble", ",".join(evaluated_files), None, "evaluated",
                               test_loss=ensemble_loss, n_models_in_ensemble=len(all_predictions)))
save_test_results(results_rows)

if __name__ == "__main__":
    pass
