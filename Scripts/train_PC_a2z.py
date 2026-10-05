import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import shutil
import time
import plotly
import sklearn
from copy import deepcopy

# Import shared utilities from our utils.py module.
from utils_PC_a2z import (
    set_random_seeds,
    get_device,
    get_indices,
    DNADualBatchDataset,
    BatchPreprocessor,
    make_loader,
    TwoBranchCNN,
    TwoBranchCNN_OHE,
    EMPRES_CONFIG,
)

# ============================================================================
# 1. Set Random Seeds and Global Settings
# ============================================================================
set_random_seeds(42)
device = get_device()
print("device:", device)

# ============================================================================
# 2. Parse Command-Line Arguments with argparse
# ============================================================================
parser = argparse.ArgumentParser(
    description="Train EMPRES models with Hyperparameter Optimization"
)
parser.add_argument(
    "--data_dir", type=str, required=True,
    help="Absolute path to Input data directory"
)
parser.add_argument(
    "--out_dir", type=str, required=True,
    help="Base output directory under which run specific subdirs will be made"
)
parser.add_argument(
    "--val_group", type=str, default="4",
    help="Validation group number (default: 4)"
)
parser.add_argument(
    "--test_group", type=str, default="5",
    help="Test group number (default: 5)"
)
parser.add_argument(
    "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
    help="EMPRES model type to train (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)"
)
parser.add_argument(
    "--num_workers", type=int, default=0,
    help="DataLoader worker processes (default: 0 = load batches in the main process). "
         "Keep it <= cpus-per-task - 1."
)
parser.add_argument(
    "--prefetch_factor", type=int, default=2,
    help="Batches prepared ahead by each worker; only used when --num_workers > 0 (default: 2)"
)
parser.add_argument(
    "--in_ram", action="store_true",
    help="Read the input arrays fully into RAM instead of memory-mapping them"
)

args = parser.parse_args()
data_dir = args.data_dir
out_dir = args.out_dir
val_group  = args.val_group
test_group = args.test_group
EMPRES_type = args.EMPRES_type
num_workers = args.num_workers
prefetch_factor = args.prefetch_factor
in_ram = args.in_ram
cfg = EMPRES_CONFIG[EMPRES_type]
print(f"Reading input data from: {data_dir}")
print(f"\nUsing validation group: {val_group} and test group: {test_group}, EMPRES_type: {EMPRES_type}")
print(f"Checkpoints will be saved under subdir: {cfg['subdir']}")
print(f"Data loading: {'in RAM' if in_ram else 'memory-mapped'}, num_workers={num_workers}"
      + (f", prefetch_factor={prefetch_factor}" if num_workers > 0 else "")
      + f", CPUs available to this job: {len(os.sched_getaffinity(0))}")
print("Early stopping: patience=10 epochs, min_improvement=0.01 (a new best is recorded only if val_loss drops by at least 0.01)")

# Canonical CV fold numbering used throughout the project:
#   fold 1: val1_test2, fold 2: val2_test3, ..., fold 5: val5_test1
FOLD_NUMBER = {(1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4, (5, 1): 5}
fold_num = FOLD_NUMBER.get((int(val_group), int(test_group)))
fold_label = f"Fold {fold_num}" if fold_num is not None else f"val{val_group}_test{test_group}"

# ============================================================================
# 3. Global directory for input data
# ============================================================================
# Directory where input data files and standardization statistics files are stored.
DATA_DIR = data_dir

# ============================================================================
# 4. Data Loading (memory-mapped by default, fully in RAM with --in_ram)
# ============================================================================
# With --in_ram the base and extra arrays are kept as SEPARATE in-RAM arrays;
# they are never concatenated on the CPU (that would need twice the memory).
# The concatenation happens per batch on the GPU in BatchPreprocessor.
input_mmap_mode = None if in_ram else 'r'
tss = np.load(os.path.join(DATA_DIR, cfg["base_tss_file"]), mmap_mode = input_mmap_mode, allow_pickle = True)
tts = np.load(os.path.join(DATA_DIR, cfg["base_tts_file"]), mmap_mode = input_mmap_mode, allow_pickle = True)
TPM = np.load(os.path.join(DATA_DIR, "TPM.npy"), mmap_mode = 'r', allow_pickle = True)
groups = np.load(os.path.join(DATA_DIR,"group_for_cross_validation.npy"), mmap_mode = 'r', allow_pickle = True)

print("Loaded shapes:")
print("tss:",    tss.shape)     # Expected: (N, C, L) ; e.g. (N, 384, 20) for EMPRES 1-3, (N, 925, 20) for 4, (N, 4, 5000) for 0
print("tts:",    tts.shape)     # Expected: same shape as tss
print("TPM:",    TPM.shape)     # Expected: (N, )
print("groups:", groups.shape)  # Expected: (N, )

# Transform TPM values to log(1+TPM) in base 10.
TPM = np.log10(1 + TPM)

# Optionally load extra channels (a2z_preds or a2z_embeddings)
if cfg["extra_tss_file"] is not None:
    extra_tss = np.load(os.path.join(DATA_DIR, cfg["extra_tss_file"]), mmap_mode = input_mmap_mode, allow_pickle = True)  # Expected: (N, 1, 20) for pred, (N, 925, 20) for emb
    extra_tts = np.load(os.path.join(DATA_DIR, cfg["extra_tts_file"]), mmap_mode = input_mmap_mode, allow_pickle = True)  # Expected: (N, 1, 20) for pred, (N, 925, 20) for emb
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
storage_url = f"sqlite:////{run_dir}/optuna_study_history.db"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# 6. Load Global Statistics for Standardization
# ============================================================================
if cfg["standardize"]:
    train_groups_sorted = np.sort(train_groups)
    train_groups_str = "_".join(map(str, train_groups_sorted))
    global_stats_file = os.path.join(DATA_DIR,f"global_stats_train_{train_groups_str}.npz")
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

# Determine in_channels for the model
base_channels = tss_mean.shape[1]   # Expected: 384 (EMPRES 1-3), 925 (EMPRES 4), 4 (EMPRES 0)
extra_channels = extra_tss.shape[1] if extra_tss is not None else 0
in_channels = base_channels + extra_channels     # Expected: base_channels + extra_channels

# # BEGIN SANITY CHECK (per‐channel, per‐position mean/std of standardized tss/tts)
# # (Remove this block when done with the check)
# tss_train = (tss[train_idx] - tss_mean) / tss_std
# tts_train = (tts[train_idx] - tts_mean) / tts_std
# tss_mean_chk = np.mean(tss_train, axis=0)
# tss_std_chk  = np.std( tss_train, axis=0)
# tts_mean_chk = np.mean(tts_train, axis=0)
# tts_std_chk  = np.std( tts_train, axis=0)

# if extra != "none":
#     # loading mean and std of a2z predictions from stats file
#     extra_tss_train = (extra_tss[train_idx] - extra_tss_mean) / extra_tss_std
#     extra_tts_train = (extra_tts[train_idx] - extra_tts_mean) / extra_tts_std
#     extra_tss_mean_chk = np.mean(extra_tss_train, axis=0)
#     extra_tss_std_chk  = np.std( extra_tss_train, axis=0)
#     extra_tts_mean_chk = np.mean(extra_tts_train, axis=0)
#     extra_tts_std_chk  = np.std( extra_tts_train, axis=0)

# print("Sanity check — standardized TSS mean shape:", tss_mean_chk.shape)
# print(tss_mean_chk)
# print("Sanity check — standardized TSS std shape:", tss_std_chk.shape)
# print(tss_std_chk)
# print("Sanity check — standardized TTS mean shape:", tts_mean_chk.shape)
# print(tts_mean_chk)
# print("Sanity check — standardized TTS std shape:", tts_std_chk.shape)
# print(tts_std_chk)

# if extra != "none":
#     print(f"\nThe extra channel(s) data used is a2z {extra}\n")
#     print(f"Sanity check — a2z {extra} standardized TSS mean shape:", extra_tss_mean_chk.shape)
#     print(extra_tss_mean_chk)
#     print(f"Sanity check — a2z {extra} standardized TSS std shape:", extra_tss_std_chk.shape)
#     print(extra_tss_std_chk)
#     print(f"Sanity check — a2z {extra} standardized TTS mean shape:", extra_tts_mean_chk.shape)
#     print(extra_tts_mean_chk)
#     print(f"Sanity check — a2z {extra} standardized TTS std shape:", extra_tts_std_chk.shape)
#     print(extra_tts_std_chk)
# # END SANITY CHECK

# ============================================================================
# 7. Create Dataset Instances and the GPU-side Batch Preprocessor
# ============================================================================
# The datasets return RAW batches (no standardization, no concatenation).
# Creating Training Dataset
train_dataset = DNADualBatchDataset(
    train_idx,
    tss, tts, TPM,
    extra_tss = extra_tss,
    extra_tts = extra_tts,
)

# Creating Validation Dataset
val_dataset = DNADualBatchDataset(
    val_idx,
    tss, tts, TPM,
    extra_tss = extra_tss,
    extra_tts = extra_tts,
)

# Standardization (base and extra channels each with their own mean/std) and
# channel concatenation, executed on `device` for every batch.
preprocess = BatchPreprocessor(
    device,
    tss_mean, tss_std,
    tts_mean, tts_std,
    extra_tss_mean = extra_tss_mean,
    extra_tss_std  = extra_tss_std,
    extra_tts_mean = extra_tts_mean,
    extra_tts_std  = extra_tts_std,
    standardize    = cfg["standardize"],
)

# ============================================================================
# 8. Define the Objective Function for Optuna with Early Stopping
# ============================================================================
def objective(trial):
    # Per-trial resource report: start the clock and reset the GPU peak-memory counter
    trial_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # DataLoaders for this trial
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    train_loader = make_loader(train_dataset, batch_size, shuffle = True,
                               num_workers = num_workers, prefetch_factor = prefetch_factor)

    val_loader = make_loader(val_dataset, batch_size, shuffle = False,
                             num_workers = num_workers, prefetch_factor = prefetch_factor)

    model = cfg["model_class"](trial, in_channels=in_channels, **cfg["model_kwargs"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-2, log = True))
    criterion = nn.MSELoss()
    max_epochs = 50
    lookahead_epochs = 10
    min_improvement = 0.01

    best_val_loss = float('inf')
    best_epoch = 0
    best_checkpoint = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for tss_parts, tts_parts, target in train_loader:
            x_tss, x_tts, y = preprocess(tss_parts, tts_parts, target)
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            out = model(x_tss, x_tts)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_tss.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tss_parts, tts_parts, target in val_loader:
                x_tss, x_tts, y = preprocess(tss_parts, tts_parts, target)
                y = y.unsqueeze(1)
                out = model(x_tss, x_tts)
                loss = criterion(out, y)
                val_loss += loss.item() * x_tss.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        rmse = val_loss ** 0.5

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, RMSE={rmse:.4f}")

        # A new best is recorded only when validation loss drops by at least
        # min_improvement (0.01). That update also resets the patience clock.
        # Tiny drops (< 0.01) do not overwrite the checkpoint and do not
        # postpone early stopping.
        if val_loss < best_val_loss - min_improvement:
            improvement = best_val_loss - val_loss
            best_val_loss   = val_loss
            best_epoch      = epoch + 1
            best_checkpoint = {
                'trial_number': trial.number,
                'epoch': best_epoch,
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                'hyperparameters': trial.params,
                'val_loss': val_loss,
                'RMSE': rmse,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
            }
            print(
                f"  New best at epoch {best_epoch}: val_loss={val_loss:.4f} "
                f"(improvement={improvement:.4f} >= {min_improvement})"
            )
        elif (epoch + 1 - best_epoch) >= lookahead_epochs:
            improvement = best_val_loss - val_loss
            print(
                f"Early stopping triggered at epoch {epoch+1}: no validation-loss "
                f"improvement of at least {min_improvement} over best "
                f"({best_val_loss:.4f}) for {lookahead_epochs} epochs "
                f"(current val_loss={val_loss:.4f}, change vs best={improvement:.4f})."
            )
            trial.set_user_attr("early_stopped", True)
            break

        trial.report(val_loss, epoch)

    if best_checkpoint is not None:
        filename = f"checkpoint_trial_{trial.number}.pth"
        checkpoint_filename = os.path.join(CHECKPOINTS_DIR, filename)
        torch.save(best_checkpoint, checkpoint_filename)
        print(f"Saved best checkpoint for trial {trial.number} at epoch {best_epoch} to {checkpoint_filename}")

        # Save TorchScript version
        ts_model = cfg["model_class"](trial, in_channels=in_channels, **cfg["model_kwargs"]).to(device)
        ts_model.load_state_dict(best_checkpoint['model_state_dict'])
        ts_model.eval()
        pt_filename = f"checkpoint_trial_{trial.number}.pt"
        pt_path = os.path.join(CHECKPOINTS_DIR, pt_filename)
        torch.jit.save(torch.jit.script(ts_model), pt_path)
        print(f"Saved TorchScript model for trial {trial.number} as {pt_path}")

    # Per-trial resource report (wall time includes data loading, training, validation, checkpointing)
    n_epochs_run = len(train_loss_history)
    trial_seconds = time.perf_counter() - trial_start_time
    print(f"Trial {trial.number}: batch_size={batch_size}, {n_epochs_run} epochs in {trial_seconds:.1f} s "
          f"({trial_seconds / max(n_epochs_run, 1):.1f} s/epoch)")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    return best_val_loss

# ============================================================================
# 9. Run Hyperparameter Optimization with Optuna and Save Plots
# ============================================================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name = f"val{val_group}_test{test_group}_{cfg['study_tag']}",
        storage=storage_url,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True
    )

    # Run Optuna up to a target *total* number of finalized trials (resume-safe):
    # on a fresh study n_existing=0 so this runs the full 200; on a resumed study
    # it runs only the remainder, so the study always ends at TARGET_N_TRIALS total.
    TARGET_N_TRIALS = 200
    n_existing = len(study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
    ))
    n_remaining = max(0, TARGET_N_TRIALS - n_existing)
    print(f"Study has {n_existing} completed/pruned trials; targeting {TARGET_N_TRIALS} total, will run {n_remaining} more.")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining)
    else:
        print(f"Study already has {n_existing} >= {TARGET_N_TRIALS} trials; skipping optimize().")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Best trial number: {best_trial.number}")
    print(f"  Validation Loss: {best_trial.value:.4f}")

    candidate_filename = os.path.join(CHECKPOINTS_DIR, f"checkpoint_trial_{best_trial.number}.pth")
    best_model_filename = candidate_filename

    best_checkpoint = torch.load(best_model_filename, weights_only=False, map_location=device)
    best_epoch = best_checkpoint.get("epoch", "N/A")
    print(f"  Best epoch: {best_epoch}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    complete_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    pruned_trials  = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
    early_stopped_trials = sum(1 for t in study.trials if t.user_attrs.get("early_stopped", False))
    print(f"Total completed trials: {complete_trials}")
    print(f"Total pruned trials:    {pruned_trials}")
    print(f"Total early stopped:    {early_stopped_trials}")

    history_fig  = vis.plot_optimization_history(study)
    parallel_fig = vis.plot_parallel_coordinate(study)
    param_fig    = vis.plot_param_importances(study)
    slice_fig    = vis.plot_slice(study)

    for fig in (history_fig, parallel_fig, param_fig, slice_fig):
        fig.update_layout(width=1200, height=800)

    history_fig.write_html(os.path.join(PLOTS_DIR, "optimization_history.html"))
    parallel_fig.write_html(os.path.join(PLOTS_DIR, "parallel_coordinate.html"))
    param_fig.write_html(os.path.join(PLOTS_DIR, "parameter_importances.html"))
    slice_fig.write_html(os.path.join(PLOTS_DIR, "slice_plot.html"))

    history_fig.write_image(os.path.join(PLOTS_DIR, "optimization_history.png"), scale=3)
    parallel_fig.write_image(os.path.join(PLOTS_DIR, "parallel_coordinate.png"), scale=3)
    param_fig.write_image(os.path.join(PLOTS_DIR, "parameter_importances.png"), scale=3)
    slice_fig.write_image(os.path.join(PLOTS_DIR, "slice_plot.png"), scale=3)

    overall_best_filename = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    shutil.copy(best_model_filename, overall_best_filename)
    print(f"Best overall model saved as {overall_best_filename}")

    checkpoint = torch.load(overall_best_filename, weights_only=False, map_location=device)
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history   = checkpoint.get('val_loss_history', [])
    if train_loss_history and val_loss_history:
        epochs_range = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, train_loss_history, marker='o', label='Training Loss')
        plt.plot(epochs_range, val_loss_history,   marker='o', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Learning Curves for Best Trial - EMPRES_{EMPRES_type} - {fold_label}"
        )
        plt.legend()
        ticks = [1] + list(range(5, max(epochs_range)+1, 5))
        plt.xticks(ticks)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "learning_curve.svg"), format="svg", dpi=600)
        plt.savefig(os.path.join(PLOTS_DIR, "learning_curve.png"), format="png", dpi=600)
        plt.close()
        print(f"Learning curve plot saved in {PLOTS_DIR}")
    else:
        print("No loss history found in the best checkpoint; cannot plot learning curve.")
