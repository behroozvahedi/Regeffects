import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import shutil
import plotly
import sklearn
from copy import deepcopy

# Import shared utilities from our utils.py module.
from utils_PC_a2z import (
    set_random_seeds,
    get_device,
    get_indices,
    DNADualDataset,
    TwoBranchCNN,
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
    description="Train Dual-Branch CNN with Hyperparameter Optimization"
)
parser.add_argument(
    "--data_dir", type=str,
    help="Absolute path to Input data directory"
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
    "--use_extra", action="store_true",
    help="Include extra channels from a2z_preds.npy"
)
args = parser.parse_args()
data_dir = args.data_dir
val_group  = args.val_group
test_group = args.test_group
use_extra  = args.use_extra
print(f"Using validation group: {val_group} and test group: {test_group}, use_extra: {use_extra}")

# ============================================================================
# 3. Global directory for input data
# ============================================================================
# DATA_DIR = "/home/behrooz/WP2/Datasets/PC_Embeddings/npy_files/Bdi_Osa"
DATA_DIR = data_dir

# ============================================================================
# 4. Data Loading Using Memory Mapping
# ============================================================================
tss = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tss.npy"), mmap_mode='r', allow_pickle=True)
tts = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tts.npy"), mmap_mode='r', allow_pickle=True)
TPM = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_TPM.npy"), mmap_mode='r', allow_pickle=True)
groups = np.load(os.path.join(DATA_DIR,"PCembed_Bdi_Osa_group_for_cross_validation.npy"), mmap_mode='r', allow_pickle=True)

print("Loaded shapes:")
print("tss:",    tss.shape)     # Expected: (N, 384, 20)
print("tts:",    tts.shape)     # Expected: (N, 384, 20)
print("TPM:",    TPM.shape)     # Expected: (N, )
print("groups:", groups.shape)  # Expected: (N, )

# Transform TPM values to log(1+TPM) in base 10.
TPM = np.log10(1 + TPM)

# Optionally load extra channels (a2z_preds or a2z_embeddings)
if use_extra:
    extra = np.load(os.path.join(DATA_DIR, "a2z_preds.npy"), mmap_mode='r', allow_pickle=True)
    print("Loaded extra channels:", extra.shape)
else:
    extra = None

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

base_output_dir = (
    f"/home/behrooz/WP2/Models/2BCNN/Allspecies_PCembed/val{val_group}_test{test_group}/"
)

CHECKPOINTS_DIR = base_output_dir
PLOTS_DIR = os.path.join(CHECKPOINTS_DIR, "plots")
storage_url = f"sqlite:////{base_output_dir}optuna_study_history.db"

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
    extra_mean = stats['extra_mean']
    extra_std  = stats['extra_std']
else:
    extra_mean = extra_std = None
stats.close()
print("Loaded global stats from", global_stats_file)

# Determine in_channels for the model
base_channels  = tss_mean.shape[1]            # 384
extra_channels = extra.shape[1] if use_extra else 0
in_channels    = base_channels + extra_channels

# ============================================================================
# 7. Create Dataset Instances
# ============================================================================
train_dataset = DNADualDataset(
    train_idx,
    tss, tts, TPM,
    tss_mean, tss_std,
    tts_mean, tts_std,
    extra = extra,
    extra_mean = extra_mean,
    extra_std = extra_std,
)
val_dataset = DNADualDataset(
    val_idx,
    tss, tts, TPM,
    tss_mean, tss_std,
    tts_mean, tts_std,
    extra = extra,
    extra_mean = extra_mean,
    extra_std = extra_std,
)

# ============================================================================
# 8. Define the Objective Function for Optuna with Early Stopping
# ============================================================================
def objective(trial):
    # DataLoaders for this trial
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    train_loader = torch.utils.data.DataLoader(
        DNADualDataset(
            train_idx,
            tss, tts, TPM,
            tss_mean, tss_std,
            tts_mean, tts_std,
            extra = extra,
            extra_mean = extra_mean,
            extra_std = extra_std,
        ), 
        batch_size = batch_size, shuffle = True)
    
    val_loader = torch.utils.data.DataLoader(
        DNADualDataset(
            val_idx,
            tss, tts, TPM,
            tss_mean, tss_std,
            tts_mean, tts_std,
            extra=extra,
            extra_mean=extra_mean,
            extra_std=extra_std,
        ),
        batch_size = batch_size, shuffle = False)

    model = TwoBranchCNN(trial, in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True))
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
        for x_tss, x_tts, target in train_loader:
            x_tss, x_tts = x_tss.to(device), x_tts.to(device)
            y = target.to(device).unsqueeze(1)
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
            for x_tss, x_tts, target in val_loader:
                x_tss, x_tts = x_tss.to(device), x_tts.to(device)
                y = target.to(device).unsqueeze(1)
                out = model(x_tss, x_tts)
                loss = criterion(out, y)
                val_loss += loss.item() * x_tss.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        rmse = val_loss ** 0.5

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, RMSE={rmse:.4f}")

        if val_loss < best_val_loss:
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

        if (epoch + 1 - best_epoch) >= lookahead_epochs:
            improvement = best_val_loss - val_loss
            if improvement < min_improvement:
                # print(
                #     f"Early stopping at epoch {epoch+1}: "
                #     f"improvement {improvement:.4f} < {min_improvement}"
                # )
                print(f"Early stopping triggered at epoch {epoch+1}: Improvement over best ({best_val_loss:.4f}) is only {improvement:.4f} (< {min_improvement}) after {lookahead_epochs} epochs.")
                trial.set_user_attr("early_stopped", True)
                break

        trial.report(val_loss, epoch)

    if best_checkpoint is not None:
        filename = f"checkpoint_trial_{trial.number}.pth"
        checkpoint_filename = os.path.join(CHECKPOINTS_DIR, filename)
        torch.save(best_checkpoint, checkpoint_filename)
        print(f"Saved best checkpoint for trial {trial.number} at epoch {best_epoch} to {checkpoint_filename}")

        # Save TorchScript version
        ts_model = TwoBranchCNN(trial, in_channels=in_channels).to(device)
        ts_model.load_state_dict(best_checkpoint['model_state_dict'])
        ts_model.eval()
        pt_filename = f"checkpoint_trial_{trial.number}.pt"
        pt_path = os.path.join(CHECKPOINTS_DIR, pt_filename)
        torch.jit.save(torch.jit.script(ts_model), pt_path)
        print(f"Saved TorchScript model for trial {trial.number} as {pt_path}")

    return best_val_loss

# ============================================================================
# 9. Run Hyperparameter Optimization with Optuna and Save Plots
# ============================================================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=f"PCembed_val{val_group}_test{test_group}_round8_Bdi_Osa",
        storage=storage_url,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)

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
        plt.title("Learning Curves for Best Trial - Standardized PC Embeddings - Single GPU")
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
