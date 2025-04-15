import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import shutil
import plotly
import sklearn
from copy import deepcopy

# Import shared utilities from utils.py.
from utils import set_random_seeds, get_device, get_indices, DNADualDataset, TwoBranchCNN

# ============================================================================
# 1. Set Random Seeds and Global Settings
# ============================================================================
set_random_seeds(42)
device = get_device()
print("device:", device)

# ============================================================================
# Parse Command-Line Arguments with argparse
# ============================================================================
parser = argparse.ArgumentParser(description="Train Dual-Branch CNN with Hyperparameter Optimization")
parser.add_argument("--val_group", type=str, required=False, default="4",
                    help="Validation group number (default: 4)")
parser.add_argument("--test_group", type=str, required=False, default="5",
                    help="Test group number (default: 5)")
args = parser.parse_args()
val_group = args.val_group
test_group = args.test_group
print(f"Using validation group: {val_group} and test group: {test_group}")

# ============================================================================
# Global directory for input data
# ============================================================================
DATA_DIR = "/home/behrooz/WP2/Datasets/PC_Embeddings/npy_files/Bdi_Osa"

# ============================================================================
# 2. Data Loading Using Memory Mapping
# ============================================================================
tss = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tss.npy"), mmap_mode='r', allow_pickle=True)
tts = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_tts.npy"), mmap_mode='r', allow_pickle=True)
TPM = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_TPM.npy"), mmap_mode='r', allow_pickle=True)
groups = np.load(os.path.join(DATA_DIR, "PCembed_Bdi_Osa_group_for_cross_validation.npy"), mmap_mode='r', allow_pickle=True)

print("Loaded shapes:")
print("tss:", tss.shape)      # Expected: (N, 384, 20)
print("tts:", tts.shape)
print("TPM:", TPM.shape)
print("groups:", groups.shape)

# Transform TPM values to log(1+TPM) in base 10.
TPM = np.log10(1 + TPM)

# ============================================================================
# 3. Cross-Validation Splitting and Output Directories
# ============================================================================
train_idx, val_idx, test_idx = get_indices(val_group, test_group, groups)
print("Fold split:")
print("  Validation group:", val_group)
print("  Test group:", test_group)
train_groups = np.unique(groups[train_idx])
print("  Training groups:", train_groups)

# Dynamically build the output directories.
base_output_dir = f"/home/behrooz/WP2/Models/2BCNN/Allspecies_PCembed/val{val_group}_test{test_group}/round8_Bdi_Osa/"
CHECKPOINTS_DIR = base_output_dir
PLOTS_DIR = os.path.join(CHECKPOINTS_DIR, "plots")
storage_url = f"sqlite:////{base_output_dir}optuna_study_history.db"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# 4. Load Global Statistics for Standardization
# ============================================================================
train_groups_sorted = np.sort(train_groups)
train_groups_str = "_".join(map(str, train_groups_sorted))
global_stats_file = os.path.join(DATA_DIR, f"global_stats_train_{train_groups_str}.npz")
stats = np.load(global_stats_file)
tss_mean = stats['tss_mean']  # shape: (1, 384, 20)
tss_std = stats['tss_std']    # shape: (1, 384, 20)
tts_mean = stats['tts_mean']  # shape: (1, 384, 20)
tts_std = stats['tts_std']    # shape: (1, 384, 20)
stats.close()
print("Loaded global stats from", global_stats_file)

# ============================================================================
# 5. Create Dataset Instances
# ============================================================================
train_dataset = DNADualDataset(train_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
val_dataset = DNADualDataset(val_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
test_dataset = DNADualDataset(test_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)

# ============================================================================
# 6. Define the Objective Function for Optuna with Early Stopping
# ============================================================================
def objective(trial):
    device = get_device()
    # Create training and validation datasets.
    train_dataset = DNADualDataset(train_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
    val_dataset = DNADualDataset(val_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
    
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model using TwoBranchCNN from utils.
    model = TwoBranchCNN(trial).to(device)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    max_epochs = 50
    lookahead_epochs = 10
    min_improvement = 0.01
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_checkpoint = None
    stopped_early = False
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for x_tss, x_tts, target in train_loader:
            x_tss, x_tts = x_tss.to(device), x_tts.to(device)
            target = target.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(x_tss, x_tts)
            loss = criterion(output, target)
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
                target = target.to(device).unsqueeze(1)
                output = model(x_tss, x_tts)
                loss = criterion(output, target)
                val_loss += loss.item() * x_tss.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        rmse = val_loss ** 0.5
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, RMSE={rmse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
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
                print(f"Early stopping triggered at epoch {epoch+1}: Improvement over best ({best_val_loss:.4f}) is only {improvement:.4f} (< {min_improvement}) after {lookahead_epochs} epochs.")
                trial.set_user_attr("early_stopped", True)
                stopped_early = True
                break
        
        trial.report(val_loss, epoch)
    
    if best_checkpoint is not None:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        filename = f"checkpoint_trial_{trial.number}.pth"
        checkpoint_filename = os.path.join(CHECKPOINTS_DIR, filename)
        torch.save(best_checkpoint, checkpoint_filename)
        print(f"Saved best checkpoint for trial {trial.number} at epoch {best_epoch} to {checkpoint_filename}")
        
        # Upload the saved checkpoint using Optuna's ArtifactStore.
        artifact_store_instance = optuna.artifacts.FileSystemArtifactStore(base_path=os.path.join(CHECKPOINTS_DIR, "artifacts"))
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store_instance,
            file_path=checkpoint_filename,
            study_or_trial=trial.study,
        )
        trial.set_user_attr("artifact_id", artifact_id)
    
    return best_val_loss

# ============================================================================
# 7. Run Hyperparameter Optimization with Optuna and Save Plots
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
    # if not os.path.exists(candidate_filename):
    #     candidate_filename = os.path.join(CHECKPOINTS_DIR, f"checkpoint_trial_{best_trial.number}.pth")
    best_model_filename = candidate_filename
    
    best_checkpoint = torch.load(best_model_filename, map_location=device)
    best_epoch = best_checkpoint.get("epoch", "N/A")
    print(f"  Best epoch: {best_epoch}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    complete_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    pruned_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
    early_stopped_trials = sum(1 for t in study.trials if t.user_attrs.get("early_stopped", False))
    print(f"Total completed trials: {complete_trials}")
    print(f"Total pruned trials: {pruned_trials}")
    print(f"Total early stopped trials: {early_stopped_trials}")
    
    history_fig = optuna.visualization.plot_optimization_history(study)
    parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
    param_fig = optuna.visualization.plot_param_importances(study)
    slice_fig = optuna.visualization.plot_slice(study)
    
    history_fig.update_layout(width=1200, height=800)
    parallel_fig.update_layout(width=1200, height=800)
    param_fig.update_layout(width=1200, height=800)
    slice_fig.update_layout(width=1200, height=800)
    
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
    
    checkpoint = torch.load(overall_best_filename, map_location=device)
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    if train_loss_history and val_loss_history:
        epochs_range = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, train_loss_history, marker='o', label='Training Loss')
        plt.plot(epochs_range, val_loss_history, marker='o', label='Validation Loss')
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

