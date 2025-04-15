import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import shutil
import optuna
from copy import deepcopy


# ============================================================================ 
# 1. TwoBranchCNN Model Class
# ============================================================================ 
class TwoBranchCNN(nn.Module):
    def __init__(self, trial):
        """
        Build a dual-branch 1D CNN model with input shape (384, 20) per branch.
        Uses valid padding (i.e. no padding) and kernel sizes chosen from [1,2,3,4].
        """
        super(TwoBranchCNN, self).__init__()
        
        # --- Convolutional Layer Hyperparameters ---
        self.n_conv_layers = trial.suggest_categorical("n_conv_layers", [3, 4, 5])
        self.n_filters = trial.suggest_categorical("n_filters", [128, 192, 256, 320, 384])
        self.kernel_size = trial.suggest_categorical("kernel_size", [1, 2, 3, 4])
        self.padding_type = 'valid'
        
        # --- Dense Layer Hyperparameters (per branch, before concatenation) ---
        self.n_dense_layers = trial.suggest_categorical("n_dense_layers", [3, 4, 5])
        self.dense_units = trial.suggest_categorical("dense_units", [16, 32, 64, 128])
        
        # --- Dense Layers After Concatenation ---
        self.n_post_dense_layers = trial.suggest_categorical("n_post_dense_layers", [2, 3, 4])
        
        # --- Regularization and Other ---
        self.dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5, log=False)
        self.batch_norm = trial.suggest_categorical("batch_norm", [True])
        self.activation = nn.ReLU()
        
        # --- Build Convolutional Layers for Branch 1 ---
        self.branch1_conv = nn.ModuleList()
        in_channels = 384
        for i in range(self.n_conv_layers):
            padding = 0  # valid padding
            conv = nn.Conv1d(in_channels, self.n_filters, kernel_size=self.kernel_size,
                             stride=1, dilation=1, padding=padding)
            self.branch1_conv.append(conv)
            if self.batch_norm:
                self.branch1_conv.append(nn.BatchNorm1d(self.n_filters))
            in_channels = self.n_filters
        
        # --- Build Convolutional Layers for Branch 2 ---
        self.branch2_conv = nn.ModuleList()
        in_channels = 384
        for i in range(self.n_conv_layers):
            padding = 0
            conv = nn.Conv1d(in_channels, self.n_filters, kernel_size=self.kernel_size,
                             stride=1, dilation=1, padding=padding)
            self.branch2_conv.append(conv)
            if self.batch_norm:
                self.branch2_conv.append(nn.BatchNorm1d(self.n_filters))
            in_channels = self.n_filters
        
        # --- Determine Output Length after Convolutions ---
        L_in = 20
        for i in range(self.n_conv_layers):
            L_in = L_in - (self.kernel_size - 1)
        L_out = max(L_in, 1)
        branch_output_size = self.n_filters * L_out
        
        # --- Build Dense Layers for Branch 1 ---
        self.branch1_dense = nn.ModuleList()
        in_features = branch_output_size
        for i in range(self.n_dense_layers):
            dense = nn.Linear(in_features, self.dense_units)
            self.branch1_dense.append(dense)
            in_features = self.dense_units
        
        # --- Build Dense Layers for Branch 2 ---
        self.branch2_dense = nn.ModuleList()
        in_features = branch_output_size
        for i in range(self.n_dense_layers):
            dense = nn.Linear(in_features, self.dense_units)
            self.branch2_dense.append(dense)
            in_features = self.dense_units
        
        # --- Build Dense Layers After Concatenation ---
        self.post_dense_layers = nn.ModuleList()
        in_features = 2 * self.dense_units
        for i in range(self.n_post_dense_layers):
            dense = nn.Linear(in_features, self.dense_units)
            self.post_dense_layers.append(dense)
            in_features = self.dense_units
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc_output = nn.Linear(in_features, 1)
    
    def forward(self, x1, x2):
        # Process Branch 1
        for layer in self.branch1_conv:
            # Apply layer and then activation regardless of layer type.
            if isinstance(layer, nn.Conv1d):
                x1 = layer(x1)
                x1 = self.activation(x1)
            else:
                x1 = layer(x1)
        x1 = x1.view(x1.size(0), -1)
        for dense in self.branch1_dense:
            x1 = self.activation(dense(x1))
        
        # Process Branch 2
        for layer in self.branch2_conv:
            if isinstance(layer, nn.Conv1d):
                x2 = layer(x2)
                x2 = self.activation(x2)
            else:
                x2 = layer(x2)
        x2 = x2.view(x2.size(0), -1)
        for dense in self.branch2_dense:
            x2 = self.activation(dense(x2))
        
        x = torch.cat((x1, x2), dim=1)
        for dense in self.post_dense_layers:
            x = self.activation(dense(x))
        x = self.dropout(x)
        x = self.fc_output(x)
        return x

# The dummy trial class used for instantiating and evaluating saved models with different hyperparameter values on validation/test datasets. 
class DummyTrial:
    """
    A simple dummy trial class for supplying fixed hyperparameter values.
    This can be used to instantiate the model with a given hyperparameter dictionary.
    """
    def __init__(self, hp_dict):
        self.hp = hp_dict
    def suggest_categorical(self, name, choices):
        val = self.hp[name]
        if val not in choices:
            raise ValueError(f"Value {val} for {name} not in allowed choices {choices}")
        return val
    def suggest_float(self, name, low, high, log=False):
        return self.hp[name]

# ============================================================================ 
# 2. Custom PyTorch Dataset (DNADualDataset)
# ============================================================================    
class DNADualDataset(Dataset):
    def __init__(self, indices, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std):
        """
        indices: array of sample indices for this split.
        tss, tts, TPM: memmapped arrays.
        tss_mean, tss_std, tts_mean, tts_std: global statistics for standardization.
        """
        self.indices = indices
        self.tss = tss
        self.tts = tts
        self.TPM = TPM
        self.tss_mean = tss_mean
        self.tss_std = tss_std
        self.tts_mean = tts_mean
        self.tts_std = tts_std

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        tss_sample = np.array(self.tss[real_idx])
        tts_sample = np.array(self.tts[real_idx])
        target = self.TPM[real_idx]
        
        # Standardize on the fly using global stats.
        tss_sample = (tss_sample - self.tss_mean) / self.tss_std
        tts_sample = (tts_sample - self.tts_mean) / self.tts_std

        # Remove the extra leading dimension.
        tss_sample = np.squeeze(tss_sample, axis=0)
        tts_sample = np.squeeze(tts_sample, axis=0)
        
        return (torch.tensor(tss_sample, dtype=torch.float32),
                torch.tensor(tts_sample, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))
    

# ============================================================================ 
# 3. helper Functions
# ============================================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the torch device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to split data using group numbers in the command-line. 
def get_indices(val_group, test_group, groups):
    val_idx = np.where(groups == val_group)[0]
    test_idx = np.where(groups == test_group)[0]
    train_idx = np.where((groups != val_group) & (groups != test_group))[0]
    return train_idx, val_idx, test_idx


# # Optuna Objective function for training the models and hyperparameter optimization with early stopping 
# def objective(trial):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create training and validation datasets using DNADualDataset.
#     train_dataset = DNADualDataset(train_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
#     val_dataset = DNADualDataset(val_idx, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
    
#     batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     model = TwoBranchCNN(trial).to(device)
    
#     lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
    
#     criterion = nn.MSELoss()
#     max_epochs = 50
    
#     lookahead_epochs = 10
#     min_improvement = 0.01
    
#     best_val_loss = float('inf')
#     best_epoch = 0
#     best_checkpoint = None
#     stopped_early = False
    
#     train_loss_history = []
#     val_loss_history = []
    
#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0.0
#         for x_tss, x_tts, target in train_loader:
#             x_tss, x_tts = x_tss.to(device), x_tts.to(device)
#             target = target.to(device).unsqueeze(1)
#             optimizer.zero_grad()
#             output = model(x_tss, x_tts)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * x_tss.size(0)
#         train_loss /= len(train_loader.dataset)
#         train_loss_history.append(train_loss)
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x_tss, x_tts, target in val_loader:
#                 x_tss, x_tts = x_tss.to(device), x_tts.to(device)
#                 target = target.to(device).unsqueeze(1)
#                 output = model(x_tss, x_tts)
#                 loss = criterion(output, target)
#                 val_loss += loss.item() * x_tss.size(0)
#         val_loss /= len(val_loader.dataset)
#         val_loss_history.append(val_loss)
#         rmse = val_loss ** 0.5
        
#         print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, RMSE={rmse:.4f}")
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_epoch = epoch + 1
#             best_checkpoint = {
#                 'trial_number': trial.number,
#                 'epoch': best_epoch,
#                 'model_state_dict': deepcopy(model.state_dict()),
#                 'optimizer_state_dict': deepcopy(optimizer.state_dict()),
#                 'hyperparameters': trial.params,
#                 'val_loss': val_loss,
#                 'RMSE': rmse,
#                 'train_loss_history': train_loss_history,
#                 'val_loss_history': val_loss_history,
#             }

#             # I can also save the best models immediately here each time.
        
#         if (epoch + 1 - best_epoch) >= lookahead_epochs:
#             improvement = best_val_loss - val_loss
#             if improvement < min_improvement:
#                 print(f"Early stopping triggered at epoch {epoch+1}: Improvement over best ({best_val_loss:.4f}) is only {improvement:.4f} (< {min_improvement}) after {lookahead_epochs} epochs.")
#                 trial.set_user_attr("early_stopped", True)
#                 stopped_early = True
#                 break
        
#         trial.report(val_loss, epoch)
    
#     if best_checkpoint is not None:
#         os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
#         if stopped_early:
#             filename = f"checkpoint_trial_{trial.number}_stopped_early.pth"
#         else:
#             filename = f"checkpoint_trial_{trial.number}.pth"
#         checkpoint_filename = os.path.join(CHECKPOINTS_DIR, filename)
#         torch.save(best_checkpoint, checkpoint_filename)
#         print(f"Saved best checkpoint for trial {trial.number} at epoch {best_epoch} to {checkpoint_filename}")
        
#         # Upload the saved checkpoint using Optuna's ArtifactStore per its documentation.
#         artifact_id = optuna.artifacts.upload_artifact(
#             artifact_store=artifact_store,
#             file_path=checkpoint_filename,
#             study_or_trial=trial.study,
#         )
#         trial.set_user_attr("artifact_id", artifact_id)
    
#     return best_val_loss 
