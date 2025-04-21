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
    """
    Dual-branch 1D CNN model for regression tasks on DNA sequence representations.
    
    This definition applies ReLU activation only after a convolutional layer.
    For each branch's ModuleList, the layers are added in pairs: Conv1d then BatchNorm1d 
    (if enabled). In the forward pass, ReLU is applied immediately after a convolutional 
    layer (i.e., for layers that are instances of nn.Conv1d), and NOT applied after the 
    corresponding BatchNorm layers.
    
    Dense layers (both per-branch and after concatenation) are followed by a ReLU.
    """
    def __init__(self, trial):
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
            conv = nn.Conv1d(in_channels, self.n_filters, kernel_size=self.kernel_size,
                             stride=1, dilation=1, padding=0)
            self.branch1_conv.append(conv)
            if self.batch_norm:
                # Append BatchNorm layer without following activation.
                self.branch1_conv.append(nn.BatchNorm1d(self.n_filters))
            in_channels = self.n_filters
        
        # --- Build Convolutional Layers for Branch 2 ---
        self.branch2_conv = nn.ModuleList()
        in_channels = 384
        for i in range(self.n_conv_layers):
            conv = nn.Conv1d(in_channels, self.n_filters, kernel_size=self.kernel_size,
                             stride=1, dilation=1, padding=0)
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
        # Process Branch 1.
        for layer in self.branch1_conv:
            # If the layer is a convolution, apply it then activation.
            if isinstance(layer, nn.Conv1d):
                x1 = layer(x1)
                x1 = self.activation(x1)
            else:
                # For BatchNorm, simply apply without additional activation.
                x1 = layer(x1)
        x1 = x1.view(x1.size(0), -1)
        for dense in self.branch1_dense:
            x1 = self.activation(dense(x1))
        
        # Process Branch 2.
        for layer in self.branch2_conv:
            if isinstance(layer, nn.Conv1d):
                x2 = layer(x2)
                x2 = self.activation(x2)
            else:
                x2 = layer(x2)
        x2 = x2.view(x2.size(0), -1)
        for dense in self.branch2_dense:
            x2 = self.activation(dense(x2))
        
        # Concatenate and process post-dense layers.
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
    
# Use this extended version of DNADualDataset class for generating predictions for each gene in the test set.
class DNADualDatasetMeta(DNADualDataset):
    """
    Extends DNADualDataset to also return gene_name and group_id.
    """
    def __init__(self, indices, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std,
                 gene, groups):
        super().__init__(indices, tss, tts, TPM, tss_mean, tss_std, tts_mean, tts_std)
        self.gene   = gene
        self.groups = groups

    def __getitem__(self, idx):
        tss_tensor, tts_tensor, target_tensor = super().__getitem__(idx)
        real_idx = self.indices[idx]
        gene_name = self.gene[real_idx]
        group_id  = self.groups[real_idx]
        return tss_tensor, tts_tensor, target_tensor, gene_name, group_id

# -----------------------------------------------------------------------
# Evaluation Utility
# -----------------------------------------------------------------------
def evaluate_model(model, dataloader, device, criterion):
    """
    Loop over dataloader, compute average loss and collect predictions.
    Returns (avg_loss, predictions_array).
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds = []
    with torch.no_grad():
        for x_tss, x_tts, target, *_ in dataloader:
            x_tss, x_tts = x_tss.to(device), x_tts.to(device)
            target = target.to(device).unsqueeze(1)
            output = model(x_tss, x_tts)
            loss = criterion(output, target)
            bs = x_tss.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            all_preds.append(output.cpu().numpy())
    avg_loss    = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    return avg_loss, predictions

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

# Function that shows whether GPU or CPU is used for training and/or inference.
def get_device():
    """Return the torch device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to split data into training-validation-test subsets using group numbers specified in command-line. 
def get_indices(val_group, test_group, groups):
    val_idx = np.where(groups == val_group)[0]
    test_idx = np.where(groups == test_group)[0]
    train_idx = np.where((groups != val_group) & (groups != test_group))[0]
    return train_idx, val_idx, test_idx

# Function to evaluate model(s) on test sets. Returns average loss and predictions array.
def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluate the model on a given DataLoader.

    Returns:
      - average loss over the dataset.
      - predictions as a numpy array.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for x_tss, x_tts, target, *rest in dataloader:
            x_tss = x_tss.to(device)
            x_tts = x_tts.to(device)
            target = target.to(device).unsqueeze(1)
            output = model(x_tss, x_tts)
            loss = criterion(output, target)
            batch_size = x_tss.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.append(output.cpu().numpy())

    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_predictions, axis=0)
    return avg_loss, predictions
