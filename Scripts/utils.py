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

