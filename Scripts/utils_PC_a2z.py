# utils.py

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ============================================================================
# 1. Reproducibility and Device Utilities
# ============================================================================
def set_random_seeds(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """
    Return the available device ('cuda' if available, else 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_indices(val_group, test_group, groups):
    """
    Given string labels for val_group and test_group and an array of group labels,
    return (train_idx, val_idx, test_idx) arrays of integer indices.
    """
    val_idx   = np.where(groups == val_group)[0]
    test_idx  = np.where(groups == test_group)[0]
    train_idx = np.where((groups != val_group) & (groups != test_group))[0]
    return train_idx, val_idx, test_idx

# ============================================================================
# 2. Dataset Classes
# ============================================================================
class DNADualDataset(Dataset):
    """
    PyTorch Dataset for dual-branch inputs (tss, tts), 
    with optional extra channels for tss and tts, each standardized by its own mean/std.

    Args:
      indices:        array of sample indices for this split.
      tss, tts:       memmapped arrays of shape (N, C, P).
      TPM:            array of shape (N,) of targets (log-transformed).
      tss_mean:       array of shape (1, C, P) for standardization.
      tss_std:        array of shape (1, C, P) for standardization.
      tts_mean:       array of shape (1, C, P) for standardization.
      tts_std:        array of shape (1, C, P) for standardization.
      extra_tss:      optional memmapped array of shape (N, C_extra, P) for TSS branch.
      extra_tss_mean: array of shape (1, C_extra, P) for standardization.
      extra_tss_std:  array of shape (1, C_extra, P) for standardization.
      extra_tts:      optional memmapped array of shape (N, C_extra, P) for TTS branch.
      extra_tts_mean: array of shape (1, C_extra, P) for standardization.
      extra_tts_std:  array of shape (1, C_extra, P) for standardization.
    """
    def __init__(
        self,
        indices,
        tss, tts, TPM,
        tss_mean, tss_std, tts_mean, tts_std,
        *,                           # ← enforces keyword-only for the extras
        extra_tss=None, extra_tts=None,
        extra_tss_mean=None, extra_tss_std=None,
        extra_tts_mean=None, extra_tts_std=None,
    ):
        self.indices        = indices
        self.tss            = tss
        self.tts            = tts
        self.TPM            = TPM
        self.tss_mean       = tss_mean
        self.tss_std        = tss_std
        self.tts_mean       = tts_mean
        self.tts_std        = tts_std
        self.extra_tss      = extra_tss
        self.extra_tts      = extra_tts
        self.extra_tss_mean = extra_tss_mean
        self.extra_tss_std  = extra_tss_std
        self.extra_tts_mean = extra_tts_mean
        self.extra_tts_std  = extra_tts_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx   = self.indices[idx]

        # Load and standardize tss
        tss_sample = np.array(self.tss[real_idx])      # shape (1, C, P)
        tss_sample = (tss_sample - self.tss_mean) / self.tss_std
        tss_sample = np.squeeze(tss_sample, axis=0)    # shape (C, P)

        # Load and standardize tts
        tts_sample = np.array(self.tts[real_idx])
        tts_sample = (tts_sample - self.tts_mean) / self.tts_std
        tts_sample = np.squeeze(tts_sample, axis=0)

        # If extra channels provided, standardize and concatenate along channels axis
        if self.extra_tss is not None:
            extra_tss_sample = np.array(self.extra_tss[real_idx])     # shape: (1, C_extra, P)
            extra_tss_sample = (extra_tss_sample - self.extra_tss_mean) / self.extra_tss_std
            extra_tss_sample = np.squeeze(extra_tss_sample, axis=0))  # shape: (C_extra, P)

        if self.extra_tts is not None:
            extra_tts_sample = np.array(self.extra_tts[real_idx])     # shape: (1, C_extra, P
            extra_tts_sample = (extra_tts_sample - self.extra_tts_mean) / self.extra_tts_std
            extra_tts_sample = np.squeeze(extra_tts_sample, axis=0)   # shape: (C_extra, P)

            # Concatenate along channel dimension
            tss_sample = np.concatenate([tss_sample, extra_tss_sample], axis=0)
            tts_sample = np.concatenate([tts_sample, extra_tts_sample], axis=0)

        # Load target
        target = self.TPM[real_idx]

        return (
            torch.tensor(tss_sample, dtype=torch.float32),
            torch.tensor(tts_sample, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

# ============================================================================
# 3. Model Definition
# ============================================================================
class TwoBranchCNN(nn.Module):
    """
    Dual-branch 1D CNN with configurable input channels and hyperparameters from an Optuna trial.
    """
    def __init__(self, trial, in_channels: int):
        super().__init__()

        # --- Hyperparameters from trial ---
        self.n_conv_layers       = trial.suggest_categorical("n_conv_layers", [3, 4, 5])
        self.n_filters           = trial.suggest_categorical("n_filters", [128, 192, 256, 320, 384])
        self.kernel_size         = trial.suggest_categorical("kernel_size", [1, 2, 3, 4])
        self.n_dense_layers      = trial.suggest_categorical("n_dense_layers", [3, 4, 5])
        self.dense_units         = trial.suggest_categorical("dense_units", [16, 32, 64, 128])
        self.n_post_dense_layers = trial.suggest_categorical("n_post_dense_layers", [2, 3, 4])
        self.dropout_rate        = trial.suggest_float("dropout_rate", 0.0, 0.5)
        self.batch_norm          = trial.suggest_categorical("batch_norm", [True])
        self.activation          = nn.ReLU()

        self.in_channels = in_channels  # e.g. 384 or 384 + C_extra

        # --- Build Branch 1 Convolutions ---
        self.branch1_conv = nn.ModuleList()
        cin = self.in_channels
        for _ in range(self.n_conv_layers):
            self.branch1_conv.append(
                nn.Conv1d(cin, self.n_filters, kernel_size=self.kernel_size)
            )
            if self.batch_norm:
                self.branch1_conv.append(nn.BatchNorm1d(self.n_filters))
            cin = self.n_filters

        # --- Build Branch 2 Convolutions ---
        self.branch2_conv = nn.ModuleList()
        cin = self.in_channels
        for _ in range(self.n_conv_layers):
            self.branch2_conv.append(
                nn.Conv1d(cin, self.n_filters, kernel_size=self.kernel_size)
            )
            if self.batch_norm:
                self.branch2_conv.append(nn.BatchNorm1d(self.n_filters))
            cin = self.n_filters

        # Compute output length after conv layers
        length = 20
        for _ in range(self.n_conv_layers):
            length -= (self.kernel_size - 1)
        length = max(length, 1)
        flat_size = self.n_filters * length

        # --- Build Branch 1 Dense Layers ---
        self.branch1_dense = nn.ModuleList()
        fin = flat_size
        for _ in range(self.n_dense_layers):
            self.branch1_dense.append(nn.Linear(fin, self.dense_units))
            fin = self.dense_units

        # --- Build Branch 2 Dense Layers ---
        self.branch2_dense = nn.ModuleList()
        fin = flat_size
        for _ in range(self.n_dense_layers):
            self.branch2_dense.append(nn.Linear(fin, self.dense_units))
            fin = self.dense_units

        # --- Build Post-Concatenation Dense Layers ---
        self.post_dense_layers = nn.ModuleList()
        fin = 2 * self.dense_units
        for _ in range(self.n_post_dense_layers):
            self.post_dense_layers.append(nn.Linear(fin, self.dense_units))
            fin = self.dense_units

        self.dropout   = nn.Dropout(self.dropout_rate)
        self.fc_output = nn.Linear(fin, 1)

    def forward(self, x1, x2):
        # Branch 1
        for layer in self.branch1_conv:
            if isinstance(layer, nn.Conv1d):
                x1 = self.activation(layer(x1))
            else:
                x1 = layer(x1)
        x1 = x1.view(x1.size(0), -1)
        for dense in self.branch1_dense:
            x1 = self.activation(dense(x1))

        # Branch 2
        for layer in self.branch2_conv:
            if isinstance(layer, nn.Conv1d):
                x2 = self.activation(layer(x2))
            else:
                x2 = layer(x2)
        x2 = x2.view(x2.size(0), -1)
        for dense in self.branch2_dense:
            x2 = self.activation(dense(x2))

        # Concatenate and post-processing
        x = torch.cat((x1, x2), dim=1)
        for dense in self.post_dense_layers:
            x = self.activation(dense(x))
        x = self.dropout(x)
        return self.fc_output(x)

# ============================================================================
# 4. Evaluation Utility
# ============================================================================
def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluate the model on a DataLoader and return
    (average_loss, predictions_array).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for x_tss, x_tts, target in dataloader:
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

# ============================================================================
# 5. DummyTrial for Fixed Hyperparameters
# ============================================================================
class DummyTrial:
    """
    Supplies fixed hyperparameter values via suggest_* methods.
    """
    def __init__(self, hp_dict):
        self.hp = hp_dict

    def suggest_categorical(self, name, choices):
        val = self.hp[name]
        if val not in choices:
            raise ValueError(f"{name}={val} not in {choices}")
        return val

    def suggest_float(self, name, low, high, log=False):
        return self.hp[name]

