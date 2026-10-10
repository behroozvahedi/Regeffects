# utils.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler
import optuna

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
            extra_tss_sample = np.squeeze(extra_tss_sample, axis=0)  # shape: (C_extra, P)
            tss_sample = np.concatenate([tss_sample, extra_tss_sample], axis=0)   # shape: (1, C + C_extra, P)

        if self.extra_tts is not None:
            extra_tts_sample = np.array(self.extra_tts[real_idx])     # shape: (1, C_extra, P
            extra_tts_sample = (extra_tts_sample - self.extra_tts_mean) / self.extra_tts_std
            extra_tts_sample = np.squeeze(extra_tts_sample, axis=0)   # shape: (C_extra, P)
            tts_sample = np.concatenate([tts_sample, extra_tts_sample], axis=0)   # shape: (1, C + C_extra, P)

        # Load target
        target = self.TPM[real_idx]

        return (
            torch.tensor(tss_sample, dtype=torch.float32),
            torch.tensor(tts_sample, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

# ============================================================================
# 2.b Fast input pipeline: batch-level Dataset + GPU-side preprocessing
# ============================================================================
class DNADualBatchDataset(Dataset):
    """
    Batch-level counterpart of DNADualDataset.

    __getitem__ receives a *list* of positions (one whole batch) and returns the
    RAW, un-standardized, un-concatenated arrays for that batch. All arithmetic
    (standardization) and the channel concatenation are done afterwards on the
    GPU by BatchPreprocessor. Use it through make_loader(), which wires up the
    batch sampler.

    Works identically whether tss/tts/extra_* are np.memmap objects or ordinary
    in-RAM np.ndarrays.

    Returns:
      tss_parts: tuple of 1 or 2 tensors, (B, C, P) and optionally (B, C_extra, P)
      tts_parts: same for the TTS branch
      target:    float32 tensor of shape (B,)

    Note: indices are sorted within each batch (ascending file offsets -> better
    read locality on a memmap). The same sorted order is used for inputs and
    targets, so they stay aligned; for a sequential (non-shuffled) loader the
    sort is a no-op and sample order is unchanged.
    """
    def __init__(self, indices, tss, tts, TPM, *, extra_tss=None, extra_tts=None):
        self.indices   = np.asarray(indices)
        self.tss       = tss
        self.tts       = tts
        self.TPM       = TPM
        self.extra_tss = extra_tss
        self.extra_tts = extra_tts

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _take(arr, real_idx):
        # One fancy-indexing call = one copy of the whole batch out of the
        # memmap / RAM array. np.ascontiguousarray strips the memmap subclass.
        return torch.from_numpy(np.ascontiguousarray(arr[real_idx]))

    def __getitem__(self, batch_idx):
        real_idx = np.sort(self.indices[batch_idx])

        tss_parts = [self._take(self.tss, real_idx)]
        tts_parts = [self._take(self.tts, real_idx)]
        if self.extra_tss is not None:
            tss_parts.append(self._take(self.extra_tss, real_idx))
        if self.extra_tts is not None:
            tts_parts.append(self._take(self.extra_tts, real_idx))

        target = torch.from_numpy(np.asarray(self.TPM[real_idx], dtype=np.float32))
        return tuple(tss_parts), tuple(tts_parts), target


class BatchPreprocessor:
    """
    Moves a raw batch from DNADualBatchDataset to `device` and performs there
    exactly what DNADualDataset.__getitem__ does on the CPU:

      1. standardize the base channels with the base mean/std,
      2. standardize the extra channels with their own mean/std,
      3. cast each part to float32,
      4. concatenate base + extra along the channel dimension.

    Each part is standardized in the dtype of its own statistics (float64 stats
    -> float64 arithmetic, then rounded to float32), which is what NumPy does in
    DNADualDataset, so the resulting float32 values are the same.
    """
    def __init__(
        self, device,
        tss_mean, tss_std, tts_mean, tts_std,
        *,
        extra_tss_mean=None, extra_tss_std=None,
        extra_tts_mean=None, extra_tts_std=None,
        standardize=True,
    ):
        self.device      = device
        self.standardize = standardize

        def _stat(a):
            return torch.as_tensor(np.asarray(a)).to(device)

        self.tss_stats = [(_stat(tss_mean), _stat(tss_std))]
        self.tts_stats = [(_stat(tts_mean), _stat(tts_std))]
        if extra_tss_mean is not None:
            self.tss_stats.append((_stat(extra_tss_mean), _stat(extra_tss_std)))
        if extra_tts_mean is not None:
            self.tts_stats.append((_stat(extra_tts_mean), _stat(extra_tts_std)))

    def _prep(self, parts, stats):
        if len(parts) != len(stats):
            raise ValueError(
                f"Got {len(parts)} input part(s) but {len(stats)} set(s) of mean/std."
            )
        out = []
        for x, (mean, std) in zip(parts, stats):
            x = x.to(self.device, non_blocking=True)
            if self.standardize:
                x = (x - mean) / std
            out.append(x.to(torch.float32))
        return out[0] if len(out) == 1 else torch.cat(out, dim=1)

    def __call__(self, tss_parts, tts_parts, target):
        x_tss  = self._prep(tss_parts, self.tss_stats)
        x_tts  = self._prep(tts_parts, self.tts_stats)
        target = target.to(self.device, non_blocking=True)
        return x_tss, x_tts, target


class AdjustableBatchSampler(Sampler):
    """
    Groups the indices produced by `sampler` into lists of `batch_size`
    (the last, smaller batch is kept).

    Unlike torch's BatchSampler, the batch size is meant to be changed between
    epochs with set_batch_size(). The sampler lives in the main process and is
    re-read at the start of every epoch, so ONE DataLoader (one set of worker
    processes, one pin-memory thread) can serve every Optuna trial, whatever
    batch size the trial uses.
    """
    def __init__(self, sampler, batch_size):
        self.sampler    = sampler
        self.batch_size = int(batch_size)

    def __iter__(self):
        batch_size = self.batch_size          # fixed for the whole epoch
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def make_loader(dataset, batch_size, shuffle, num_workers=0, prefetch_factor=2):
    """
    Build a DataLoader that asks DNADualBatchDataset for one whole batch per call.

    IMPORTANT: build each loader ONCE per job and reuse it for all trials; use
    set_batch_size(loader, n) to change the batch size between trials. Creating
    new loaders (new worker processes + pin-memory threads) for every trial made
    the number of open files grow until the job hit the per-process limit.

    batch_size=None switches off the DataLoader's own per-sample collation; the
    AdjustableBatchSampler passed as `sampler` hands the dataset a list of positions.

    persistent_workers / prefetch_factor are only legal when num_workers > 0,
    so they are only passed in that case. Workers are started with "fork" so the
    memmaps (or in-RAM arrays) are inherited instead of being pickled.
    """
    base_sampler  = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    batch_sampler = AdjustableBatchSampler(base_sampler, batch_size)
    kwargs = dict(
        sampler    = batch_sampler,
        batch_size = None,
        pin_memory = torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs.update(
            num_workers             = num_workers,
            persistent_workers      = True,
            prefetch_factor         = prefetch_factor,
            multiprocessing_context = "fork",
        )
    return DataLoader(dataset, **kwargs)


def set_batch_size(loader, batch_size):
    """
    Change the batch size of a loader built with make_loader() or of a
    GPUResidentLoader. Takes effect at the next epoch (the next
    `for ... in loader`); call it between epochs only, i.e. at the start of a trial.
    """
    if isinstance(loader, GPUResidentLoader):
        loader.batch_size = int(batch_size)
        return
    if not isinstance(loader.sampler, AdjustableBatchSampler):
        raise TypeError("set_batch_size() needs a loader built with make_loader() or a GPUResidentLoader.")
    loader.sampler.batch_size = int(batch_size)


# ============================================================================
# 2.c GPU-resident data: the whole split lives on the GPU, no DataLoader at all
# ============================================================================
def gpu_resident_bytes(n_rows, n_channels, n_positions):
    """Bytes needed on the device for ONE split: two float32 branches (tss, tts) plus targets."""
    return 2 * n_rows * n_channels * n_positions * 4 + n_rows * 4


class GPUResidentLoader:
    """
    Holds one whole split (train or validation) on `device`, already standardized
    and channel-concatenated as float32, and serves batches by indexing those
    tensors on the device.

    Why: with a DataLoader, every batch is read from host RAM / disk by worker
    processes, handed to the main process, pinned and copied to the GPU. For a
    small model that delivery is the bottleneck, and it depends on the node's
    RAM and disk staying responsive. Here the data are copied to the GPU ONCE,
    when the loader is built; after that an epoch touches neither the disk, the
    host RAM, nor any worker process.

    The values are produced by the SAME BatchPreprocessor as the DataLoader path
    (chunk by chunk), so every sample is bit-identical to what that path feeds
    the model.

    Iterating yields (x_tss, x_tts, target), all on `device`:
      x_tss, x_tts: float32, shape (B, C_total, P)
      target:       float32, shape (B,)
    The last, smaller batch is kept. With shuffle=True a new random permutation
    is drawn every epoch.

    Args:
      indices:       sample indices of this split (rows of the arrays on disk).
      tss, tts, TPM: arrays (memmap or in RAM) as for DNADualBatchDataset.
      preprocessor:  a BatchPreprocessor built for `device`.
      batch_size:    initial batch size (change it with set_batch_size()).
      shuffle:       True for the training split, False for validation.
      chunk_rows:    rows copied to the device per step while loading.
    """
    def __init__(self, indices, tss, tts, TPM, preprocessor, *,
                 extra_tss=None, extra_tts=None,
                 batch_size=64, shuffle=False, chunk_rows=4096, name="split"):
        self.device     = preprocessor.device
        self.batch_size = int(batch_size)
        self.shuffle    = shuffle
        self.name       = name

        indices = np.sort(np.asarray(indices))      # ascending = sequential read of the files
        self.num_samples = n = len(indices)
        if n == 0:
            raise ValueError(f"GPUResidentLoader({name}): the split is empty.")

        raw = DNADualBatchDataset(indices, tss, tts, TPM, extra_tss=extra_tss, extra_tts=extra_tts)
        self.x_tss = self.x_tts = self.target = None
        with torch.no_grad():
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                tss_parts, tts_parts, target = raw[np.arange(start, stop)]
                x_tss, x_tts, target = preprocessor(tss_parts, tts_parts, target)
                if self.x_tss is None:
                    # Allocate the full tensors once, sized from the first chunk
                    self.x_tss  = torch.empty((n,) + tuple(x_tss.shape[1:]), dtype=torch.float32, device=self.device)
                    self.x_tts  = torch.empty((n,) + tuple(x_tts.shape[1:]), dtype=torch.float32, device=self.device)
                    self.target = torch.empty((n,), dtype=torch.float32, device=self.device)
                self.x_tss[start:stop]  = x_tss
                self.x_tts[start:stop]  = x_tts
                self.target[start:stop] = target
                del x_tss, x_tts, target, tss_parts, tts_parts

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.x_tss, self.x_tts, self.target))

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n, batch_size = self.num_samples, self.batch_size      # fixed for the whole epoch
        if self.shuffle:
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                yield (self.x_tss.index_select(0, idx),
                       self.x_tts.index_select(0, idx),
                       self.target.index_select(0, idx))
        else:
            for start in range(0, n, batch_size):
                stop = start + batch_size
                yield self.x_tss[start:stop], self.x_tts[start:stop], self.target[start:stop]


def check_gpu_capacity(device, needed_bytes, headroom_bytes=8 * 2**30):
    """
    Raise a clear error BEFORE loading if the data (plus headroom for the model,
    its activations and the loading chunks) cannot fit on the device.
    Returns (free_bytes, total_bytes); (None, None) on a CPU device.
    """
    if torch.device(device).type != "cuda":
        return None, None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if needed_bytes + headroom_bytes > free_bytes:
        raise RuntimeError(
            f"--data_on_gpu: the data need {needed_bytes / 2**30:.1f} GiB "
            f"(+ {headroom_bytes / 2**30:.0f} GiB headroom) but only {free_bytes / 2**30:.1f} GiB "
            f"of {total_bytes / 2**30:.1f} GiB are free on this GPU. "
            f"Run without --data_on_gpu (DataLoader path) or use a GPU with more memory."
        )
    return free_bytes, total_bytes


def raise_open_file_limit():
    """
    Raise this process's soft limit on open files to the hard limit (what
    `ulimit -n` shows is the soft limit, often only 1024). DataLoader workers hand
    every tensor to the main process through a file descriptor, so they need
    headroom. Returns (old_soft, new_soft, hard).
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = soft
    if hard == resource.RLIM_INFINITY or soft < hard:
        target = 1048576 if hard == resource.RLIM_INFINITY else hard
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            new_soft = target
        except (ValueError, OSError):
            pass
    return soft, new_soft, hard


def process_resource_report():
    """
    One-line summary of what the main process currently holds: open files (by
    kind), live child processes, and threads. Printed once per trial; every
    number should stay flat from trial to trial.
    """
    import multiprocessing, threading, collections
    kinds = collections.Counter()
    try:
        fd_names = os.listdir("/proc/self/fd")
    except OSError:
        return "open files: n/a"
    for fd in fd_names:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("pipe:"):
            kinds["pipes"] += 1
        elif target.startswith("socket:"):
            kinds["sockets"] += 1
        elif "nvidia" in target:
            kinds["nvidia"] += 1
        elif target.startswith("/dev/shm") or "torch_" in target:
            kinds["shared-mem"] += 1
        elif target.startswith("anon_inode:"):
            kinds["anon"] += 1
        else:
            kinds["files"] += 1
    total  = sum(kinds.values())
    detail = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    return (f"open files: {total} ({detail}) | "
            f"child processes: {len(multiprocessing.active_children())} | "
            f"threads: {threading.active_count()}")

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
# 3.b Model Definition for EMPRES 0 (OHE input)
# ============================================================================
class TwoBranchCNN_OHE(nn.Module):
    """
    Dual-branch 1D CNN for EMPRES 0 (one-hot-encoded DNA input).

    Same overall structure as TwoBranchCNN (two identical conv+dense branches,
    concatenation, post-merge dense layers, scalar output), but adapted for the
    long OHE input (input_length = 5000 per branch, 4 OHE channels). Each conv
    "layer" is: Conv1d(stride=1, padding='same', kernel=k, dilation=d)
    -> ReLU -> BatchNorm1d -> MaxPool1d(pool_size). The Conv -> ReLU -> BN
    order is kept identical to TwoBranchCNN (EMPRES 1-4); MaxPool1d is then
    appended after the BN to bring the long OHE input length down.

    Extra Optuna hyperparameters w.r.t. TwoBranchCNN: dilation, pool_size.
    Fixed (not searched): stride=1, padding='same'.

    A dimension precheck is performed in __init__: if the simulated length
    would fall below pool_size at any conv layer, optuna.TrialPruned() is
    raised so that the trial is pruned cleanly (no PyTorch RuntimeError).
    """
    def __init__(self, trial, in_channels: int, input_length: int):
        super().__init__()

        # --- Hyperparameters from trial ---
        self.n_conv_layers       = trial.suggest_categorical("n_conv_layers", [3, 4, 5])
        self.n_filters           = trial.suggest_categorical("n_filters", [64, 128, 192, 256, 320, 384])
        self.kernel_size         = trial.suggest_categorical("kernel_size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.dilation            = trial.suggest_categorical("dilation", [1, 2, 3, 4, 5])
        self.pool_size           = trial.suggest_categorical("pool_size", [3, 4, 5])
        self.n_dense_layers      = trial.suggest_categorical("n_dense_layers", [3, 4, 5])
        self.dense_units         = trial.suggest_categorical("dense_units", [16, 32, 64, 128])
        self.n_post_dense_layers = trial.suggest_categorical("n_post_dense_layers", [2, 3, 4])
        self.dropout_rate        = trial.suggest_float("dropout_rate", 0.0, 0.5)
        self.batch_norm          = trial.suggest_categorical("batch_norm", [True])
        self.activation          = nn.ReLU()

        self.in_channels  = in_channels    # 4 for OHE input
        self.input_length = input_length   # 5000 for EMPRES 0

        # --- Dimension precheck (avoid PyTorch RuntimeError; prune cleanly) ---
        # With padding='same' and stride=1, Conv1d preserves L. MaxPool1d with
        # kernel_size=pool_size (and default stride=pool_size) reduces L:
        #   L_out = floor((L_in - pool_size) / pool_size) + 1
        # If L_in < pool_size at any layer, MaxPool1d would crash; prune instead.
        length = input_length
        for _ in range(self.n_conv_layers):
            if length < self.pool_size:
                raise optuna.TrialPruned()
            length = (length - self.pool_size) // self.pool_size + 1
        length = max(length, 1)
        flat_size = self.n_filters * length

        # --- Build Branch 1 Convolutions ---
        # Per layer order matches EMPRES 1-4 (Conv -> ReLU -> BN), with MaxPool
        # appended after BN. ReLU is applied in forward(), only after Conv1d.
        self.branch1_conv = nn.ModuleList()
        cin = self.in_channels
        for _ in range(self.n_conv_layers):
            self.branch1_conv.append(
                nn.Conv1d(
                    cin, self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding='same',
                    dilation=self.dilation,
                )
            )
            if self.batch_norm:
                self.branch1_conv.append(nn.BatchNorm1d(self.n_filters))
            self.branch1_conv.append(nn.MaxPool1d(kernel_size=self.pool_size))
            cin = self.n_filters

        # --- Build Branch 2 Convolutions ---
        self.branch2_conv = nn.ModuleList()
        cin = self.in_channels
        for _ in range(self.n_conv_layers):
            self.branch2_conv.append(
                nn.Conv1d(
                    cin, self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding='same',
                    dilation=self.dilation,
                )
            )
            if self.batch_norm:
                self.branch2_conv.append(nn.BatchNorm1d(self.n_filters))
            self.branch2_conv.append(nn.MaxPool1d(kernel_size=self.pool_size))
            cin = self.n_filters

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
        # Branch 1: activation only after Conv1d (BatchNorm1d and MaxPool1d pass through)
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
def evaluate_model(model, dataloader, device, criterion, preprocessor=None):
    """
    Evaluate the model on a DataLoader and return
    (average_loss, predictions_array).

    preprocessor: None for a DataLoader over DNADualDataset (old behaviour), or
    a BatchPreprocessor for a loader built with make_loader() over
    DNADualBatchDataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for x_tss, x_tts, target in dataloader:
            if preprocessor is not None:
                x_tss, x_tts, target = preprocessor(x_tss, x_tts, target)
                target = target.unsqueeze(1)
            else:
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

# ============================================================================
# 6. EMPRES type configuration (single source of truth for per-type behavior)
# ============================================================================
# For each EMPRES type, this dict specifies:
#   base_tss_file, base_tts_file : canonical .npy filenames for the base branch input
#   extra_tss_file, extra_tts_file : optional extra .npy filenames (None if not used)
#   base_stats_keys  : 4-tuple of stat keys to read from global_stats_train_*.npz
#                      for the base branch (None if standardize is False)
#   extra_stats_keys : 4-tuple of stat keys for the extra channels (None if not used)
#   standardize      : if False, scripts skip the .npz load and pass zeros/ones
#                      (identity standardization) to DNADualDataset
#   subdir           : run-directory subfolder under out_dir/val{v}_test{t}/
#   study_tag        : Optuna study-name suffix
#   model_class      : model class to instantiate
#   model_kwargs     : extra constructor kwargs for the model class
EMPRES_CONFIG = {
    1: dict(
        base_tss_file   = "tss_embeddings_PlantCad.npy",
        base_tts_file   = "tts_embeddings_PlantCad.npy",
        extra_tss_file  = None,
        extra_tts_file  = None,
        base_stats_keys = ("tss_mean", "tss_std", "tts_mean", "tts_std"),
        extra_stats_keys= None,
        standardize     = True,
        subdir          = "EMPRES_1",
        study_tag       = "EMPRES_1",
        model_class     = TwoBranchCNN,
        model_kwargs    = {},
    ),
    2: dict(
        base_tss_file   = "tss_embeddings_PlantCad.npy",
        base_tts_file   = "tts_embeddings_PlantCad.npy",
        extra_tss_file  = "tss_predictions_a2z.npy",
        extra_tts_file  = "tts_predictions_a2z.npy",
        base_stats_keys = ("tss_mean", "tss_std", "tts_mean", "tts_std"),
        extra_stats_keys= ("tss_pred_mean", "tss_pred_std", "tts_pred_mean", "tts_pred_std"),
        standardize     = True,
        subdir          = "EMPRES_2",
        study_tag       = "EMPRES_2",
        model_class     = TwoBranchCNN,
        model_kwargs    = {},
    ),
    3: dict(
        base_tss_file   = "tss_embeddings_PlantCad.npy",
        base_tts_file   = "tts_embeddings_PlantCad.npy",
        extra_tss_file  = "tss_embeddings_a2z.npy",
        extra_tts_file  = "tts_embeddings_a2z.npy",
        base_stats_keys = ("tss_mean", "tss_std", "tts_mean", "tts_std"),
        extra_stats_keys= ("tss_emb_mean", "tss_emb_std", "tts_emb_mean", "tts_emb_std"),
        standardize     = True,
        subdir          = "EMPRES_3",
        study_tag       = "EMPRES_3",
        model_class     = TwoBranchCNN,
        model_kwargs    = {},
    ),
    4: dict(
        base_tss_file   = "tss_embeddings_a2z.npy",
        base_tts_file   = "tts_embeddings_a2z.npy",
        extra_tss_file  = None,
        extra_tts_file  = None,
        base_stats_keys = ("tss_emb_mean", "tss_emb_std", "tts_emb_mean", "tts_emb_std"),
        extra_stats_keys= None,
        standardize     = True,
        subdir          = "EMPRES_4",
        study_tag       = "EMPRES_4",
        model_class     = TwoBranchCNN,
        model_kwargs    = {},
    ),
    0: dict(
        base_tss_file   = "tss_OHE.npy",
        base_tts_file   = "tts_OHE.npy",
        extra_tss_file  = None,
        extra_tts_file  = None,
        base_stats_keys = None,
        extra_stats_keys= None,
        standardize     = False,
        subdir          = "EMPRES_0",
        study_tag       = "OHE",
        model_class     = TwoBranchCNN_OHE,
        model_kwargs    = {"input_length": 5000},
    ),
}
