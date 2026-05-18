#!/usr/bin/env python
import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

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

def main():
    # ------------------------------------------------------------------------
    # 1. Parse CLI (same as test_PC_a2z.py)
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Generate per gene predictions for top models + ensemble"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory where input .npy/.npz files are located")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Base output dir (must match train.py --out_dir)")
    parser.add_argument("--val_group", type=str, required=True,
                        help="Validation group ID (e.g. '4')")
    parser.add_argument("--test_group", type=str, required=True,
                        help="Test group ID (e.g. '5')")
    parser.add_argument("--top_x", type=int, default=1,
                        help="Number of top models to evaluate")
    parser.add_argument("--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help="EMPRES model type (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)")
    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # 2. Set seeds & device
    # ------------------------------------------------------------------------
    set_random_seeds(42)
    device = get_device()

    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    VAL_GROUP = args.val_group
    TEST_GROUP = args.test_group
    TOP_X = args.top_x
    EMPRES_type = args.EMPRES_type
    cfg = EMPRES_CONFIG[EMPRES_type]

    # ------------------------------------------------------------------------
    # 3. Load raw arrays (mmap) & groups & gene/family names
    # ------------------------------------------------------------------------
    tss = np.load(os.path.join(DATA_DIR, cfg["base_tss_file"]), mmap_mode="r", allow_pickle=True)
    tts = np.load(os.path.join(DATA_DIR, cfg["base_tts_file"]), mmap_mode="r", allow_pickle=True)
    TPM = np.load(os.path.join(DATA_DIR, "TPM.npy"), mmap_mode="r", allow_pickle=True)
    groups = np.load(os.path.join(DATA_DIR, "group_for_cross_validation.npy"), mmap_mode="r", allow_pickle=True)

    # load gene & family string arrays
    gene = np.load(os.path.join(DATA_DIR, "gene.npy"), allow_pickle=True)
    family = np.load(os.path.join(DATA_DIR, "family.npy"), allow_pickle=True)

    TPM = np.log10(1 + TPM)

    if cfg["extra_tss_file"] is not None:
        extra_tss = np.load(os.path.join(DATA_DIR, cfg["extra_tss_file"]), mmap_mode="r", allow_pickle=True)
        extra_tts = np.load(os.path.join(DATA_DIR, cfg["extra_tts_file"]), mmap_mode="r", allow_pickle=True)
    else:
        extra_tss = extra_tts = None

    # ------------------------------------------------------------------------
    # 4. CV split
    # ------------------------------------------------------------------------
    train_idx, val_idx, test_idx = get_indices(VAL_GROUP, TEST_GROUP, groups)
    train_groups = np.unique(groups[train_idx])
    train_groups_str = "_".join(map(str, np.sort(train_groups)))

    # ------------------------------------------------------------------------
    # 5. Load standardization stats
    # ------------------------------------------------------------------------
    if cfg["standardize"]:
        stats_path = os.path.join(
            DATA_DIR, f"global_stats_train_{train_groups_str}.npz"
        )
        stats = np.load(stats_path)
        base_keys = cfg["base_stats_keys"]
        tss_mean, tss_std = stats[base_keys[0]], stats[base_keys[1]]
        tts_mean, tts_std = stats[base_keys[2]], stats[base_keys[3]]

        if cfg["extra_stats_keys"] is not None:
            extra_keys = cfg["extra_stats_keys"]
            extra_tss_mean = stats[extra_keys[0]]
            extra_tss_std  = stats[extra_keys[1]]
            extra_tts_mean = stats[extra_keys[2]]
            extra_tts_std  = stats[extra_keys[3]]
        else:
            extra_tss_mean = extra_tss_std = extra_tts_mean = extra_tts_std = None
        stats.close()
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

    base_channels = tss_mean.shape[1]
    extra_channels = extra_tss.shape[1] if extra_tss is not None else 0
    in_channels = base_channels + extra_channels

    # ------------------------------------------------------------------------
    # 6. Build Test DataLoader once
    # ------------------------------------------------------------------------
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
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ------------------------------------------------------------------------
    # 7. Locate top-X trials CSV
    # ------------------------------------------------------------------------
    run_dir = os.path.join(
        OUT_DIR,
        f"val{VAL_GROUP}_test{TEST_GROUP}",
        cfg["subdir"],
    )
    CHECKPOINTS_DIR = run_dir
    TRIALS_CSV = os.path.join(CHECKPOINTS_DIR, "trial_results.csv")

    df_trials = pd.read_csv(TRIALS_CSV)
    top_trials = df_trials.head(TOP_X).reset_index(drop=True)

    # ------------------------------------------------------------------------
    # 8. Evaluate each of the top-X models
    # ------------------------------------------------------------------------
    all_predictions = []
    criterion = nn.MSELoss()

    for _, row in top_trials.iterrows():
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
        trial_obj = DummyTrial(hp)

        model = cfg["model_class"](trial_obj, in_channels=in_channels, **cfg["model_kwargs"]).to(device)
        checkpoint_file = row["checkpoint_file"].strip()
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _, preds = evaluate_model(model, test_loader, device, criterion)
        all_predictions.append(preds.squeeze())

    # ------------------------------------------------------------------------
    # 9. Ensemble & assemble DataFrame
    # ------------------------------------------------------------------------
    pred_stack = np.stack(all_predictions, axis=0)
    ensemble_pred = pred_stack.mean(axis=0)

    output_df = pd.DataFrame({
        "gene": gene[test_idx],
        "family": family[test_idx],
        "group": groups[test_idx],
        "observed_TPM": TPM[test_idx],
        "ensemble_pred": ensemble_pred,
    })
    for j, preds in enumerate(all_predictions, start=1):
        output_df[f"model_{j}_pred"] = preds

    # reorder columns
    cols = ["gene","family","group"] + \
           [f"model_{i}_pred" for i in range(1, TOP_X+1)] + \
           ["ensemble_pred","observed_TPM"]
    output_df = output_df[cols]

    # ------------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    out_csv = os.path.join(CHECKPOINTS_DIR, "predictions.csv")
    output_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    main()
