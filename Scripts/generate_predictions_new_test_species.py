#!/usr/bin/env python
"""
generate_predictions_new_test_species.py

Cross-species inference script for the EMPRES models.

For an entirely held-out new-species dataset (no CV split is applied, because
none of these species were used during EMPRES training or HPO), this script
runs the top-X EMPRES models from each of the 5 cross-validation folds and
stores, per gene, ONE ensemble prediction per fold.

The output mirrors the column layout of PhytoExpr's predictions file
(reviews_pred_out_modelBC.csv) so that downstream comparison between
EMPRES and PhytoExpr is row-aligned. Two output CSVs are written:

  1. A standalone EMPRES predictions file in array_index order (i.e.
     per-species blocks, matching the row order of the input .npy files).
  2. A joined file that merges the 5 EMPRES per-fold ensemble columns onto
     the PhytoExpr predictions file, on (gene, species, transcript).

If row counts or join keys differ between EMPRES and PhytoExpr, the script
prints warnings and still writes the joined file (NaN for unmatched rows).

Both output prediction values are kept in log10(1 + TPM) space, matching
what the EMPRES models output.

CLI mirrors test_PC_a2z.py, generate_predictions.py, model_info_df.py,
and aggregate_predictions.py.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils_PC_a2z import (
    set_random_seeds,
    get_device,
    DNADualDataset,
    TwoBranchCNN,
    TwoBranchCNN_OHE,
    DummyTrial,
    evaluate_model,
    EMPRES_CONFIG,
)


def main():
    # ------------------------------------------------------------------
    # 1) Parse CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-fold EMPRES ensemble predictions for an entirely "
            "held-out new-species dataset, save them, and merge them onto "
            "the PhytoExpr predictions file."
        )
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help=(
            "Directory containing the new-species input files "
            "(tss_OHE.npy, tts_OHE.npy, etc., depending on --EMPRES_type) "
            "and clean_metadata_exact_10000_original_5000_extracted.csv."
        ),
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help=(
            "Base output directory of the trained EMPRES models, containing "
            "val{V}_test{T}/{subdir}/trial_results.csv and the corresponding "
            ".pth checkpoints. This directory is read only; nothing is written here."
        ),
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help=(
            "Directory in which to save the standalone and joined output CSVs. "
            "Also the default location to look for the PhytoExpr predictions file."
        ),
    )
    parser.add_argument(
        "--phyto_pred_file", type=str, default=None,
        help=(
            "Full path to PhytoExpr's predictions CSV "
            "(default: {save_dir}/reviews_pred_out_modelBC.csv)."
        ),
    )
    parser.add_argument(
        "--global_stats_dir", type=str, default=None,
        help=(
            "Directory containing global_stats_train_*.npz files for the "
            "fold-specific standardization. Only used when cfg['standardize'] "
            "is True (i.e. EMPRES 1-4). Defaults to --data_dir."
        ),
    )
    parser.add_argument(
        "--top_x", type=int, default=5,
        help="Number of top models to ensemble per fold (default: 5).",
    )
    parser.add_argument(
        "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
        help=(
            "EMPRES model type (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)."
        ),
    )
    parser.add_argument(
        "--join_only", action="store_true",
        help=(
            "Skip model inference; load an existing standalone EMPRES CSV from "
            "--save_dir and only build the joined PhytoExpr output file."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    save_dir = args.save_dir
    phyto_pred_file = (
        args.phyto_pred_file
        if args.phyto_pred_file is not None
        else os.path.join(save_dir, "reviews_pred_out_modelBC.csv")
    )
    global_stats_dir = (
        args.global_stats_dir if args.global_stats_dir is not None else data_dir
    )
    top_x = args.top_x
    EMPRES_type = args.EMPRES_type
    join_only = args.join_only
    cfg = EMPRES_CONFIG[EMPRES_type]
    subdir = cfg["subdir"]
    empres_cols = [
        f"EMPRES_{EMPRES_type}_pred_median_TPM{fold_num}" for fold_num in range(1, 6)
    ]

    os.makedirs(save_dir, exist_ok=True)

    print("Configuration:")
    print(f"  DATA_DIR         = {data_dir}")
    print(f"  OUT_DIR          = {out_dir}  (read only)")
    print(f"  SAVE_DIR         = {save_dir}")
    print(f"  PHYTO_PRED_FILE  = {phyto_pred_file}")
    print(f"  GLOBAL_STATS_DIR = {global_stats_dir}  (used only when standardize=True)")
    print(f"  TOP_X            = {top_x}")
    print(f"  EMPRES_TYPE      = {EMPRES_type}  (subdir='{subdir}')")
    print(f"  JOIN_ONLY        = {join_only}")

    if join_only:
        standalone_csv = os.path.join(
            save_dir, f"EMPRES_{EMPRES_type}_pred_new_test_species.csv"
        )
        if not os.path.exists(standalone_csv):
            raise FileNotFoundError(
                f"--join_only requested but standalone file not found: {standalone_csv}"
            )
        out_df = pd.read_csv(standalone_csv)
        print(f"\nLoaded existing standalone EMPRES predictions: {standalone_csv}")
        print(f"  Shape: {out_df.shape}")
        missing_cols = [c for c in empres_cols if c not in out_df.columns]
        if missing_cols:
            raise ValueError(
                f"Standalone file is missing expected prediction columns: {missing_cols}"
            )
        n_samples = len(out_df)
    else:
        # ------------------------------------------------------------------
        # 2) Seeds & device
        # ------------------------------------------------------------------
        set_random_seeds(42)
        device = get_device()
        print(f"device: {device}")

        # ------------------------------------------------------------------
        # 3) Load input .npy arrays (memory-mapped)
        # ------------------------------------------------------------------
        tss_path = os.path.join(data_dir, cfg["base_tss_file"])
        tts_path = os.path.join(data_dir, cfg["base_tts_file"])
        tss = np.load(tss_path, mmap_mode="r", allow_pickle=True)
        tts = np.load(tts_path, mmap_mode="r", allow_pickle=True)
        print(f"\nLoaded {cfg['base_tss_file']}: shape {tss.shape}")
        print(f"Loaded {cfg['base_tts_file']}: shape {tts.shape}")
        n_samples = tss.shape[0]
        if tts.shape[0] != n_samples:
            raise ValueError(
                f"tss and tts row counts disagree: tss={tss.shape[0]}, tts={tts.shape[0]}"
            )

        if cfg["extra_tss_file"] is not None:
            extra_tss = np.load(
                os.path.join(data_dir, cfg["extra_tss_file"]),
                mmap_mode="r", allow_pickle=True,
            )
            extra_tts = np.load(
                os.path.join(data_dir, cfg["extra_tts_file"]),
                mmap_mode="r", allow_pickle=True,
            )
            if extra_tss.shape[0] != n_samples or extra_tts.shape[0] != n_samples:
                raise ValueError(
                    "Extra TSS/TTS arrays do not match base TSS/TTS row count."
                )
        else:
            extra_tss = extra_tts = None

        # ------------------------------------------------------------------
        # 4) Load and sort metadata CSV by array_index
        #    (CRITICAL: the OHE-generation script writes metadata in
        #    chunk x species order, while .npy rows are in species x chunk
        #    order. The 'array_index' column is the canonical link between
        #    metadata rows and .npy rows. We sort by it here so that
        #    meta.iloc[i] corresponds to tss[i] and tts[i] for every i.)
        # ------------------------------------------------------------------
        metadata_path = os.path.join(
            data_dir, "clean_metadata_exact_10000_original_5000_extracted.csv"
        )
        meta = pd.read_csv(metadata_path)
        print(
            f"\nLoaded metadata: {len(meta):,} rows; columns: {list(meta.columns)}"
        )
        if "array_index" not in meta.columns:
            raise ValueError(
                f"metadata CSV is missing required 'array_index' column: {metadata_path}"
            )
        if len(meta) != n_samples:
            raise ValueError(
                f"Metadata row count ({len(meta):,}) does not match .npy row count "
                f"({n_samples:,})."
            )

        meta = meta.sort_values("array_index").reset_index(drop=True)
        if not np.array_equal(meta["array_index"].to_numpy(), np.arange(n_samples)):
            raise ValueError(
                "After sorting by array_index, expected contiguous range [0, N-1] not found."
            )
        if "Unnamed: 0" in meta.columns:
            meta = meta.drop(columns=["Unnamed: 0"])

        # ------------------------------------------------------------------
        # 5) Build target array for DNADualDataset.
        # ------------------------------------------------------------------
        median_TPM_raw = meta["median_TPM"].to_numpy(dtype=np.float64)
        valid_mask = ~np.isnan(median_TPM_raw)
        log_observed = np.full(n_samples, np.nan, dtype=np.float32)
        log_observed[valid_mask] = np.log10(1.0 + median_TPM_raw[valid_mask]).astype(np.float32)
        dataset_target = np.nan_to_num(log_observed, nan=0.0).astype(np.float32)

        # ------------------------------------------------------------------
        # 6) Per-fold inference loop (all 5 folds in one script invocation)
        # ------------------------------------------------------------------
        FOLDS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        indices_all = np.arange(n_samples)
        fold_ensemble_preds = {}
        criterion = nn.MSELoss()

        for fold_num, (val_group, test_group) in enumerate(FOLDS, start=1):
            print()
            print("=" * 75)
            print(
                f"Fold {fold_num}: val_group={val_group}, test_group={test_group}, "
                f"EMPRES_type={EMPRES_type}, top_x={top_x}"
            )
            print("=" * 75)

            run_dir = os.path.join(out_dir, f"val{val_group}_test{test_group}", subdir)
            trials_csv = os.path.join(run_dir, "trial_results.csv")
            if not os.path.exists(trials_csv):
                raise FileNotFoundError(f"trial_results.csv not found: {trials_csv}")
            df_trials = pd.read_csv(trials_csv)
            top_trials = df_trials.head(top_x).reset_index(drop=True)

            if cfg["standardize"]:
                all_groups = ["1", "2", "3", "4", "5"]
                train_groups = sorted(
                    set(all_groups) - {str(val_group), str(test_group)}, key=int
                )
                train_groups_str = "_".join(train_groups)
                stats_path = os.path.join(
                    global_stats_dir, f"global_stats_train_{train_groups_str}.npz"
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
                print(f"Loaded global stats from {stats_path}")
            else:
                base_C = tss.shape[1]
                base_L = tss.shape[2]
                tss_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
                tss_std  = np.ones((1, base_C, base_L), dtype=np.float32)
                tts_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
                tts_std  = np.ones((1, base_C, base_L), dtype=np.float32)
                extra_tss_mean = extra_tss_std = extra_tts_mean = extra_tts_std = None
                print("Skipped global stats load (identity standardization for OHE input).")

            base_channels = tss_mean.shape[1]
            extra_channels = extra_tss.shape[1] if extra_tss is not None else 0
            in_channels = base_channels + extra_channels

            dataset = DNADualDataset(
                indices_all,
                tss, tts, dataset_target,
                tss_mean, tss_std,
                tts_mean, tts_std,
                extra_tss = extra_tss,
                extra_tss_mean = extra_tss_mean,
                extra_tss_std = extra_tss_std,
                extra_tts = extra_tts,
                extra_tts_mean = extra_tts_mean,
                extra_tts_std = extra_tts_std,
            )
            loader = DataLoader(dataset, batch_size=256, shuffle=False)

            per_model_preds = []
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
                if EMPRES_type == 0:
                    hp["dilation"]  = int(row["dilation"])
                    hp["pool_size"] = int(row["pool_size"])
                trial_obj = DummyTrial(hp)

                model = cfg["model_class"](
                    trial_obj, in_channels=in_channels, **cfg["model_kwargs"]
                ).to(device)
                checkpoint_file = row["checkpoint_file"].strip()
                checkpoint_path = os.path.join(run_dir, checkpoint_file)
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                _, preds = evaluate_model(model, loader, device, criterion)
                per_model_preds.append(preds.squeeze())
                print(f"  Loaded and ran inference for {checkpoint_file}")

            if len(per_model_preds) != top_x:
                raise RuntimeError(
                    f"Expected {top_x} model predictions for fold {fold_num}, "
                    f"got {len(per_model_preds)}."
                )

            pred_stack = np.stack(per_model_preds, axis=0)
            ensemble = pred_stack.mean(axis=0).astype(np.float32)
            fold_ensemble_preds[fold_num] = ensemble
            print(
                f"Fold {fold_num} ensemble predictions: "
                f"n={len(ensemble):,}, "
                f"min={float(ensemble.min()):.4f}, max={float(ensemble.max()):.4f}, "
                f"mean={float(ensemble.mean()):.4f}, std={float(ensemble.std()):.4f}"
            )

        # ------------------------------------------------------------------
        # 7) Build and save the standalone EMPRES output (in array_index order)
        # ------------------------------------------------------------------
        out_df = pd.DataFrame({
            "gene":                       meta["gene"].values,
            "species":                    meta["species"].values,
            "transcript":                 meta["transcript"].values,
            "group_for_cross_validation": meta["group_for_cross_validation"].values,
            "gene_family":                meta["gene_family"].values,
            "median_TPM":                 meta["median_TPM"].values,
        })
        for fold_num, col in zip(range(1, 6), empres_cols):
            out_df[col] = fold_ensemble_preds[fold_num]

        standalone_csv = os.path.join(
            save_dir, f"EMPRES_{EMPRES_type}_pred_new_test_species.csv"
        )
        out_df.to_csv(standalone_csv, index=False)
        print(f"\nSaved standalone EMPRES predictions to: {standalone_csv}")
        print(f"  Shape: {out_df.shape}")

    # ------------------------------------------------------------------
    # 8) Load PhytoExpr predictions and join onto them (lenient mode)
    # ------------------------------------------------------------------
    if not os.path.exists(phyto_pred_file):
        raise FileNotFoundError(
            f"PhytoExpr predictions file not found: {phyto_pred_file}"
        )
    phyto = pd.read_csv(phyto_pred_file)
    print(f"\nLoaded PhytoExpr predictions: shape {phyto.shape}")
    print(f"PhytoExpr columns: {list(phyto.columns)}")

    join_keys = ["gene", "species", "transcript"]
    for k in join_keys:
        if k not in phyto.columns:
            raise ValueError(f"PhytoExpr file is missing required column '{k}'.")
        if k not in out_df.columns:
            raise ValueError(f"EMPRES standalone file is missing required column '{k}'.")

    if len(phyto) != n_samples:
        print(
            f"\nWARNING: PhytoExpr row count ({len(phyto):,}) does not match "
            f"EMPRES/new-species row count ({n_samples:,}). "
            f"Difference: {n_samples - len(phyto):+,} rows. "
            f"Proceeding with lenient left join onto PhytoExpr rows."
        )

    out_df_for_join = out_df[join_keys + empres_cols]
    n_empres_dupes = int(out_df_for_join.duplicated(subset=join_keys).sum())
    if n_empres_dupes > 0:
        print(
            f"WARNING: {n_empres_dupes:,} duplicate (gene, species, transcript) keys "
            f"found in EMPRES predictions."
        )

    joined = phyto.merge(
        out_df_for_join, on=join_keys, how="left", validate="many_to_one"
    )

    if len(joined) != len(phyto):
        print(
            f"WARNING: Joined row count ({len(joined):,}) differs from PhytoExpr row "
            f"count ({len(phyto):,}). This can happen if duplicate join keys exist on "
            f"either side."
        )

    n_unmatched_phyto = int(joined[empres_cols[0]].isna().sum())
    if n_unmatched_phyto > 0:
        print(
            f"WARNING: {n_unmatched_phyto:,} PhytoExpr rows have no matching EMPRES "
            f"prediction on (gene, species, transcript). EMPRES columns will be NaN "
            f"for these rows."
        )

    phyto_keys = phyto[join_keys].drop_duplicates()
    empres_keys = out_df_for_join[join_keys].drop_duplicates()
    n_empres_only = int(
        empres_keys.merge(phyto_keys, on=join_keys, how="left", indicator=True)
        ["_merge"].eq("left_only").sum()
    )
    if n_empres_only > 0:
        print(
            f"WARNING: {n_empres_only:,} EMPRES rows are not present in the PhytoExpr "
            f"file and will not appear in the joined output."
        )

    joined_csv = os.path.join(
        save_dir, f"EMPRES_{EMPRES_type}_pred_new_test_species_with_PhytoExpr.csv"
    )
    joined.to_csv(joined_csv, index=False)
    print(f"Saved joined predictions to: {joined_csv}")
    print(f"  Shape: {joined.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
