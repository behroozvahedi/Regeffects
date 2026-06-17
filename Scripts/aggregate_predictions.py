#!/usr/bin/env python
"""
aggregate_predictions.py

Concatenates the five per-fold predictions.csv files produced by
generate_predictions.py for a given EMPRES_type, saves the combined CSV
to disk, and reports the aggregate ensemble MSE and the Pearson correlation
between ensemble_pred and observed_TPM across all folds.

Both ensemble_pred and observed_TPM in each per-fold predictions.csv are
already stored in the log10(1 + TPM) space, so no further transformation
is applied here.

CLI mirrors test_PC_a2z.py, generate_predictions.py, and model_info_df.py:
    --out_dir       Base output directory containing
                    val{V}_test{T}/{subdir}/predictions.csv files.
    --save_dir      (Optional) Directory in which to save the concatenated CSV.
                    Defaults to --out_dir if not provided.
    --EMPRES_type   EMPRES model type in {0, 1, 2, 3, 4}.
"""
import os
import argparse

import numpy as np
import pandas as pd

from utils_PC_a2z import EMPRES_CONFIG


def main():
    # ------------------------------------------------------------------
    # 1) Parse CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Concatenate per-fold predictions.csv files and report aggregate ensemble metrics."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Base output directory containing val{V}_test{T}/{subdir}/predictions.csv files."
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory in which to save the concatenated CSV. Defaults to --out_dir if not provided."
    )
    parser.add_argument(
        "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
        help="EMPRES model type (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    save_dir = args.save_dir if args.save_dir is not None else out_dir
    EMPRES_type = args.EMPRES_type
    cfg = EMPRES_CONFIG[EMPRES_type]
    subdir = cfg["subdir"]

    # The five cross-validation folds (val_group, test_group), matching
    # the FOLDS array used in the all-folds bash wrapper scripts.
    FOLDS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]

    # ------------------------------------------------------------------
    # 2) Load all 5 per-fold predictions.csv files
    # ------------------------------------------------------------------
    print(f"Loading per-fold predictions for EMPRES_type={EMPRES_type} (subdir='{subdir}'):")
    dfs = []
    for val_group, test_group in FOLDS:
        pred_file = os.path.join(
            out_dir,
            f"val{val_group}_test{test_group}",
            subdir,
            "predictions.csv",
        )
        if not os.path.exists(pred_file):
            raise FileNotFoundError(
                f"predictions.csv not found for fold val={val_group}, test={test_group}:\n  {pred_file}"
            )
        df_fold = pd.read_csv(pred_file)
        print(f"  fold val={val_group}, test={test_group}: {len(df_fold):>8,} rows  ({pred_file})")
        dfs.append(df_fold)

    # ------------------------------------------------------------------
    # 3) Concatenate along rows and save the combined CSV
    # ------------------------------------------------------------------
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"\nConcatenated all 5 folds: total {len(df_all):,} rows")

    # Light sanity check: every gene in the 5-fold partition should appear exactly once.
    n_unique_genes = df_all["gene"].nunique()
    if n_unique_genes != len(df_all):
        print(
            f"WARNING: concatenated table has {len(df_all):,} rows but only "
            f"{n_unique_genes:,} unique gene values. Expected one row per gene "
            f"if folds are non-overlapping."
        )

    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"predictions_all_folds_EMPRES_{EMPRES_type}.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Saved concatenated predictions to: {out_csv}")

    # ------------------------------------------------------------------
    # 4) Aggregate metrics across all folds (in log10(1 + TPM) space)
    # ------------------------------------------------------------------
    obs = df_all["observed_TPM"].to_numpy()
    ens = df_all["ensemble_pred"].to_numpy()

    mse_total = float(np.mean((ens - obs) ** 2))
    pearson_total = float(np.corrcoef(ens, obs)[0, 1])

    print()
    print("=" * 79)
    print(f"Cross-validation aggregate metrics for EMPRES_type={EMPRES_type} (subdir='{subdir}')")
    print("=" * 79)
    print(f"Total samples across all folds:            {len(df_all):,}")
    print(f"Ensemble MSE (log10(1+TPM) space):         {mse_total:.6f}")
    print(f"Pearson r (ensemble vs observed):          {pearson_total:.6f}")
    print("=" * 79)

    # ------------------------------------------------------------------
    # 5) Per-fold breakdown (sanity check vs test_PC_a2z.py)
    #    The per-fold ensemble MSE printed here should match the
    #    "Ensemble test loss" line in the corresponding test_PC_a2z.py
    #    run for the same EMPRES_type and fold.
    # ------------------------------------------------------------------
    print()
    print("Per-fold breakdown (group = test_group; ensemble MSE should match test_PC_a2z.py output):")
    print(f"{'group (test)':<14} {'n_samples':>10} {'MSE':>14} {'Pearson r':>14}")
    print("-" * 54)
    for g in sorted(df_all["group"].unique()):
        sub = df_all[df_all["group"] == g]
        sub_obs = sub["observed_TPM"].to_numpy()
        sub_ens = sub["ensemble_pred"].to_numpy()
        sub_mse = float(np.mean((sub_ens - sub_obs) ** 2))
        sub_r = float(np.corrcoef(sub_ens, sub_obs)[0, 1])
        print(f"{str(g):<14} {len(sub):>10,} {sub_mse:>14.6f} {sub_r:>14.6f}")


if __name__ == "__main__":
    main()
