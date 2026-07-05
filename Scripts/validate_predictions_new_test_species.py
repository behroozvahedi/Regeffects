#!/usr/bin/env python
"""
validate_predictions_new_test_species.py

Compute MSE and Pearson correlation coefficient between observed log-TPM and
predicted log-TPM for PhytoExpr Model B, PhytoExpr Model C, and EMPRES_{type}
on the entirely held-out new (test) species dataset, using the joined
predictions CSV produced by generate_predictions_new_test_species.py.

For each of the three model families, metrics are computed at two levels:
  - per-fold : one (MSE, Pearson_r) pair per CV fold (five in total).
  - ensemble : mean of the five per-fold predictions per gene, then scored
               once (this parallels the across-fold ensembling logic used in
               aggregate_predictions.py for the training species).

Both levels are further computed at two scopes:
  - overall     : across all genes of all held-out test species combined.
  - per species : separately for each held-out test species.

All predicted values in the joined file, and the log10(1 + median_TPM)
transformation applied here to the observed values, are in log10(1 + TPM)
space. The observed 'median_TPM' column in the joined file is in linear TPM
scale and is log-transformed by this script into a new 'log_median_TPM' column.

CLI mirrors the other scripts in this repository:
    --save_dir     Directory containing the joined predictions CSV; also where
                   the output metrics CSV is written.
    --EMPRES_type  EMPRES model type in {0, 1, 2, 3, 4}. Determines the
                   EMPRES column prefix and default file names.
    --joined_file  Full path to the joined predictions CSV.
                   Default: {save_dir}/EMPRES_{type}_pred_new_test_species_with_PhytoExpr.csv
    --out_file     Full path to the output metrics CSV.
                   Default: {save_dir}/validation_test_species_EMPRES_{type}.csv
"""

import os
import argparse

import numpy as np
import pandas as pd


def score(pred, obs):
    """
    Compute (n_valid, MSE, Pearson_r) for a predicted vector against an
    observed vector, dropping any position where either is NaN or infinite.
    Returns (n, NaN, NaN) if fewer than 2 valid pairs remain (Pearson_r is
    undefined for n < 2).
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs  = np.asarray(obs,  dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(obs)
    n = int(mask.sum())
    if n < 2:
        return n, float("nan"), float("nan")
    p = pred[mask]
    o = obs[mask]
    mse = float(np.mean((p - o) ** 2))
    r = float(np.corrcoef(p, o)[0, 1])
    return n, mse, r


def main():
    # ------------------------------------------------------------------
    # 1) Parse CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Validate test-species predictions from PhytoExpr Model B, "
            "PhytoExpr Model C, and EMPRES_{type} against the observed "
            "log10(1 + median_TPM) values."
        )
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help=(
            "Directory containing the joined predictions CSV and where the "
            "output metrics CSV is saved."
        ),
    )
    parser.add_argument(
        "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
        help=(
            "EMPRES model type (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)."
        ),
    )
    parser.add_argument(
        "--joined_file", type=str, default=None,
        help=(
            "Path to the joined predictions CSV. Defaults to "
            "{save_dir}/EMPRES_{type}_pred_new_test_species_with_PhytoExpr.csv."
        ),
    )
    parser.add_argument(
        "--out_file", type=str, default=None,
        help=(
            "Path for the output metrics CSV. Defaults to "
            "{save_dir}/validation_test_species_EMPRES_{type}.csv."
        ),
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    EMPRES_type = args.EMPRES_type
    joined_file = (
        args.joined_file if args.joined_file is not None
        else os.path.join(
            save_dir,
            f"EMPRES_{EMPRES_type}_pred_new_test_species_with_PhytoExpr.csv",
        )
    )
    out_file = (
        args.out_file if args.out_file is not None
        else os.path.join(
            save_dir, f"validation_test_species_EMPRES_{EMPRES_type}.csv"
        )
    )

    print("Configuration:")
    print(f"  SAVE_DIR    = {save_dir}")
    print(f"  EMPRES_TYPE = {EMPRES_type}")
    print(f"  JOINED_FILE = {joined_file}")
    print(f"  OUT_FILE    = {out_file}")

    # ------------------------------------------------------------------
    # 2) Load the joined predictions CSV
    # ------------------------------------------------------------------
    if not os.path.exists(joined_file):
        raise FileNotFoundError(f"Joined predictions file not found: {joined_file}")

    df = pd.read_csv(joined_file)
    print(f"\nLoaded joined predictions: shape {df.shape}")

    # ------------------------------------------------------------------
    # 3) Log-transform the observed median_TPM (linear -> log10(1+x))
    # ------------------------------------------------------------------
    if "median_TPM" not in df.columns:
        raise ValueError("Input file is missing 'median_TPM' column.")
    df["log_median_TPM"] = np.log10(1.0 + df["median_TPM"].astype(np.float64))

    # ------------------------------------------------------------------
    # 4) Define the three model families and their per-fold column prefixes
    # ------------------------------------------------------------------
    model_families = [
        ("PhytoExpr_ModelB",       "ModelB_pred_median_TPM"),
        ("PhytoExpr_ModelC",       "ModelC_pred_median_TPM"),
        (f"EMPRES_{EMPRES_type}",  f"EMPRES_{EMPRES_type}_pred_median_TPM"),
    ]
    fold_labels = ["1", "2", "3", "4", "5"]

    # Sanity check that all expected per-fold columns are present.
    for family_name, col_prefix in model_families:
        for f in fold_labels:
            col = f"{col_prefix}{f}"
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' not found in joined file.")

    # ------------------------------------------------------------------
    # 5) Compute the across-fold ensemble column per family
    #    (mean of the 5 per-fold predictions per gene, NaN if any fold NaN)
    # ------------------------------------------------------------------
    ensemble_col_of = {}
    for family_name, col_prefix in model_families:
        cols = [f"{col_prefix}{f}" for f in fold_labels]
        ens_col = f"{col_prefix}_ensemble"
        df[ens_col] = df[cols].mean(axis=1, skipna=False)
        ensemble_col_of[family_name] = ens_col

    # ------------------------------------------------------------------
    # 6) Iterate over (scope, model family, fold-or-ensemble) and score
    # ------------------------------------------------------------------
    species_list = sorted(df["species"].unique())
    print(f"\nFound {len(species_list)} test species: {species_list}")

    records = []
    scopes = [("ALL", df)] + [(sp, df[df["species"] == sp]) for sp in species_list]

    for scope_name, scope_df in scopes:
        scope_obs = scope_df["log_median_TPM"].values
        scope_kind = "overall" if scope_name == "ALL" else "species"

        for family_name, col_prefix in model_families:
            for f in fold_labels:
                col = f"{col_prefix}{f}"
                n, mse, r = score(scope_df[col].values, scope_obs)
                records.append({
                    "scope":     scope_kind,
                    "species":   scope_name,
                    "model":     family_name,
                    "fold":      f,
                    "n_samples": n,
                    "MSE":       mse,
                    "Pearson_r": r,
                })
            ens_col = ensemble_col_of[family_name]
            n, mse, r = score(scope_df[ens_col].values, scope_obs)
            records.append({
                "scope":     scope_kind,
                "species":   scope_name,
                "model":     family_name,
                "fold":      "ensemble",
                "n_samples": n,
                "MSE":       mse,
                "Pearson_r": r,
            })

    results = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 7) Print overall summary (all species combined) - full 5 folds + ensemble
    # ------------------------------------------------------------------
    print()
    print("=" * 79)
    print(f"OVERALL (across all {len(species_list)} test species combined)")
    print("=" * 79)
    overall = results[results["scope"] == "overall"]
    for family_name, _ in model_families:
        sub = overall[overall["model"] == family_name]
        print(f"\n{family_name}:")
        print(f"  {'fold':<10} {'n':>10} {'MSE':>12} {'Pearson_r':>12}")
        for _, row in sub.iterrows():
            print(
                f"  {str(row['fold']):<10} "
                f"{row['n_samples']:>10,} "
                f"{row['MSE']:>12.6f} "
                f"{row['Pearson_r']:>12.6f}"
            )

    # ------------------------------------------------------------------
    # 8) Print per-species summary (ensemble only, per model)
    #    (Full per-fold per-species metrics are still in the saved CSV.)
    # ------------------------------------------------------------------
    print()
    print("=" * 79)
    print("PER-SPECIES (ensemble = mean of the 5 per-fold predictions)")
    print("=" * 79)
    per_species_ens = results[
        (results["scope"] == "species") & (results["fold"] == "ensemble")
    ]
    print(f"\n{'species':<28} {'model':<22} {'n':>10} {'MSE':>12} {'Pearson_r':>12}")
    print("-" * 88)
    for sp in species_list:
        for family_name, _ in model_families:
            row = per_species_ens[
                (per_species_ens["species"] == sp)
                & (per_species_ens["model"] == family_name)
            ].iloc[0]
            print(
                f"{sp:<28} "
                f"{family_name:<22} "
                f"{row['n_samples']:>10,} "
                f"{row['MSE']:>12.6f} "
                f"{row['Pearson_r']:>12.6f}"
            )
        print("-" * 88)

    # ------------------------------------------------------------------
    # 9) Save the full long-format metrics table
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    results.to_csv(out_file, index=False)
    print(f"\nSaved validation metrics to: {out_file}")
    print(f"  Shape: {results.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
