#!/usr/bin/env python
"""
aggregate_predictions.py

Concatenates the five per-fold predictions.csv files produced by
generate_predictions.py for one or more EMPRES types, saves the combined CSV
per type to disk, and reports the aggregate ensemble MSE and the Pearson
correlation between ensemble_pred and observed_TPM across all folds, per fold
and (with --npy_dir) per species.

With more than one EMPRES type, or with the PhytoExpr transformer B/C prediction
files, it also compares the models on the same genes, overall and per species,
and writes all metrics to one TSV file.

Both ensemble_pred and observed_TPM in each per-fold predictions.csv are
already stored in the log10(1 + TPM) space, so no further transformation
is applied to them. The PhytoExpr predictions are used as given, unless
--phyto_pred_transform log10_1p is set.

CLI mirrors test_PC_a2z.py, generate_predictions.py, and model_info_df.py:
    --out_dir       Base output directory containing
                    val{V}_test{T}/{subdir}/predictions.csv files.
    --save_dir      (Optional) Directory in which to save the concatenated CSVs
                    and the metrics TSV. Defaults to --out_dir if not provided.
    --EMPRES_type   One or more EMPRES model types in {0, 1, 2, 3, 4}.
    --npy_dir       (Optional) The training npy_files directory (gene.npy,
                    species.npy, group_for_cross_validation.npy). Needed for the
                    per-species metrics and for the PhytoExpr comparison.
    --use_saved     (Optional) Read <save_dir>/predictions_all_folds_EMPRES_{type}.csv
                    instead of the five per-fold files (nothing is rewritten).
    --phyto_B_file, --phyto_C_file
                    (Optional) PhytoExpr transformer B / C prediction CSVs. If given,
                    each is merged with the EMPRES predictions on (gene, species) and
                    all models are compared on the genes they share.
    --phyto_gene_col, --phyto_species_col, --phyto_pred_col
                    Column names in the PhytoExpr files (defaults: gene, species,
                    and auto-detection of a single prediction column).
    --phyto_pred_transform
                    none (default; predictions already in log10(1 + TPM)) or
                    log10_1p (predictions in linear TPM).
    --metrics_file  (Optional) Path of the metrics TSV. Default:
                    <save_dir>/cv_metrics_EMPRES_<types>[_with_PhytoExpr].tsv
"""
import os
import argparse

import numpy as np
import pandas as pd

from utils_PC_a2z import EMPRES_CONFIG


# The five cross-validation folds (val_group, test_group), matching
# the FOLDS array used in the all-folds bash wrapper scripts.
FOLDS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]

METRIC_COLS = ["gene_set", "scope", "species", "fold", "model", "n_samples", "MSE", "Pearson_r"]


def score(pred, obs):
    """(n_valid, MSE, Pearson_r) over positions where both values are finite."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(obs)
    n = int(mask.sum())
    if n < 2:
        return n, float("nan"), float("nan")
    p, o = pred[mask], obs[mask]
    return n, float(np.mean((p - o) ** 2)), float(np.corrcoef(p, o)[0, 1])


def load_type(out_dir, save_dir, EMPRES_type, use_saved):
    """Return the concatenated all-folds predictions for one EMPRES type."""
    cfg = EMPRES_CONFIG[EMPRES_type]
    subdir = cfg["subdir"]

    if use_saved:
        saved_csv = os.path.join(save_dir, f"predictions_all_folds_EMPRES_{EMPRES_type}.csv")
        if not os.path.exists(saved_csv):
            raise FileNotFoundError(f"--use_saved given but file not found: {saved_csv}")
        df_all = pd.read_csv(saved_csv, dtype={"gene": str})
        print(f"Loaded saved all-folds predictions for EMPRES_type={EMPRES_type}: "
              f"{len(df_all):,} rows  ({saved_csv})")
        print(f"\nConcatenated all 5 folds: total {len(df_all):,} rows")
        return df_all, subdir

    # ------------------------------------------------------------------
    # Load all 5 per-fold predictions.csv files
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
        df_fold = pd.read_csv(pred_file, dtype={"gene": str})
        print(f"  fold val={val_group}, test={test_group}: {len(df_fold):>8,} rows  ({pred_file})")
        dfs.append(df_fold)

    # ------------------------------------------------------------------
    # Concatenate along rows and save the combined CSV
    # ------------------------------------------------------------------
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"\nConcatenated all 5 folds: total {len(df_all):,} rows")

    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"predictions_all_folds_EMPRES_{EMPRES_type}.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Saved concatenated predictions to: {out_csv}")
    return df_all, subdir


def attach_species(df_all, npy):
    """
    Add a 'species' column. The rows of the concatenated table are the test rows
    of the five folds in FOLDS order, each in npy row order (generate_predictions.py
    uses np.where(groups == test_group)), so species is taken from species.npy at
    those positions after checking that gene.npy gives the same gene at every row.
    Falls back to a gene -> species lookup if the order differs and gene IDs are unique.
    """
    gene_npy, species_npy, groups_npy = npy
    idx_all = np.concatenate([np.where(groups_npy == str(t))[0] for _, t in FOLDS])
    genes = df_all["gene"].astype(str).to_numpy()
    if len(idx_all) == len(df_all) and np.array_equal(gene_npy[idx_all], genes):
        df_all["species"] = species_npy[idx_all]
        print("  species: taken from species.npy by fold/row order (gene IDs checked at every row)")
        return df_all
    lookup = pd.Series(species_npy, index=gene_npy)
    if lookup.index.is_unique and set(genes) <= set(lookup.index):
        df_all["species"] = lookup.loc[genes].to_numpy()
        print("  species: taken from species.npy by gene ID (row order differs from npy_files)")
        return df_all
    raise ValueError(
        "Cannot assign species: the predictions do not follow the npy_files row order and "
        "gene IDs are not unique (or not all present) in gene.npy."
    )


def report_type(df_all, EMPRES_type, subdir, species_order, records):
    """Print the per-type metrics (as before, plus per species) and record them."""
    model = f"EMPRES_{EMPRES_type}"

    # Light sanity check: every gene in the 5-fold partition should appear exactly once.
    keys = ["gene", "species"] if "species" in df_all.columns else ["gene"]
    n_unique = len(df_all.drop_duplicates(subset=keys))
    if n_unique != len(df_all):
        print(
            f"WARNING: concatenated table has {len(df_all):,} rows but only "
            f"{n_unique:,} unique {'/'.join(keys)} values. Expected one row per gene "
            f"if folds are non-overlapping."
        )

    # ------------------------------------------------------------------
    # Aggregate metrics across all folds (in log10(1 + TPM) space)
    # ------------------------------------------------------------------
    obs = df_all["observed_TPM"].to_numpy()
    ens = df_all["ensemble_pred"].to_numpy()

    mse_total = float(np.mean((ens - obs) ** 2))
    pearson_total = float(np.corrcoef(ens, obs)[0, 1])
    records.append(dict(gene_set=f"{model}_all", scope="overall", species="ALL", fold="all",
                        model=model, n_samples=len(df_all), MSE=mse_total, Pearson_r=pearson_total))

    print()
    print("=" * 79)
    print(f"Cross-validation aggregate metrics for EMPRES_type={EMPRES_type} (subdir='{subdir}')")
    print("=" * 79)
    print(f"Total samples across all folds:            {len(df_all):,}")
    print(f"Ensemble MSE (log10(1+TPM) space):         {mse_total:.6f}")
    print(f"Pearson r (ensemble vs observed):          {pearson_total:.6f}")
    print("=" * 79)

    # ------------------------------------------------------------------
    # Per-fold breakdown (sanity check vs test_PC_a2z.py)
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
        records.append(dict(gene_set=f"{model}_all", scope="fold", species="ALL", fold=str(g),
                            model=model, n_samples=len(sub), MSE=sub_mse, Pearson_r=sub_r))

    # ------------------------------------------------------------------
    # Per-species breakdown
    # ------------------------------------------------------------------
    if "species" in df_all.columns:
        print()
        print("Per-species breakdown (all folds combined):")
        print(f"{'species':<14} {'n_samples':>10} {'MSE':>14} {'Pearson r':>14}")
        print("-" * 54)
        for sp in species_order:
            sub = df_all[df_all["species"] == sp]
            if len(sub) == 0:
                continue
            n, mse, r = score(sub["ensemble_pred"], sub["observed_TPM"])
            print(f"{sp:<14} {n:>10,} {mse:>14.6f} {r:>14.6f}")
            records.append(dict(gene_set=f"{model}_all", scope="species", species=sp, fold="all",
                                model=model, n_samples=n, MSE=mse, Pearson_r=r))


def load_phyto(path, label, args, species_set):
    """Read one PhytoExpr prediction file -> DataFrame(gene, species, <label>)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PhytoExpr file not found: {path}")
    ph = pd.read_csv(path, dtype={args.phyto_gene_col: str, args.phyto_species_col: str})
    print(f"\nLoaded {label}: {len(ph):,} rows ({path})")
    print(f"  columns: {list(ph.columns)}")
    for c in (args.phyto_gene_col, args.phyto_species_col):
        if c not in ph.columns:
            raise ValueError(f"{label}: column '{c}' not found. Set --phyto_gene_col / --phyto_species_col.")
    pred_col = args.phyto_pred_col
    if pred_col is None:
        cands = [c for c in ph.columns if "pred" in c.lower()]
        if len(cands) != 1:
            raise ValueError(f"{label}: cannot pick the prediction column automatically "
                             f"(candidates: {cands}). Set --phyto_pred_col.")
        pred_col = cands[0]
    if pred_col not in ph.columns:
        raise ValueError(f"{label}: prediction column '{pred_col}' not found.")
    print(f"  using gene='{args.phyto_gene_col}', species='{args.phyto_species_col}', prediction='{pred_col}'")

    out = pd.DataFrame({
        "gene": ph[args.phyto_gene_col].astype(str).to_numpy(),
        "species": ph[args.phyto_species_col].astype(str).to_numpy(),
        label: pd.to_numeric(ph[pred_col], errors="coerce").to_numpy(dtype=np.float64),
    })
    if args.phyto_pred_transform == "log10_1p":
        out[label] = np.log10(1.0 + out[label])
    unknown = sorted(set(out["species"]) - set(species_set))
    if unknown:
        raise ValueError(f"{label}: species not found in the EMPRES data: {unknown[:20]} "
                         f"(EMPRES species: {sorted(species_set)})")
    n_dup = int(out.duplicated(subset=["gene", "species"]).sum())
    if n_dup:
        raise ValueError(f"{label}: {n_dup:,} duplicated (gene, species) rows.")
    n_nan = int(out[label].isna().sum())
    v = out[label].to_numpy()
    print(f"  predictions: min {np.nanmin(v):.3f}, median {np.nanmedian(v):.3f}, max {np.nanmax(v):.3f}; "
          f"missing {n_nan:,}")
    return out


def compare(wide, models, gene_set, species_order, records):
    """Score every model on the rows of `wide`, overall and per species, and print a table."""
    print()
    print("=" * 79)
    print(f"Model comparison on gene set '{gene_set}': {len(wide):,} genes, "
          f"{wide['species'].nunique()} species (same genes for every model)")
    print("=" * 79)
    print(f"\n{'model':<20} {'n':>10} {'MSE':>12} {'Pearson_r':>12}")
    print("-" * 57)
    for m in models:
        n, mse, r = score(wide[m], wide["observed_TPM"])
        print(f"{m:<20} {n:>10,} {mse:>12.6f} {r:>12.6f}")
        records.append(dict(gene_set=gene_set, scope="overall", species="ALL", fold="all",
                            model=m, n_samples=n, MSE=mse, Pearson_r=r))
    print(f"\nPER-SPECIES\n{'species':<14} {'model':<20} {'n':>10} {'MSE':>12} {'Pearson_r':>12}")
    print("-" * 72)
    for sp in species_order:
        sub = wide[wide["species"] == sp]
        if len(sub) == 0:
            continue
        for m in models:
            n, mse, r = score(sub[m], sub["observed_TPM"])
            print(f"{sp:<14} {m:<20} {n:>10,} {mse:>12.6f} {r:>12.6f}")
            records.append(dict(gene_set=gene_set, scope="species", species=sp, fold="all",
                                model=m, n_samples=n, MSE=mse, Pearson_r=r))
        print("-" * 72)


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
        help="Directory in which to save the concatenated CSVs and metrics TSV. Defaults to --out_dir."
    )
    parser.add_argument(
        "--EMPRES_type", type=int, nargs="+", required=True, choices=[0, 1, 2, 3, 4],
        help="One or more EMPRES model types (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)"
    )
    parser.add_argument("--npy_dir", type=str, default=None,
                        help="Training npy_files directory (gene.npy, species.npy, "
                             "group_for_cross_validation.npy); enables per-species metrics.")
    parser.add_argument("--use_saved", action="store_true",
                        help="Read <save_dir>/predictions_all_folds_EMPRES_{type}.csv instead of the per-fold files.")
    parser.add_argument("--phyto_B_file", type=str, default=None, help="PhytoExpr transformer B predictions CSV.")
    parser.add_argument("--phyto_C_file", type=str, default=None, help="PhytoExpr transformer C predictions CSV.")
    parser.add_argument("--phyto_gene_col", type=str, default="gene")
    parser.add_argument("--phyto_species_col", type=str, default="species")
    parser.add_argument("--phyto_pred_col", type=str, default=None,
                        help="Prediction column in the PhytoExpr files (default: the single column containing 'pred').")
    parser.add_argument("--phyto_pred_transform", choices=["none", "log10_1p"], default="none",
                        help="none: PhytoExpr predictions already in log10(1+TPM); log10_1p: they are linear TPM.")
    parser.add_argument("--metrics_file", type=str, default=None, help="Path of the metrics TSV.")
    args = parser.parse_args()

    out_dir = args.out_dir
    save_dir = args.save_dir if args.save_dir is not None else out_dir
    types = list(dict.fromkeys(args.EMPRES_type))          # keep order, drop repeats
    phyto_files = [(lab, p) for lab, p in (("PhytoExpr_B", args.phyto_B_file),
                                           ("PhytoExpr_C", args.phyto_C_file)) if p]
    if phyto_files and args.npy_dir is None:
        parser.error("--phyto_B_file / --phyto_C_file need --npy_dir (species is required for the merge).")
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = args.metrics_file or os.path.join(
        save_dir, f"cv_metrics_EMPRES_{'_'.join(map(str, types))}"
                  f"{'_with_PhytoExpr' if phyto_files else ''}.tsv")

    npy, species_order = None, []
    if args.npy_dir:
        gene_npy = np.load(os.path.join(args.npy_dir, "gene.npy"), allow_pickle=True).astype(str)
        species_npy = np.load(os.path.join(args.npy_dir, "species.npy"), allow_pickle=True).astype(str)
        groups_npy = np.load(os.path.join(args.npy_dir, "group_for_cross_validation.npy"),
                             allow_pickle=True).astype(str)
        if not (len(gene_npy) == len(species_npy) == len(groups_npy)):
            raise ValueError("gene.npy, species.npy and group_for_cross_validation.npy differ in length.")
        npy = (gene_npy, species_npy, groups_npy)
        species_order = list(dict.fromkeys(species_npy))   # order of the npy_files (species blocks)

    # ------------------------------------------------------------------
    # 2) Per EMPRES type: concatenate, save, report
    # ------------------------------------------------------------------
    records, per_type = [], {}
    for t in types:
        if len(types) > 1:
            print()
            print("#" * 79)
            print(f"# EMPRES_type={t}")
            print("#" * 79)
        df_all, subdir = load_type(out_dir, save_dir, t, args.use_saved)
        if npy is not None:
            df_all = attach_species(df_all, npy)
        report_type(df_all, t, subdir, species_order, records)
        per_type[t] = df_all

    # ------------------------------------------------------------------
    # 3) Model comparison on shared genes (EMPRES types, optionally PhytoExpr B/C)
    # ------------------------------------------------------------------
    if npy is not None and (len(types) > 1 or phyto_files):
        keys = ["gene", "species"]
        wide = None
        for t in types:
            d = per_type[t][keys + ["observed_TPM", "ensemble_pred"]].rename(
                columns={"ensemble_pred": f"EMPRES_{t}", "observed_TPM": f"obs_{t}"})
            wide = d if wide is None else wide.merge(d, on=keys, how="inner", validate="one_to_one")
        obs_cols = [f"obs_{t}" for t in types]
        max_obs_diff = float(np.nanmax(np.abs(wide[obs_cols].to_numpy() - wide[[obs_cols[0]]].to_numpy())))
        if max_obs_diff > 1e-6:
            raise ValueError(f"observed_TPM differs between EMPRES types (max |diff| {max_obs_diff:.3g}).")
        wide = wide.rename(columns={obs_cols[0]: "observed_TPM"}).drop(columns=obs_cols[1:])
        empres_models = [f"EMPRES_{t}" for t in types]
        print(f"\nGenes with predictions from all {len(types)} EMPRES type(s): {len(wide):,}")

        if len(types) > 1:
            compare(wide, empres_models, "EMPRES_all", species_order, records)

        if phyto_files:
            n_before = len(wide)
            for lab, path in phyto_files:
                ph = load_phyto(path, lab, args, set(species_order))
                m = wide.merge(ph, on=keys, how="outer", indicator=True)
                n_both = int((m["_merge"] == "both").sum())
                print(f"  {lab}: {n_both:,} genes matched to EMPRES; "
                      f"{int((m['_merge'] == 'left_only').sum()):,} EMPRES genes without a {lab} prediction; "
                      f"{int((m['_merge'] == 'right_only').sum()):,} {lab} rows without an EMPRES prediction")
                wide = wide.merge(ph, on=keys, how="inner", validate="one_to_one")
            wide = wide[np.isfinite(wide[[lab for lab, _ in phyto_files]]).all(axis=1)]
            print(f"\nGenes shared by all models: {len(wide):,} of {n_before:,}")
            compare(wide, [lab for lab, _ in phyto_files] + empres_models,
                    "shared_with_PhytoExpr", species_order, records)

    # ------------------------------------------------------------------
    # 4) Save all metrics
    # ------------------------------------------------------------------
    pd.DataFrame(records, columns=METRIC_COLS).to_csv(metrics_file, sep="\t", index=False, float_format="%.6f")
    print(f"\nSaved metrics ({len(records):,} rows) to: {metrics_file}")


if __name__ == "__main__":
    main()
