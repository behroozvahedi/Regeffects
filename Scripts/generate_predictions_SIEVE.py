#!/usr/bin/env python
"""
generate_predictions_SIEVE.py

Generate EMPRES predictions for the SIEVE in-planta validation dataset
(Brachypodium distachyon population from `data.bd.csv`) and write the
output in the exact tab-separated format used by Camous's
`Data_processing/generate_predictions_hd5.py`, so that Guillaume's
downstream R script `1_Validating_predictions.R` can consume it without
modification.

Pipeline
--------
    1. Load SIEVE OHE .npy files and `sieve_metadata.csv`
       (produced by `SIEVE_OHE_seqs_efficient.py`).
    2. Build `test_groups[gene] = "1".."5"` by loading `data.csv` and
       `gene.id.translation.tsv` exactly the way Camous's
       `GetTranslations` and `GetGroups` functions do it.
    3. For each SIEVE row, look up its CV group via `meta["gene"]`
       (which is Camous's BD21.3 short gene ID, e.g. "1G0000200").
       Rows whose gene is not in `test_groups` are skipped, matching
       Camous's "skipping gene" behaviour.
    4. For each of the 5 CV folds (val, test) in
       [(1,2), (2,3), (3,4), (4,5), (5,1)]:
           - Bucket the SIEVE rows whose CV group == test_group.
           - Load the fold's top-X EMPRES model checkpoints.
           - Run batched inference (via DNADualDataset + DataLoader) and
             record each model's per-row prediction.
    5. Explode the resulting per-sequence predictions by the
       space-separated cohort in `group_for_cross_validation`, producing
       one output row per (plant_line_id, gene, transcript) triple.
    6. Save to `predictions_EMPRES_{type}.tsv` with Camous's exact header
       (including his `model_2_pred2` typo, preserved intentionally for
       byte-level parity with the EMPRES 1-4 files).

CLI (mirrors generate_predictions_new_test_species.py)
-------------------------------------------------------
    --data_dir           Directory holding tss_OHE.npy, tts_OHE.npy, and
                         sieve_metadata.csv (Step-1 outputs). Also the
                         default location for data.csv and
                         gene.id.translation.tsv (overridable).
    --out_dir            Base directory of the trained EMPRES models:
                             {out_dir}/val{V}_test{T}/{subdir}/trial_results.csv
                             {out_dir}/val{V}_test{T}/{subdir}/<checkpoint>.pth
                         Same layout used by test_PC_a2z.py and
                         generate_predictions.py. Read only.
    --save_dir           Directory into which predictions_EMPRES_{type}.tsv
                         is written.
    --phyto_data_csv     PhytoExpr training data CSV with an integer
                         'group_for_cross_validation' column (Camous's
                         `data.csv`). Default: {data_dir}/data.csv.
    --translation_file   BD21 <-> BD21.3 gene-ID translation TSV
                         (Camous's `gene.id.translation.tsv`).
                         Default: {data_dir}/gene.id.translation.tsv.
    --global_stats_dir   Directory of global_stats_train_*.npz files
                         (used only when cfg['standardize'] is True,
                         i.e. EMPRES 1-4). Default: --data_dir.
    --top_x              Number of top models to ensemble per fold
                         (default 5; matches Camous's TOP_X).
    --EMPRES_type        0..4. Selects the row of EMPRES_CONFIG.
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


# ============================================================
# Camous's `GetTranslations` / `GetGroups` helpers
# (kept byte-for-byte functionally identical to
#  Data_processing/generate_predictions_hd5.py)
# ============================================================
def GetTranslations(file_path):
    """
    Read gene.id.translation.tsv.
    Returns a dict {BD21_name -> BD21.3_short_name}, e.g.
        translation["Bradi1g00200"] == "1G0000200"
    (i.e. items[0].split('.')[1] gives the BD21.3 short form)
    """
    translation = dict()
    with open(file_path, "r") as f:
        for line in f:
            items = line.strip("\n").split("\t")
            translation[items[1]] = items[0].split(".")[1]
    return translation


def GetGroups(file_path, translation):
    """
    Read PhytoExpr's data.csv line-by-line and build a dict
    {gene_id -> integer_CV_group_as_string}.

    For BD genes, `data.csv`'s gene column is in BD21 form
    ("Bradi1g00200") and the translation table maps it to BD21.3 short
    ("1G0000200"), so the stored key is the BD21.3 short form -- which
    matches SIEVE metadata's `gene` column verbatim.
    """
    groups = dict()
    with open(file_path, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            items = line.strip("\n").split(",")
            gene_id = items[1].strip('"')
            if gene_id in translation:
                gene_id = translation[gene_id]
            groups[gene_id] = items[8]
    return groups


# ============================================================
# Main
# ============================================================
def main():
    # ------------------------------------------------------------------
    # 1) Parse CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Generate SIEVE (Brachypodium distachyon in-planta validation) "
            "predictions using EMPRES top-X per-fold models, and write "
            "predictions_EMPRES_{type}.tsv in Camous's exact TSV format."
        )
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help=(
            "Directory containing SIEVE input files (tss_OHE.npy, tts_OHE.npy, "
            "sieve_metadata.csv, and by default data.csv and gene.id.translation.tsv)."
        ),
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help=(
            "Base output directory of the trained EMPRES models, containing "
            "val{V}_test{T}/{subdir}/trial_results.csv and the corresponding "
            ".pth checkpoints. Read only; nothing is written here."
        ),
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory into which predictions_EMPRES_{type}.tsv is written.",
    )
    parser.add_argument(
        "--phyto_data_csv", type=str, default=None,
        help=(
            "Full path to PhytoExpr's data.csv containing the integer CV group "
            "column. Default: {data_dir}/data.csv."
        ),
    )
    parser.add_argument(
        "--translation_file", type=str, default=None,
        help=(
            "Full path to gene.id.translation.tsv mapping BD21 <-> BD21.3 "
            "gene IDs. Default: {data_dir}/gene.id.translation.tsv."
        ),
    )
    parser.add_argument(
        "--global_stats_dir", type=str, default=None,
        help=(
            "Directory containing global_stats_train_*.npz files for the "
            "per-fold standardization. Only used when cfg['standardize'] "
            "is True (EMPRES 1-4). Defaults to --data_dir."
        ),
    )
    parser.add_argument(
        "--top_x", type=int, default=5,
        help="Number of top models to ensemble per fold (default 5).",
    )
    parser.add_argument(
        "--EMPRES_type", type=int, required=True, choices=[0, 1, 2, 3, 4],
        help=(
            "EMPRES model type (0=OHE, 1=PC, 2=PC+a2z_pred, 3=PC+a2z_emb, 4=a2z_emb)."
        ),
    )
    args = parser.parse_args()

    data_dir         = args.data_dir
    out_dir          = args.out_dir
    save_dir         = args.save_dir
    phyto_data_csv   = (args.phyto_data_csv
                        if args.phyto_data_csv is not None
                        else os.path.join(data_dir, "data.csv"))
    translation_file = (args.translation_file
                        if args.translation_file is not None
                        else os.path.join(data_dir, "gene.id.translation.tsv"))
    global_stats_dir = (args.global_stats_dir
                        if args.global_stats_dir is not None
                        else data_dir)
    top_x            = args.top_x
    EMPRES_type      = args.EMPRES_type
    cfg              = EMPRES_CONFIG[EMPRES_type]
    subdir           = cfg["subdir"]

    os.makedirs(save_dir, exist_ok=True)

    print("Configuration:")
    print(f"  DATA_DIR         = {data_dir}")
    print(f"  OUT_DIR          = {out_dir}  (read only)")
    print(f"  SAVE_DIR         = {save_dir}")
    print(f"  PHYTO_DATA_CSV   = {phyto_data_csv}")
    print(f"  TRANSLATION_FILE = {translation_file}")
    print(f"  GLOBAL_STATS_DIR = {global_stats_dir}  (used only when standardize=True)")
    print(f"  TOP_X            = {top_x}")
    print(f"  EMPRES_TYPE      = {EMPRES_type}  (subdir='{subdir}')")

    # ------------------------------------------------------------------
    # 2) Seeds and device
    # ------------------------------------------------------------------
    set_random_seeds(42)
    device = get_device()
    print(f"device: {device}")

    # ------------------------------------------------------------------
    # 3) Load SIEVE input .npy files (memory-mapped) and metadata
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

    metadata_path = os.path.join(data_dir, "sieve_metadata.csv")
    meta = pd.read_csv(metadata_path)
    print(f"\nLoaded metadata: {len(meta):,} rows; columns: {list(meta.columns)}")

    for required in ("array_index", "gene", "transcript",
                     "group_for_cross_validation", "hash_seq"):
        if required not in meta.columns:
            raise ValueError(
                f"sieve_metadata.csv is missing required column '{required}'."
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

    # ------------------------------------------------------------------
    # 4) Build test_groups dict from PhytoExpr training data
    # ------------------------------------------------------------------
    print("\nBuilding test_groups via GetTranslations() + GetGroups() "
          "(Camous's convention) ...")
    if not os.path.exists(translation_file):
        raise FileNotFoundError(f"Translation file not found: {translation_file}")
    if not os.path.exists(phyto_data_csv):
        raise FileNotFoundError(f"PhytoExpr data.csv not found: {phyto_data_csv}")

    translations = GetTranslations(translation_file)
    test_groups  = GetGroups(phyto_data_csv, translations)
    print(f"  translations : {len(translations):,} entries")
    print(f"  test_groups  : {len(test_groups):,} entries")

    # Look up each SIEVE row's CV group via its `gene` column (BD21.3 short form).
    # Rows whose gene is not present in `test_groups` will have NaN and will be
    # skipped (mirrors Camous's "skipping gene: ..." branch).
    meta["test_group"] = meta["gene"].map(test_groups)
    n_matched = int(meta["test_group"].notna().sum())
    n_skipped = n_samples - n_matched

    print(f"\n  SIEVE rows matched to a CV group : {n_matched:,} / {n_samples:,}")
    print(f"  SIEVE rows skipped (gene not in test_groups) : {n_skipped:,}")

    tg_counts = meta["test_group"].value_counts(dropna=False).sort_index()
    print(f"  test_group value counts (dropna=False):")
    print(tg_counts.to_string())

    if n_matched == 0:
        raise RuntimeError(
            "No SIEVE rows had a matching CV group; nothing to predict."
        )

    valid_test_groups = {"1", "2", "3", "4", "5"}
    unexpected = set(meta["test_group"].dropna().unique()) - valid_test_groups
    if unexpected:
        raise ValueError(
            f"Unexpected test_group values in metadata (not in 1..5): {sorted(unexpected)}"
        )

    # ------------------------------------------------------------------
    # 5) Placeholder target array (inference only; targets are unused)
    # ------------------------------------------------------------------
    dataset_target = np.zeros(n_samples, dtype=np.float32)

    # ------------------------------------------------------------------
    # 6) Per-fold batched inference
    # ------------------------------------------------------------------
    FOLDS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    predictions_matrix = np.full((n_samples, top_x), np.nan, dtype=np.float32)
    criterion = nn.MSELoss()

    for fold_num, (val_group, test_group) in enumerate(FOLDS, start=1):
        test_group_str = str(test_group)
        val_group_str  = str(val_group)

        fold_mask = (meta["test_group"] == test_group_str).to_numpy()
        fold_indices = np.where(fold_mask)[0]

        print()
        print("=" * 75)
        print(
            f"Fold {fold_num}: val_group={val_group_str}, test_group={test_group_str}, "
            f"EMPRES_type={EMPRES_type}, top_x={top_x}"
        )
        print("=" * 75)
        print(f"  SIEVE rows routed to this fold: {len(fold_indices):,}")

        if len(fold_indices) == 0:
            print(f"  Skipping fold {fold_num} (no SIEVE rows have CV group "
                  f"'{test_group_str}').")
            continue

        # Standardization stats
        if cfg["standardize"]:
            all_groups = ["1", "2", "3", "4", "5"]
            train_groups = sorted(
                set(all_groups) - {val_group_str, test_group_str}, key=int
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
            print(f"  Loaded global stats from {stats_path}")
        else:
            base_C = tss.shape[1]
            base_L = tss.shape[2]
            tss_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
            tss_std  = np.ones((1, base_C, base_L), dtype=np.float32)
            tts_mean = np.zeros((1, base_C, base_L), dtype=np.float32)
            tts_std  = np.ones((1, base_C, base_L), dtype=np.float32)
            extra_tss_mean = extra_tss_std = extra_tts_mean = extra_tts_std = None
            print("  Skipped global stats load (identity standardization for OHE input).")

        base_channels  = tss_mean.shape[1]
        extra_channels = extra_tss.shape[1] if extra_tss is not None else 0
        in_channels    = base_channels + extra_channels

        dataset = DNADualDataset(
            fold_indices,
            tss, tts, dataset_target,
            tss_mean, tss_std,
            tts_mean, tts_std,
            extra_tss = extra_tss,
            extra_tss_mean = extra_tss_mean,
            extra_tss_std  = extra_tss_std,
            extra_tts = extra_tts,
            extra_tts_mean = extra_tts_mean,
            extra_tts_std  = extra_tts_std,
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        run_dir    = os.path.join(out_dir, f"val{val_group}_test{test_group}", subdir)
        trials_csv = os.path.join(run_dir, "trial_results.csv")
        if not os.path.exists(trials_csv):
            raise FileNotFoundError(f"trial_results.csv not found: {trials_csv}")
        df_trials  = pd.read_csv(trials_csv)
        top_trials = df_trials.head(top_x).reset_index(drop=True)
        if len(top_trials) < top_x:
            raise RuntimeError(
                f"trial_results.csv only lists {len(top_trials)} models, "
                f"need at least top_x={top_x}."
            )

        for model_idx, (_, row) in enumerate(top_trials.iterrows()):
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
            preds = preds.squeeze()
            if preds.shape[0] != len(fold_indices):
                raise RuntimeError(
                    f"Prediction length mismatch: got {preds.shape[0]}, "
                    f"expected {len(fold_indices)}."
                )
            predictions_matrix[fold_indices, model_idx] = preds.astype(np.float32)
            print(f"  Loaded and ran inference for {checkpoint_file}")

    # ------------------------------------------------------------------
    # 7) Assemble the output DataFrame in Camous's exact TSV format
    # ------------------------------------------------------------------
    matched = meta.dropna(subset=["test_group"]).copy()
    matched["test_group"] = matched["test_group"].astype(str)

    for j in range(top_x):
        matched[f"model_{j+1}_pred"] = predictions_matrix[
            matched["array_index"].to_numpy(), j
        ]

    pred_cols_internal = [f"model_{j+1}_pred" for j in range(top_x)]
    n_rows_with_nan_preds = int(matched[pred_cols_internal].isna().any(axis=1).sum())
    if n_rows_with_nan_preds > 0:
        print(
            f"\nWARNING: {n_rows_with_nan_preds:,} matched rows still have "
            "NaN predictions and will be dropped from the output."
        )
        matched = matched[matched[pred_cols_internal].notna().all(axis=1)].reset_index(drop=True)

    # Explode by the space-separated plant-line cohort in group_for_cross_validation,
    # producing one output row per (plant_line_id, gene, transcript) triple --
    # exactly what Camous does with `for id in ids.split(' '): out.write(...)`.
    matched["id_list"] = matched["group_for_cross_validation"].astype(str).str.split(" ")
    exploded = matched.explode("id_list", ignore_index=True)
    print(f"\nExploded per-sequence predictions by cohort: {len(exploded):,} output rows.")

    if top_x == 5:
        # Camous's exact header row, including the `model_2_pred2` typo he wrote
        # in Data_processing/generate_predictions_hd5.py. Preserved verbatim for
        # bit-level parity with predictions_none/pred/emb/a2z.tsv; Guillaume's R
        # script normalises it via `sub("_pred.*", "", names(DF))` -> "model_2".
        output_pred_col_names = [
            "model_1_pred",
            "model_2_pred2",
            "model_3_pred",
            "model_4_pred",
            "model_5_pred",
        ]
    else:
        output_pred_col_names = [f"model_{j+1}_pred" for j in range(top_x)]

    out_df = pd.DataFrame({
        "id":         exploded["id_list"].values,
        "gene":       exploded["gene"].values,
        "transcript": exploded["transcript"].values,
        "hash(seq)":  exploded["hash_seq"].values,
        "test_group": exploded["test_group"].values,
    })
    for j, colname in enumerate(output_pred_col_names):
        out_df[colname] = exploded[f"model_{j+1}_pred"].values

    output_path = os.path.join(save_dir, f"predictions_EMPRES_{EMPRES_type}.tsv")
    out_df.to_csv(output_path, sep="\t", index=False, float_format="%f")
    print(f"\nSaved predictions to: {output_path}")
    print(f"  Shape: {out_df.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
