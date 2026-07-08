#!/usr/bin/env python
"""
SIEVE_OHE_seqs_efficient.py

Read the SIEVE data file `data.bd.csv`, extract 5000-bp TSS and TTS sequences
from the 10000-bp `promoter` and `terminator` columns, one-hot-encode them
into (N, 4, 5000) uint8 arrays, and save them alongside a per-row metadata
CSV that links every .npy row back to its (gene, transcript, cohort, hash).

Output files (all written into --out_dir):
    tss_OHE.npy               : (N, 4, 5000), uint8
    tts_OHE.npy               : (N, 4, 5000), uint8
    tss_sequences.npy         : (N,)         , fixed-width S5000 bytes
    tts_sequences.npy         : (N,)         , fixed-width S5000 bytes
    sieve_metadata.csv        : per-row metadata, aligned with .npy by array_index
    sieve_sequence_length_summary.csv : counts by promoter/terminator length

Metadata columns of sieve_metadata.csv:
    array_index                        # 0..N-1, aligned with .npy rows
    Unnamed: 0                         # original row number from data.bd.csv
    gene, species, transcript          # convention
    group_for_cross_validation         # line IDs (space-separated)
    hash_seq                           # Python hash(tss_extracted + tts_extracted)
    original_promoter_length           # sanity fields
    original_terminator_length
    extracted_tss_length
    extracted_tts_length

Column semantics (following `make.bd.data.py` convention exactly):
    'gene'                       BD21.3 short gene ID (e.g. "1G0000200")
    'species'                    BD21 long gene ID  (e.g. "Bradi1g00200"),
                                 used by generate_predictions_SIEVE.py to look up
                                 the CV group via gene.id.translation.tsv + data.csv.
    'transcript'                 BD21.3 transcript ID (e.g. "BdiBd21-3.1G0000200.1")
    'group_for_cross_validation' space-separated plant line IDs
    'hash_seq'                   Python hash(tss_extracted + tts_extracted) for downstream pass-through into predictions_EMPRES_0.tsv.

CLI:
    --data_file   Path to data.bd.csv
    --out_dir     Directory into which the .npy / metadata files are written
    --chunk_size  Chunk size for the streaming pass (default 10000)
"""

import argparse
import os

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================
EXPECTED_ORIGINAL_SEQ_LEN  = 10000
EXPECTED_EXTRACTED_SEQ_LEN = 5000

TSS_UPSTREAM   = 4000
TSS_DOWNSTREAM = 1000

TTS_UPSTREAM   = 1000
TTS_DOWNSTREAM = 4000

PROMOTER_COL   = "promoter"
TERMINATOR_COL = "terminator"

# From data.bd.csv: the columns we carry through into sieve_metadata.csv,
# in the same order written in make.bd.data.py.
# `median_TPM` and `gene_family` are dropped because they are 100% NaN in data.bd.csv.
METADATA_COLS = [
    "Unnamed: 0",
    "gene",
    "species",
    "transcript",
    "group_for_cross_validation",
]


# ============================================================
# Helpers
# ============================================================
def iter_csv_chunks(csv_path, chunksize):
    """Chunked reader for a .csv or .csv.gz file (compression auto-detected)."""
    reader = pd.read_csv(csv_path, chunksize=chunksize, compression="infer")
    for chunk in reader:
        yield chunk


def clean_sequence_columns(chunk):
    """Normalize sequence columns to nullable string dtype and empty->NA."""
    chunk[PROMOTER_COL]   = chunk[PROMOTER_COL].astype("string").str.strip()
    chunk[TERMINATOR_COL] = chunk[TERMINATOR_COL].astype("string").str.strip()
    chunk[PROMOTER_COL]   = chunk[PROMOTER_COL].replace("", pd.NA)
    chunk[TERMINATOR_COL] = chunk[TERMINATOR_COL].replace("", pd.NA)
    return chunk


def extract_tss_tts_sequences(chunk):
    """
    Extract 5000-bp TSS and TTS sequences from the 10000-bp promoter/terminator
    values (same slicing convention as the training-species and new-test-species
    OHE scripts):

        promoter[1000 : 6000]   -> TSS   (4000 upstream + 1000 downstream)
        terminator[4000 : 9000] -> TTS   (1000 upstream + 4000 downstream)
    """
    middle    = EXPECTED_ORIGINAL_SEQ_LEN // 2
    tss_start = middle - TSS_UPSTREAM
    tss_end   = middle + TSS_DOWNSTREAM
    tts_start = middle - TTS_UPSTREAM
    tts_end   = middle + TTS_DOWNSTREAM
    tss = chunk[PROMOTER_COL].str.slice(tss_start, tss_end)
    tts = chunk[TERMINATOR_COL].str.slice(tts_start, tts_end)
    return tss, tts


def one_hot_encode_batch(sequences):
    """
    Vectorized one-hot encoding for a batch of fixed-length DNA sequences.

    Output shape:  (batch_size, 4, EXPECTED_EXTRACTED_SEQ_LEN), dtype uint8.
    Channel order: A -> 0, C -> 1, G -> 2, T -> 3.
    Any non-ACGT base (including N and IUPAC ambiguity codes) is left as
    all-zero across the 4 channels (identical to the prior OHE scripts).
    """
    batch_size = len(sequences)

    seq_arr = np.asarray(sequences, dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}")
    seq_bytes = seq_arr.view(np.uint8).reshape(batch_size, EXPECTED_EXTRACTED_SEQ_LEN)

    lookup = np.full(256, 255, dtype=np.uint8)
    lookup[ord("A")] = 0
    lookup[ord("C")] = 1
    lookup[ord("G")] = 2
    lookup[ord("T")] = 3
    lookup[ord("a")] = 0
    lookup[ord("c")] = 1
    lookup[ord("g")] = 2
    lookup[ord("t")] = 3

    encoded = lookup[seq_bytes]

    ohe = np.zeros((batch_size, 4, EXPECTED_EXTRACTED_SEQ_LEN), dtype=np.uint8)
    valid = encoded != 255
    row_idx, pos_idx = np.nonzero(valid)
    channel_idx = encoded[row_idx, pos_idx]
    ohe[row_idx, channel_idx, pos_idx] = 1
    return ohe


# ============================================================
# Main
# ============================================================
def main():
    # ------------------------------------------------------------------
    # 1) Parse CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Extract 5000-bp TSS/TTS one-hot-encoded sequences from the SIEVE "
            "data.bd.csv file, and save them together with a per-row metadata CSV."
        )
    )
    parser.add_argument(
        "--data_file", type=str, required=True,
        help="Path to data.bd.csv (produced by make.bd.data.py).",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory in which to write the .npy files and sieve_metadata.csv.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10000,
        help="Chunk size for the streaming pandas passes over data.bd.csv (default 10000).",
    )
    args = parser.parse_args()

    data_file  = args.data_file
    out_dir    = args.out_dir
    chunk_size = args.chunk_size

    os.makedirs(out_dir, exist_ok=True)

    print("Configuration:")
    print(f"  DATA_FILE  = {data_file}")
    print(f"  OUT_DIR    = {out_dir}")
    print(f"  CHUNK_SIZE = {chunk_size}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"data.bd.csv not found: {data_file}")

    print(f"  File size  = {os.path.getsize(data_file) / (1024 ** 3):.3f} GiB")

    # ------------------------------------------------------------------
    # 2) Output paths
    # ------------------------------------------------------------------
    tss_sequences_path  = os.path.join(out_dir, "tss_sequences.npy")
    tts_sequences_path  = os.path.join(out_dir, "tts_sequences.npy")
    tss_ohe_path        = os.path.join(out_dir, "tss_OHE.npy")
    tts_ohe_path        = os.path.join(out_dir, "tts_OHE.npy")
    metadata_path       = os.path.join(out_dir, "sieve_metadata.csv")
    length_summary_path = os.path.join(out_dir, "sieve_sequence_length_summary.csv")

    # Remove any prior append-mode metadata output so this run starts fresh.
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    # ------------------------------------------------------------------
    # 3) PASS 1: count rows, validate lengths, count missing values
    # ------------------------------------------------------------------
    print("\nPASS 1: scanning dataset in chunks ...")

    total_rows = 0
    rows_after_removing_missing_seq = 0
    rows_exact_original_10000 = 0

    prom_lt_10000 = prom_eq_10000 = prom_gt_10000 = 0
    term_lt_10000 = term_eq_10000 = term_gt_10000 = 0

    for chunk_id, chunk in enumerate(iter_csv_chunks(data_file, chunk_size), start=1):
        total_rows += len(chunk)
        chunk = clean_sequence_columns(chunk)
        chunk_no_missing = chunk.dropna(subset=[PROMOTER_COL, TERMINATOR_COL])
        rows_after_removing_missing_seq += len(chunk_no_missing)

        p_len = chunk_no_missing[PROMOTER_COL].str.len()
        t_len = chunk_no_missing[TERMINATOR_COL].str.len()

        prom_lt_10000 += int((p_len < EXPECTED_ORIGINAL_SEQ_LEN).sum())
        prom_eq_10000 += int((p_len == EXPECTED_ORIGINAL_SEQ_LEN).sum())
        prom_gt_10000 += int((p_len > EXPECTED_ORIGINAL_SEQ_LEN).sum())

        term_lt_10000 += int((t_len < EXPECTED_ORIGINAL_SEQ_LEN).sum())
        term_eq_10000 += int((t_len == EXPECTED_ORIGINAL_SEQ_LEN).sum())
        term_gt_10000 += int((t_len > EXPECTED_ORIGINAL_SEQ_LEN).sum())

        exact_mask = (
            (p_len == EXPECTED_ORIGINAL_SEQ_LEN)
            & (t_len == EXPECTED_ORIGINAL_SEQ_LEN)
        )
        rows_exact_original_10000 += int(exact_mask.sum())

        if chunk_id % 10 == 0:
            print(f"  Scanned chunk {chunk_id:,} | total rows so far: {total_rows:,}")

    print("\nPASS 1 complete.")
    print(f"  Total rows in file             : {total_rows:,}")
    print(f"  Rows after missing-seq filter  : {rows_after_removing_missing_seq:,}")
    print(f"  Rows exactly 10000 / 10000     : {rows_exact_original_10000:,}")

    pd.DataFrame({
        "problem": [
            "promoter_length < 10000",
            "promoter_length == 10000",
            "promoter_length > 10000",
            "terminator_length < 10000",
            "terminator_length == 10000",
            "terminator_length > 10000",
            "rows with both promoter and terminator exactly 10000",
        ],
        "count": [
            prom_lt_10000, prom_eq_10000, prom_gt_10000,
            term_lt_10000, term_eq_10000, term_gt_10000,
            rows_exact_original_10000,
        ],
    }).to_csv(length_summary_path, index=False)
    print(f"  Length summary saved to        : {length_summary_path}")

    N = rows_exact_original_10000
    if N == 0:
        raise RuntimeError(
            "No rows with both promoter and terminator exactly 10000 bp. "
            "Nothing to write."
        )

    # ------------------------------------------------------------------
    # 4) Allocate memory-mapped output arrays
    # ------------------------------------------------------------------
    print("\nAllocating output .npy files ...")

    tss_sequences_memmap = np.lib.format.open_memmap(
        tss_sequences_path, mode="w+",
        dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}",
        shape=(N,),
    )
    tts_sequences_memmap = np.lib.format.open_memmap(
        tts_sequences_path, mode="w+",
        dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}",
        shape=(N,),
    )
    tss_ohe_memmap = np.lib.format.open_memmap(
        tss_ohe_path, mode="w+",
        dtype=np.uint8,
        shape=(N, 4, EXPECTED_EXTRACTED_SEQ_LEN),
    )
    tts_ohe_memmap = np.lib.format.open_memmap(
        tts_ohe_path, mode="w+",
        dtype=np.uint8,
        shape=(N, 4, EXPECTED_EXTRACTED_SEQ_LEN),
    )
    print(f"  tss_sequences: {tss_sequences_memmap.shape}, {tss_sequences_memmap.dtype}")
    print(f"  tts_sequences: {tts_sequences_memmap.shape}, {tts_sequences_memmap.dtype}")
    print(f"  tss_OHE      : {tss_ohe_memmap.shape}, {tss_ohe_memmap.dtype}")
    print(f"  tts_OHE      : {tts_ohe_memmap.shape}, {tts_ohe_memmap.dtype}")

    # ------------------------------------------------------------------
    # 5) PASS 2: extract, OHE-encode, write sequences and metadata
    # ------------------------------------------------------------------
    print("\nPASS 2: writing sequence, OHE, and metadata files ...")

    write_index = 0

    for chunk_id, chunk in enumerate(iter_csv_chunks(data_file, chunk_size), start=1):
        chunk = clean_sequence_columns(chunk)
        chunk = chunk.dropna(subset=[PROMOTER_COL, TERMINATOR_COL]).copy()

        p_len = chunk[PROMOTER_COL].str.len()
        t_len = chunk[TERMINATOR_COL].str.len()
        exact_mask = (
            (p_len == EXPECTED_ORIGINAL_SEQ_LEN)
            & (t_len == EXPECTED_ORIGINAL_SEQ_LEN)
        )
        chunk_exact = chunk.loc[exact_mask].copy()
        if len(chunk_exact) == 0:
            continue

        tss_extr, tts_extr = extract_tss_tts_sequences(chunk_exact)
        chunk_exact["tss_extracted"] = tss_extr
        chunk_exact["tts_extracted"] = tts_extr

        # Sanity check: extracted length must be exactly EXPECTED_EXTRACTED_SEQ_LEN.
        if not (
            chunk_exact["tss_extracted"].str.len().eq(EXPECTED_EXTRACTED_SEQ_LEN).all()
            and
            chunk_exact["tts_extracted"].str.len().eq(EXPECTED_EXTRACTED_SEQ_LEN).all()
        ):
            raise RuntimeError(
                "Extracted TSS/TTS length check failed: at least one sequence "
                f"is not exactly {EXPECTED_EXTRACTED_SEQ_LEN} bp."
            )

        n = len(chunk_exact)
        start = write_index
        end = start + n

        tss_batch = chunk_exact["tss_extracted"].astype(str).to_numpy()
        tts_batch = chunk_exact["tts_extracted"].astype(str).to_numpy()

        # Raw sequences (fixed-width bytes)
        tss_sequences_memmap[start:end] = np.asarray(
            tss_batch, dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}"
        )
        tts_sequences_memmap[start:end] = np.asarray(
            tts_batch, dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}"
        )

        # OHE arrays
        tss_ohe_memmap[start:end] = one_hot_encode_batch(tss_batch)
        tts_ohe_memmap[start:end] = one_hot_encode_batch(tts_batch)

        # Per-row hash of the extracted (tss + tts) 10000-char string, matching
        # the spirit of `hash(tss+tts)` in make.bd.data.py. Only used
        # as pass-through into the final predictions_EMPRES_0.tsv output
        hashes = [hash(str(tss_batch[i]) + str(tts_batch[i])) for i in range(n)]

        # Metadata slice, chunk-appended to sieve_metadata.csv.
        metadata_to_save = chunk_exact[METADATA_COLS].copy()
        metadata_to_save["array_index"]                = np.arange(start, end)
        metadata_to_save["hash_seq"]                   = hashes
        metadata_to_save["original_promoter_length"]   = chunk_exact[PROMOTER_COL].str.len().to_numpy()
        metadata_to_save["original_terminator_length"] = chunk_exact[TERMINATOR_COL].str.len().to_numpy()
        metadata_to_save["extracted_tss_length"]       = chunk_exact["tss_extracted"].str.len().to_numpy()
        metadata_to_save["extracted_tts_length"]       = chunk_exact["tts_extracted"].str.len().to_numpy()

        write_header = not os.path.exists(metadata_path)
        metadata_to_save.to_csv(metadata_path, mode="a", index=False, header=write_header)

        write_index = end

        if chunk_id % 10 == 0:
            print(f"  Wrote chunk {chunk_id:,} | rows written so far: {write_index:,}")

    # Flush all memmaps to disk before exiting.
    tss_sequences_memmap.flush()
    tts_sequences_memmap.flush()
    tss_ohe_memmap.flush()
    tts_ohe_memmap.flush()

    print("\nPASS 2 complete.")
    print(f"  Final rows written: {write_index:,}")

    if write_index != N:
        raise RuntimeError(
            f"Mismatch: wrote {write_index:,} rows, but expected {N:,}."
        )

    print("\nDone.")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
