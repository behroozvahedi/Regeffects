"""
Test_species_OHE_seqs_efficient.py

Extract 5000-bp TSS and TTS sequences from the 10000-bp `promoter` and `terminator`
columns of the test-species table (data.review.csv, plain or gzip-compressed), one-hot
encode them into (N, 4, 5000) uint8 arrays, and save them with a per-row metadata CSV
(linked to the .npy rows by `array_index`).

    TSS = promoter[1000:6000]   (4000 bp upstream + 1000 bp downstream)
    TTS = terminator[4000:9000] (1000 bp upstream + 4000 bp downstream)
    A, C, G, T -> channels 0..3 (lowercase accepted); any other character -> all-zero.

Rows are written in species blocks (order of test_species_set below), in file order
within each species. After all rows are written, the metadata CSV is sorted by
array_index, so metadata row i describes .npy row i.

CLI:
    --data_file      data.review.csv or data.review.csv.gz (compression detected from the name)
    --out_dir        directory for the .npy files, metadata and reports
    --chunk_size     rows per streamed chunk (default 10000)
    --expected_rows  optional: expected number of rows in the input file. If given and
                     different, a WARNING is printed (as before); if omitted, the count
                     is only reported.
    --expected_rows_after_filter
                     optional: expected rows left after removing missing promoter/terminator
                     (same WARNING behaviour).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Settings
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Extract 5000-bp TSS/TTS sequences from the test-species table, one-hot encode "
        "them, and save them with a per-row metadata CSV."
    )
)
parser.add_argument("--data_file", type=str, required=True,
                    help="Path to data.review.csv or data.review.csv.gz.")
parser.add_argument("--out_dir", type=str, required=True,
                    help="Directory in which to write the .npy files, metadata and reports.")
parser.add_argument("--chunk_size", type=int, default=10000,
                    help="Rows per streamed chunk (default 10000).")
parser.add_argument("--expected_rows", type=int, default=None,
                    help="Optional expected number of rows in the input file (WARNING if different).")
parser.add_argument("--expected_rows_after_filter", type=int, default=None,
                    help="Optional expected rows after removing missing promoter/terminator (WARNING if different).")
args = parser.parse_args()

DATA_PATH = args.data_file
OUT_DIR = Path(args.out_dir)
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Input file not found: {DATA_PATH}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = args.chunk_size

EXPECTED_INITIAL_ROWS = args.expected_rows                          # None = report only
EXPECTED_AFTER_REMOVE_NAN_SEQ_ROWS = args.expected_rows_after_filter  # None = report only
EXPECTED_COLUMNS = 9

print("Configuration:")
print(f"  DATA_FILE                  = {DATA_PATH}")
print(f"  OUT_DIR                    = {OUT_DIR}")
print(f"  CHUNK_SIZE                 = {CHUNK_SIZE}")
print(f"  EXPECTED_ROWS              = {EXPECTED_INITIAL_ROWS}")
print(f"  EXPECTED_ROWS_AFTER_FILTER = {EXPECTED_AFTER_REMOVE_NAN_SEQ_ROWS}")

EXPECTED_ORIGINAL_SEQ_LEN = 10000
EXPECTED_EXTRACTED_SEQ_LEN = 5000

TSS_UPSTREAM = 4000
TSS_DOWNSTREAM = 1000

TTS_UPSTREAM = 1000
TTS_DOWNSTREAM = 4000

PROMOTER_COL = "promoter"      # original promoter sequence, 10000 bp
TERMINATOR_COL = "terminator"  # original terminator sequence, 10000 bp

# test_species_set = [
#     'Csa', 'Sly', 'Sbi', 'Ath', 'Vvi', 'Osa', 'Mtr', 'Ppa', 'Gma',
#     'Bdi', 'Sit', 'Ptr', 'Svi', 'Bvu', 'Cre', 'Zma', 'Stu'
# ]

# # new test species for cross-species validation
test_species_set = ['Actinidia chinensis', 'Brachypodium sylvaticum',
       'Brassica oleracea', 'Panicum hallii', 'Pisum sativum',
       'Prunus persica', 'Secale cereale', 'Triticum dicoccoides',
       'Vigna unguiculata', 'Ziziphus jujuba']

metadata_cols = [
    "Unnamed: 0",
    "gene",
    "species",
    "transcript",
    "median_TPM",
    "gene_family",
    "group_for_cross_validation",
]


# ============================================================
# Helpers
# ============================================================

# def get_csv_name_inside_zip(zip_path):
#     with zipfile.ZipFile(zip_path, "r") as z:
#         csv_files = [f for f in z.namelist() if f.endswith(".csv")]
#         if len(csv_files) == 0:
#             raise ValueError("No .csv file found inside the zip file.")
#         if len(csv_files) > 1:
#             print("Multiple CSV files found. Reading:", csv_files[0])
#         return csv_files[0]


# def iter_csv_chunks(zip_path, chunksize=20000):
#     """
#     Chunked CSV reader from a zipped CSV.
#     This does not extract the CSV to disk.
#     """
#     csv_name = get_csv_name_inside_zip(zip_path)

#     with zipfile.ZipFile(zip_path, "r") as z:
#         with z.open(csv_name) as f:
#             reader = pd.read_csv(f, chunksize=chunksize)

#             for chunk in reader:
#                 yield chunk

def iter_csv_chunks(csv_path, chunksize=10000):
    """
    Chunked CSV reader from a zipped CSV.gz gzip file.
    This does not extract the CSV to disk.
    """
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        compression="infer"
    )

    for chunk in reader:
        yield chunk


def clean_sequence_columns(chunk):
    """
    Convert sequence columns to pandas string dtype,
    strip whitespace, and treat empty strings as missing.
    """
    chunk[PROMOTER_COL] = chunk[PROMOTER_COL].astype("string").str.strip()
    chunk[TERMINATOR_COL] = chunk[TERMINATOR_COL].astype("string").str.strip()

    chunk[PROMOTER_COL] = chunk[PROMOTER_COL].replace("", pd.NA)
    chunk[TERMINATOR_COL] = chunk[TERMINATOR_COL].replace("", pd.NA)

    return chunk


def update_missing_report(chunk, missing_nan_counts, missing_empty_counts):
    """
    Updates missing-value counters for all columns.
    """
    nan_counts = chunk.isna().sum()

    for col in chunk.columns:
        missing_nan_counts[col] = missing_nan_counts.get(col, 0) + int(nan_counts[col])

        if chunk[col].dtype == "object" or str(chunk[col].dtype) == "string":
            empty_count = chunk[col].astype("string").str.strip().eq("").sum()
            missing_empty_counts[col] = missing_empty_counts.get(col, 0) + int(empty_count)
        else:
            missing_empty_counts[col] = missing_empty_counts.get(col, 0)


def write_append_csv(df, path):
    """
    Append dataframe to CSV, writing header only if file does not exist.
    """
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def extract_tss_tts_sequences(chunk):
    """
    Extract final 5000-bp tss and tts sequences from original 10000-bp
    promoter and terminator sequences.

    For promoter/tss:
        TSS is at the exact middle of the 10000-bp promoter sequence.
        Extract 4000 bp upstream and 1000 bp downstream:
            promoter[1000:6000]

    For terminator/tts:
        TTS is at the exact middle of the 10000-bp terminator sequence.
        Extract 1000 bp upstream and 4000 bp downstream:
            terminator[4000:9000]
    """
    middle = EXPECTED_ORIGINAL_SEQ_LEN // 2

    tss_start = middle - TSS_UPSTREAM
    tss_end = middle + TSS_DOWNSTREAM

    tts_start = middle - TTS_UPSTREAM
    tts_end = middle + TTS_DOWNSTREAM

    tss_sequences = chunk[PROMOTER_COL].str.slice(tss_start, tss_end)
    tts_sequences = chunk[TERMINATOR_COL].str.slice(tts_start, tts_end)

    return tss_sequences, tts_sequences


def one_hot_encode_batch(sequences):
    """
    Vectorized one-hot encoding for a batch of fixed-length sequences.

    Output shape:
        (batch_size, 4, 5000)

    Encoding:
        A -> channel 0
        C -> channel 1
        G -> channel 2
        T -> channel 3

    Ambiguous bases such as N remain all-zero across the 4 channels.
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
# Output paths
# ============================================================

missing_original_path = OUT_DIR / "missing_values_original.csv"
missing_after_seq_filter_path = OUT_DIR / "missing_values_after_sequence_filtering.csv"

long_sequences_path = OUT_DIR / "original_sequences_longer_than_10000_with_metadata.csv"
not_exact_path = OUT_DIR / "original_sequences_not_exactly_10000_with_metadata.csv"
length_summary_path = OUT_DIR / "original_sequence_length_summary.csv"

clean_metadata_path = OUT_DIR / "clean_metadata_exact_10000_original_5000_extracted.csv"

tss_sequences_path = OUT_DIR / "tss_sequences.npy"
tts_sequences_path = OUT_DIR / "tts_sequences.npy"

tss_ohe_path = OUT_DIR / "tss_OHE.npy"
tts_ohe_path = OUT_DIR / "tts_OHE.npy"


# Remove old append-output files if they exist
for path in [
    long_sequences_path,
    not_exact_path,
    clean_metadata_path,
]:
    if path.exists():
        path.unlink()


# ============================================================
# Pass 1: validate, count, and write reports
# ============================================================

total_rows = 0
total_cols = None

rows_after_removing_missing_seq = 0
rows_exact_original_10000 = 0

promoter_lt_10000 = 0
promoter_eq_10000 = 0
promoter_gt_10000 = 0

terminator_lt_10000 = 0
terminator_eq_10000 = 0
terminator_gt_10000 = 0

rows_with_long_sequence = 0
rows_not_exact_10000 = 0

species_counts_exact = {sp: 0 for sp in test_species_set}
unknown_species_count = 0

missing_nan_original = {}
missing_empty_original = {}

missing_nan_after_seq_filter = {}
missing_empty_after_seq_filter = {}

print("PASS 1: scanning dataset in chunks...")

for chunk_id, chunk in enumerate(iter_csv_chunks(DATA_PATH, CHUNK_SIZE), start=1):
    total_rows += len(chunk)

    if total_cols is None:
        total_cols = len(chunk.columns)
        print("Detected columns:", chunk.columns.tolist())
        missing_cols = [c for c in metadata_cols + [PROMOTER_COL, TERMINATOR_COL] if c not in chunk.columns]
        if missing_cols:
            raise ValueError(f"Input file is missing required column(s): {missing_cols}")

    # Missing report before filtering
    update_missing_report(chunk, missing_nan_original, missing_empty_original)

    # Clean sequence columns
    chunk = clean_sequence_columns(chunk)

    # Remove missing promoter/terminator
    chunk_no_missing_seq = chunk.dropna(subset=[PROMOTER_COL, TERMINATOR_COL]).copy()
    rows_after_removing_missing_seq += len(chunk_no_missing_seq)

    # Missing report after sequence filtering
    update_missing_report(
        chunk_no_missing_seq,
        missing_nan_after_seq_filter,
        missing_empty_after_seq_filter
    )

    # Original sequence length checks
    promoter_len = chunk_no_missing_seq[PROMOTER_COL].str.len()
    terminator_len = chunk_no_missing_seq[TERMINATOR_COL].str.len()

    chunk_no_missing_seq["promoter_length"] = promoter_len
    chunk_no_missing_seq["terminator_length"] = terminator_len

    promoter_lt_10000 += int((promoter_len < EXPECTED_ORIGINAL_SEQ_LEN).sum())
    promoter_eq_10000 += int((promoter_len == EXPECTED_ORIGINAL_SEQ_LEN).sum())
    promoter_gt_10000 += int((promoter_len > EXPECTED_ORIGINAL_SEQ_LEN).sum())

    terminator_lt_10000 += int((terminator_len < EXPECTED_ORIGINAL_SEQ_LEN).sum())
    terminator_eq_10000 += int((terminator_len == EXPECTED_ORIGINAL_SEQ_LEN).sum())
    terminator_gt_10000 += int((terminator_len > EXPECTED_ORIGINAL_SEQ_LEN).sum())

    long_mask = (
        (promoter_len > EXPECTED_ORIGINAL_SEQ_LEN) |
        (terminator_len > EXPECTED_ORIGINAL_SEQ_LEN)
    )

    not_exact_mask = (
        (promoter_len != EXPECTED_ORIGINAL_SEQ_LEN) |
        (terminator_len != EXPECTED_ORIGINAL_SEQ_LEN)
    )

    exact_mask = (
        (promoter_len == EXPECTED_ORIGINAL_SEQ_LEN) &
        (terminator_len == EXPECTED_ORIGINAL_SEQ_LEN)
    )

    rows_with_long_sequence += int(long_mask.sum())
    rows_not_exact_10000 += int(not_exact_mask.sum())
    rows_exact_original_10000 += int(exact_mask.sum())

    # Save problematic rows immediately, without storing them in memory
    if long_mask.any():
        write_append_csv(
            chunk_no_missing_seq.loc[long_mask],
            long_sequences_path
        )

    if not_exact_mask.any():
        write_append_csv(
            chunk_no_missing_seq.loc[not_exact_mask],
            not_exact_path
        )

    # Species counts among final exact rows
    exact_chunk = chunk_no_missing_seq.loc[exact_mask]

    species_counts = exact_chunk["species"].value_counts()

    for sp in test_species_set:
        species_counts_exact[sp] += int(species_counts.get(sp, 0))

    unknown_species_count += int((~exact_chunk["species"].isin(test_species_set)).sum())

    if chunk_id % 10 == 0:
        print(f"Scanned chunk {chunk_id:,} | total rows scanned: {total_rows:,}")


print("\nPASS 1 complete.")
print("Initial shape detected:", (total_rows, total_cols))
print("Rows after removing missing promoter/terminator:", rows_after_removing_missing_seq)
print("Rows with both original promoter and terminator exactly 10000 bp:", rows_exact_original_10000)


# ============================================================
# Save missing-value reports
# ============================================================

all_cols = list(missing_nan_original.keys())

missing_original_df = pd.DataFrame({
    "column": all_cols,
    "NaN_count": [missing_nan_original.get(c, 0) for c in all_cols],
    "empty_string_count": [missing_empty_original.get(c, 0) for c in all_cols],
})

missing_original_df["total_empty_or_NaN"] = (
    missing_original_df["NaN_count"] +
    missing_original_df["empty_string_count"]
)

missing_original_df.to_csv(missing_original_path, index=False)


all_cols_after = list(missing_nan_after_seq_filter.keys())

missing_after_df = pd.DataFrame({
    "column": all_cols_after,
    "NaN_count": [missing_nan_after_seq_filter.get(c, 0) for c in all_cols_after],
    "empty_string_count": [missing_empty_after_seq_filter.get(c, 0) for c in all_cols_after],
})

missing_after_df["total_empty_or_NaN"] = (
    missing_after_df["NaN_count"] +
    missing_after_df["empty_string_count"]
)

missing_after_df.to_csv(missing_after_seq_filter_path, index=False)


# ============================================================
# Save length summary
# ============================================================

length_summary = pd.DataFrame({
    "problem": [
        "promoter_length < 10000",
        "promoter_length == 10000",
        "promoter_length > 10000",
        "terminator_length < 10000",
        "terminator_length == 10000",
        "terminator_length > 10000",
        "rows with promoter or terminator longer than 10000",
        "rows with promoter or terminator not exactly 10000",
        "rows with both promoter and terminator exactly 10000",
    ],
    "count": [
        promoter_lt_10000,
        promoter_eq_10000,
        promoter_gt_10000,
        terminator_lt_10000,
        terminator_eq_10000,
        terminator_gt_10000,
        rows_with_long_sequence,
        rows_not_exact_10000,
        rows_exact_original_10000,
    ]
})

length_summary.to_csv(length_summary_path, index=False)

species_counts_df = pd.DataFrame({
    "species": test_species_set,
    "count_exact_10000_original": [species_counts_exact[sp] for sp in test_species_set]
})

species_counts_df.to_csv(OUT_DIR / "species_counts_exact_10000_original.csv", index=False)


if unknown_species_count > 0:
    raise ValueError(
        f"{unknown_species_count:,} rows have species not included in test_species_set. "
        "Please inspect the species names before continuing."
    )

species_start_offsets = {}
cursor = 0

for sp in test_species_set:
    species_start_offsets[sp] = cursor
    cursor += species_counts_exact[sp]

species_write_offsets = species_start_offsets.copy()

print("\nSpecies write offsets:")
for sp in test_species_set:
    print(
        sp,
        "start:",
        species_start_offsets[sp],
        "count:",
        species_counts_exact[sp]
    )


# ============================================================
# Checks
# ============================================================

if EXPECTED_INITIAL_ROWS is None:
    if total_cols != EXPECTED_COLUMNS:
        print(f"WARNING: detected {total_cols} columns, expected {EXPECTED_COLUMNS}")
    print(f"Initial shape {(total_rows, total_cols)} (no expected row count given; reported only).")
elif (total_rows, total_cols) != (EXPECTED_INITIAL_ROWS, EXPECTED_COLUMNS):
    print(
        f"WARNING: detected initial shape {(total_rows, total_cols)}, "
        f"expected {(EXPECTED_INITIAL_ROWS, EXPECTED_COLUMNS)}"
    )
else:
    print("Initial shape check passed.")

if EXPECTED_AFTER_REMOVE_NAN_SEQ_ROWS is None:
    print(f"Rows after removing missing promoter/terminator: {rows_after_removing_missing_seq:,} "
          f"(no expected count given; reported only).")
elif rows_after_removing_missing_seq != EXPECTED_AFTER_REMOVE_NAN_SEQ_ROWS:
    print(
        f"WARNING: rows after removing missing promoter/terminator = "
        f"{rows_after_removing_missing_seq:,}, expected "
        f"{EXPECTED_AFTER_REMOVE_NAN_SEQ_ROWS:,}"
    )
else:
    print(f"Missing-sequence filtering check passed: {rows_after_removing_missing_seq:,} rows.")

if rows_not_exact_10000 > 0:
    print(
        f"WARNING: {rows_not_exact_10000:,} rows have promoter or terminator "
        f"sequence length not exactly 10000 bp."
    )
else:
    print("All non-missing original promoter and terminator sequences are exactly 10000 bp.")

if unknown_species_count > 0:
    print(f"WARNING: {unknown_species_count:,} rows have species not in test_species_set.")


# ============================================================
# Allocate memory-mapped output arrays
# ============================================================

print("\nAllocating output .npy files...")

tss_sequences_memmap = np.lib.format.open_memmap(
    tss_sequences_path,
    mode="w+",
    dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}",
    shape=(rows_exact_original_10000,)
)

tts_sequences_memmap = np.lib.format.open_memmap(
    tts_sequences_path,
    mode="w+",
    dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}",
    shape=(rows_exact_original_10000,)
)

tss_ohe_memmap = np.lib.format.open_memmap(
    tss_ohe_path,
    mode="w+",
    dtype=np.uint8,
    shape=(rows_exact_original_10000, 4, EXPECTED_EXTRACTED_SEQ_LEN)
)

tts_ohe_memmap = np.lib.format.open_memmap(
    tts_ohe_path,
    mode="w+",
    dtype=np.uint8,
    shape=(rows_exact_original_10000, 4, EXPECTED_EXTRACTED_SEQ_LEN)
)

print("Allocated:")
print("tss_sequences:", tss_sequences_memmap.shape, tss_sequences_memmap.dtype)
print("tts_sequences:", tts_sequences_memmap.shape, tts_sequences_memmap.dtype)
print("tss_OHE:", tss_ohe_memmap.shape, tss_ohe_memmap.dtype)
print("tts_OHE:", tts_ohe_memmap.shape, tts_ohe_memmap.dtype)


# ============================================================
# Pass 2: write valid sequences and OHE arrays chunk by chunk
# ============================================================

print("\nPASS 2: writing sequence and OHE files...")

write_index = 0

for chunk_id, chunk in enumerate(iter_csv_chunks(DATA_PATH, CHUNK_SIZE), start=1):
    chunk = clean_sequence_columns(chunk)

    chunk = chunk.dropna(subset=[PROMOTER_COL, TERMINATOR_COL]).copy()

    promoter_len = chunk[PROMOTER_COL].str.len()
    terminator_len = chunk[TERMINATOR_COL].str.len()

    exact_mask = (
        (promoter_len == EXPECTED_ORIGINAL_SEQ_LEN) &
        (terminator_len == EXPECTED_ORIGINAL_SEQ_LEN)
    )

    chunk_exact = chunk.loc[exact_mask].copy()

    if len(chunk_exact) == 0:
        continue

    tss_extracted, tts_extracted = extract_tss_tts_sequences(chunk_exact)

    chunk_exact["tss_extracted"] = tss_extracted
    chunk_exact["tts_extracted"] = tts_extracted

    if not (
        chunk_exact["tss_extracted"].str.len().eq(EXPECTED_EXTRACTED_SEQ_LEN).all() and
        chunk_exact["tts_extracted"].str.len().eq(EXPECTED_EXTRACTED_SEQ_LEN).all()
    ):
        raise RuntimeError(
            "Extracted tss/tts sequence length check failed. "
            "At least one extracted sequence is not exactly 5000 bp."
        )

    for sp in test_species_set:
        sp_chunk = chunk_exact[chunk_exact["species"] == sp]

        if len(sp_chunk) == 0:
            continue

        n = len(sp_chunk)
        start = species_write_offsets[sp]
        end = start + n

        tss_batch = sp_chunk["tss_extracted"].astype(str).to_numpy()
        tts_batch = sp_chunk["tts_extracted"].astype(str).to_numpy()

        # Save raw fixed-width extracted sequences
        tss_sequences_memmap[start:end] = np.asarray(
            tss_batch,
            dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}"
        )

        tts_sequences_memmap[start:end] = np.asarray(
            tts_batch,
            dtype=f"S{EXPECTED_EXTRACTED_SEQ_LEN}"
        )

        # Save OHE arrays
        tss_ohe_memmap[start:end] = one_hot_encode_batch(tss_batch)
        tts_ohe_memmap[start:end] = one_hot_encode_batch(tts_batch)

        species_write_offsets[sp] = end
        write_index += n

        # Save cleaned metadata, chunk by chunk
        metadata_to_save = sp_chunk[metadata_cols].copy()
        metadata_to_save["array_index"] = np.arange(start, end)
        metadata_to_save["original_promoter_length"] = sp_chunk[PROMOTER_COL].str.len().to_numpy()
        metadata_to_save["original_terminator_length"] = sp_chunk[TERMINATOR_COL].str.len().to_numpy()
        metadata_to_save["extracted_tss_length"] = sp_chunk["tss_extracted"].str.len().to_numpy()
        metadata_to_save["extracted_tts_length"] = sp_chunk["tts_extracted"].str.len().to_numpy()
        write_append_csv(metadata_to_save, clean_metadata_path)

    if chunk_id % 10 == 0:
        print(f"Wrote chunk {chunk_id:,} | rows written: {write_index:,}")


for sp in test_species_set:
    expected_end = species_start_offsets[sp] + species_counts_exact[sp]
    observed_end = species_write_offsets[sp]

    if observed_end != expected_end:
        raise RuntimeError(
            f"Species writing mismatch for {sp}: "
            f"observed_end={observed_end}, expected_end={expected_end}"
        )


# Flush all memmaps
tss_sequences_memmap.flush()
tts_sequences_memmap.flush()
tss_ohe_memmap.flush()
tts_ohe_memmap.flush()

print("\nPASS 2 complete.")
print("Final rows written:", write_index)

if write_index != rows_exact_original_10000:
    raise RuntimeError(
        f"Mismatch: wrote {write_index:,} rows, but expected {rows_exact_original_10000:,}."
    )

# ============================================================
# Sort the metadata CSV by array_index
# ============================================================
# The .npy rows are written into species blocks, while metadata rows are appended in
# reading order (chunk by chunk), so with more than one chunk the CSV is not in .npy
# order. Re-write it sorted by array_index so that metadata row i describes .npy row i.
# All columns are read as text (no NA conversion) so values are written back unchanged.

print("\nSorting metadata by array_index ...")
meta_sorted = pd.read_csv(clean_metadata_path, dtype=str, keep_default_na=False)
array_index_values = meta_sorted["array_index"].astype(np.int64).to_numpy()
order = np.argsort(array_index_values, kind="stable")
meta_sorted = meta_sorted.iloc[order]

if len(meta_sorted) != write_index or not np.array_equal(
    array_index_values[order], np.arange(write_index)
):
    raise RuntimeError(
        "Metadata array_index is not exactly 0..N-1 "
        f"({len(meta_sorted):,} metadata rows, {write_index:,} .npy rows)."
    )

tmp_metadata_path = clean_metadata_path.with_name(clean_metadata_path.name + ".tmp")
if tmp_metadata_path.exists():
    tmp_metadata_path.unlink()
write_append_csv(meta_sorted, tmp_metadata_path)
tmp_metadata_path.replace(clean_metadata_path)
print(f"Metadata sorted: {len(meta_sorted):,} rows, array_index 0..{write_index - 1:,}")
del meta_sorted

print("\nDone.")
print("Output directory:", OUT_DIR.resolve())
