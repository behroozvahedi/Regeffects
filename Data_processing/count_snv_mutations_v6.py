#!/usr/bin/env python3
"""Count SNV mutation classes per individual ID in primary-transcript TSS/TTS windows.

Input requirements:
  - One row per reference-or-alternate TSS/TTS sequence.
  - Required sequence columns: gene, transcript, promoter, terminator,
    group_for_cross_validation.
  - Rows for a gene are expected to be contiguous.
  - The transcript column identifies the transcript for each row.

Primary transcript filtering:
  - Provide --primary-transcripts with a one-column or delimited file containing
    the transcript IDs to include.
  - The primary transcript file may have a header named transcript. If not, the
    first column is used.
  - Only input rows whose transcript ID is present in the primary transcript
    file are included.

Reference choice:
  - Within each gene + transcript comparison group, the row with the most IDs in
    group_for_cross_validation is used as the reference by default.
  - Use --reference-policy first to force the first row in each group to be the
    reference.

Truncation:
  - promoter/TSS:    4000 bp upstream + 1000 bp downstream = seq[1000:6000]
  - terminator/TTS: 1000 bp upstream + 4000 bp downstream = seq[4000:9000]

IUPAC ambiguity codes:
  - Sequences may contain IUPAC heterozygosity codes (W, S, M, K, R, Y, B, D,
    H, V) produced by make.bd.data.py for heterozygous variant positions.
  - Every differing position counts as exactly one integer mutation.  The
    substitution class is resolved as follows:

      pure ref, IUPAC alt  -- one alt allele matches ref; the other is the
                               mutant allele → ref_base>mutant_allele (e.g.
                               ref=A, alt=W=AT → A>T).

      IUPAC ref, pure alt,
      alt ∈ ref alleles    -- the occasionally-het reference fixed one allele;
                               the complementary ref allele is treated as the
                               source → other_ref_allele>alt_base (e.g.
                               ref=W=AT, alt=A → T>A).

  - All other IUPAC combinations (alt not in ref alleles, both sides IUPAC,
    N, unknown characters) are unresolvable to a single typed substitution and
    are skipped; they are recorded in the skip count reported to stderr.

Output:
  - One TSV row per included gene + transcript + individual ID that appears on
    an alternate-sequence row.
  - First columns: gene, transcript, id.
  - Remaining columns: the 12 possible A/T/C/G substitution classes, counted as
    reference_base>alternate_base across both TSS/promoter and TTS/terminator
    windows for that individual ID.  All values are non-negative integers.
  - If the same ID appears in multiple alternate rows for the same gene +
    transcript, counts are summed for that ID.
  - Reference IDs are omitted by default because they have zero mutations; pass
    --include-reference-ids to emit zero-count rows for them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional, TextIO

BASES = "ATCG"
MUTATION_TYPES = [f"{ref}>{alt}" for ref in BASES for alt in BASES if alt != ref]
REQUIRED_COLUMNS = {"gene", "transcript", "promoter", "terminator", "group_for_cross_validation"}

# IUPAC ambiguity-code expansion to constituent unambiguous bases.
# N (no-data sentinel from make.bd.data.py) and any other character not listed
# here are treated as unresolvable and are skipped.
IUPAC_EXPANSION: dict[str, list[str]] = {
    "A": ["A"],
    "T": ["T"],
    "C": ["C"],
    "G": ["G"],
    "W": ["A", "T"],        # Weak
    "S": ["C", "G"],        # Strong
    "M": ["A", "C"],        # aMino
    "K": ["G", "T"],        # Keto
    "R": ["A", "G"],        # puRine
    "Y": ["C", "T"],        # pYrimidine
    "B": ["C", "G", "T"],   # not A
    "D": ["A", "G", "T"],   # not C
    "H": ["A", "C", "T"],   # not G
    "V": ["A", "C", "G"],   # not T
}


@dataclass
class SequenceRecord:
    gene: str
    transcript: str
    row_number: int
    tss: str
    tts: str
    ids: tuple[str, ...]

    @property
    def id_count(self) -> int:
        return len(self.ids)


@dataclass
class ProcessingStats:
    input_rows_seen: int = 0
    rows_kept_after_transcript_filter: int = 0
    rows_skipped_by_transcript_filter: int = 0
    gene_transcript_groups_seen: int = 0
    alternate_rows_seen: int = 0
    alternate_id_assignments_seen: int = 0
    output_rows_written: int = 0
    references_not_first: int = 0
    skipped_non_atcg: int = 0
    alternate_rows_without_ids: int = 0


def open_text(path: str) -> TextIO:
    """Open plain text or gzip-compressed text for reading."""
    if path == "-":
        return sys.stdin
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def dialect_from_name(delimiter: str) -> csv.Dialect:
    """Return a CSV dialect for an explicit delimiter choice."""
    class ExplicitDialect(csv.excel):
        pass
    ExplicitDialect.delimiter = "\t" if delimiter == "tab" else ","
    return ExplicitDialect


def detect_dialect(handle: TextIO, delimiter: str) -> csv.Dialect:
    """Detect CSV/TSV dialect from the beginning of an open file handle."""
    if delimiter != "auto":
        return dialect_from_name(delimiter)

    sample = handle.read(64 * 1024)
    handle.seek(0)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        class FallbackDialect(csv.excel_tab):
            pass
        return FallbackDialect


def parse_ids(value: str) -> tuple[str, ...]:
    """Return non-empty, non-NA IDs from group_for_cross_validation."""
    if value is None:
        return ()
    ids = []
    seen = set()
    for tok in re.split(r"[\s,;]+", value.strip()):
        if not tok or tok.upper() == "NA":
            continue
        # Keep the first occurrence if an ID is accidentally repeated in a cell.
        if tok not in seen:
            ids.append(tok)
            seen.add(tok)
    return tuple(ids)


def tss_window(seq: str, center: int = 5000) -> str:
    """Return 4000 bp upstream and 1000 bp downstream of TSS."""
    return seq[center - 4000 : center + 1000]


def tts_window(seq: str, center: int = 5000) -> str:
    """Return 1000 bp upstream and 4000 bp downstream of TTS."""
    return seq[center - 1000 : center + 4000]


def require_window_length(window: str, expected: int, gene: str, transcript: str, row_number: int, column: str) -> None:
    if len(window) != expected:
        raise ValueError(
            f"Gene {gene!r}, transcript {transcript!r}, input row {row_number}, column {column!r}: "
            f"expected truncated length {expected}, got {len(window)}. "
            "Check that the original sequence length and center coordinate are correct."
        )


def resolve_iupac_substitution(
    ref_base: str,
    alt_base: str,
) -> Optional[str]:
    """Return the substitution string 'X>Y' for a differing ref/alt character pair.

    Handles IUPAC ambiguity codes produced by make.bd.data.py:

      pure ref, IUPAC alt  -- the alternate is heterozygous; one allele equals
                               the reference, the other is the mutant allele.
                               Returns ref>mutant (e.g. A, W=AT → 'A>T').

      IUPAC ref, pure alt,
      alt ∈ ref alleles    -- the reference is occasionally het; the alternate
                               fixed one of the reference alleles.  The
                               complementary ref allele is the source of the
                               substitution.  Returns other>alt (e.g. W=AT,
                               A → 'T>A').

    Returns None for all other combinations (both IUPAC, alt not among ref
    alleles, triallelic positions, unknown characters) — these are logged by
    the caller as unresolvable.
    """
    ref_bases = IUPAC_EXPANSION.get(ref_base)
    alt_bases = IUPAC_EXPANSION.get(alt_base)

    if ref_bases is None or alt_bases is None:
        return None  # Unknown character (e.g. N, gap)

    if len(ref_bases) == 1 and len(alt_bases) == 1:
        # Both pure ATCG: straightforward substitution.
        return f"{ref_bases[0]}>{alt_bases[0]}"

    if len(ref_bases) == 1:
        # Pure reference, IUPAC alternate (het mutation in alternate line).
        # One alt allele matches the reference; the other is the mutant allele.
        ref_b = ref_bases[0]
        mutant = [b for b in alt_bases if b != ref_b]
        if len(mutant) == 1:
            return f"{ref_b}>{mutant[0]}"
        # Multiple non-ref alleles (triallelic): unresolvable.
        return None

    if len(alt_bases) == 1:
        # IUPAC reference (occasional het site), pure alternate.
        alt_b = alt_bases[0]
        if alt_b in ref_bases:
            # Alternate fixed one of the reference alleles; the complementary
            # ref allele is the inferred source.
            other = [b for b in ref_bases if b != alt_b]
            if len(other) == 1:
                return f"{other[0]}>{alt_b}"
        # Alt not in ref alleles, or ref had 3+ alleles: unresolvable.
        return None

    # Both IUPAC: unresolvable.
    return None


def add_snv_counts(
    counts: Counter[str],
    ref_seq: str,
    alt_seq: str,
    gene: str,
    transcript: str,
    row_number: int,
    sequence_label: str,
    skipped_non_atcg: Counter[str],
) -> int:
    """Compare two equal-length windows and add integer SNV substitution counts.

    Each position where ref and alt characters differ contributes exactly 1 to
    the appropriate substitution counter.  IUPAC ambiguity codes are resolved
    via resolve_iupac_substitution(); positions that cannot be resolved to a
    single typed substitution are recorded in skipped_non_atcg and omitted.

    Returns the number of SNVs counted (excluding skipped positions).
    """
    if len(ref_seq) != len(alt_seq):
        raise ValueError(
            f"Gene {gene!r}, transcript {transcript!r}, input row {row_number}, {sequence_label}: "
            f"reference and alternate windows have different lengths "
            f"({len(ref_seq)} vs {len(alt_seq)})."
        )

    snv_count = 0
    for ref_base, alt_base in zip(ref_seq.upper(), alt_seq.upper()):
        if ref_base == alt_base:
            continue

        substitution = resolve_iupac_substitution(ref_base, alt_base)
        if substitution is None:
            skipped_non_atcg[f"{gene}:{transcript}:{sequence_label}"] += 1
        else:
            counts[substitution] += 1
            snv_count += 1

    return snv_count


def choose_reference(records: list[SequenceRecord], reference_policy: str) -> int:
    """Return the index of the reference record for a comparison group."""
    if reference_policy == "first":
        return 0
    # max is stable: ties keep the first row with the maximum ID count.
    return max(range(len(records)), key=lambda i: records[i].id_count)


def first_nonempty_field(row: list[str]) -> str:
    """Return the first non-empty field from a list of strings."""
    for value in row:
        value = value.strip()
        if value:
            return value
    return ""


def load_primary_transcripts(
    path: str,
    delimiter: str = "auto",
    id_column: str = "auto",
) -> set[str]:
    """Load primary transcript IDs from a one-column or delimited text file."""
    if not path or path.lower() in {"none", "all"}:
        raise ValueError("A primary transcript file is required. Pass it with --primary-transcripts.")

    with open_text(path) as fh:
        dialect = detect_dialect(fh, delimiter)
        rows = list(csv.reader(fh, dialect=dialect))

    rows = [row for row in rows if row and first_nonempty_field(row)]
    if not rows:
        raise ValueError(f"Primary transcript file {path!r} is empty.")

    header = [value.strip() for value in rows[0]]
    lower_header = [value.lower() for value in header]

    has_header = False
    column_index = 0
    if id_column != "auto":
        if id_column not in header:
            raise ValueError(
                f"Requested primary transcript ID column {id_column!r}, but observed header was {header}."
            )
        has_header = True
        column_index = header.index(id_column)
    elif "transcript" in lower_header:
        has_header = True
        column_index = lower_header.index("transcript")
    elif len(rows[0]) > 1:
        # For multi-column files without a recognizable transcript header, use
        # the first column. Treat the first row as data.
        has_header = False
        column_index = 0
    else:
        # One-column files commonly either have a transcript header or no header.
        # Treat an exact "transcript" first line as a header; otherwise keep it.
        has_header = lower_header[0] == "transcript"
        column_index = 0

    data_rows = rows[1:] if has_header else rows
    transcript_ids = {
        row[column_index].strip()
        for row in data_rows
        if len(row) > column_index and row[column_index].strip()
    }

    if not transcript_ids:
        raise ValueError(f"No transcript IDs were loaded from {path!r}.")
    return transcript_ids


def zero_counts() -> Counter[str]:
    """Return an empty mutation counter."""
    return Counter({mutation_type: 0 for mutation_type in MUTATION_TYPES})


def write_count_row(writer: csv.DictWriter, gene: str, transcript: str, sample_id: str, counts: Counter[str]) -> None:
    """Write one gene + transcript + ID count row."""
    output_row = {"gene": gene, "transcript": transcript, "id": sample_id}
    output_row.update({mutation_type: counts.get(mutation_type, 0) for mutation_type in MUTATION_TYPES})
    writer.writerow(output_row)


def write_gene_transcript_id_counts(
    writer: csv.DictWriter,
    records: list[SequenceRecord],
    reference_policy: str,
    include_reference_ids: bool,
    skipped_non_atcg: Counter[str],
    stats: ProcessingStats,
    diagnostics_writer: Optional[csv.DictWriter] = None,
) -> None:
    """Count mutations and write one output row per gene + transcript + ID."""
    if not records:
        return

    groups: "OrderedDict[tuple[str, str], list[SequenceRecord]]" = OrderedDict()
    for rec in records:
        groups.setdefault((rec.gene, rec.transcript), []).append(rec)

    for (gene, transcript), group_records in groups.items():
        ref_index = choose_reference(group_records, reference_policy)
        ref = group_records[ref_index]
        id_counts: "OrderedDict[str, Counter[str]]" = OrderedDict()
        group_snv_count = 0
        group_skipped_before = sum(skipped_non_atcg.values())
        group_alt_id_assignments = 0

        stats.gene_transcript_groups_seen += 1

        if ref_index != 0:
            stats.references_not_first += 1
            print(
                f"WARNING: gene {gene!r}, transcript {transcript!r}: first row was not selected "
                f"as reference; row {ref.row_number} has the most IDs ({ref.id_count}).",
                file=sys.stderr,
            )

        for i, alt in enumerate(group_records):
            if i == ref_index:
                continue

            if not alt.ids:
                stats.alternate_rows_without_ids += 1
                continue

            row_counts: Counter[str] = Counter()
            row_snv_count = 0
            row_snv_count += add_snv_counts(
                row_counts,
                ref.tss,
                alt.tss,
                gene,
                transcript,
                alt.row_number,
                "TSS/promoter",
                skipped_non_atcg,
            )
            row_snv_count += add_snv_counts(
                row_counts,
                ref.tts,
                alt.tts,
                gene,
                transcript,
                alt.row_number,
                "TTS/terminator",
                skipped_non_atcg,
            )

            # Count the row's mutations once for each individual ID listed on
            # that alternate-sequence row.
            for sample_id in alt.ids:
                if sample_id not in id_counts:
                    id_counts[sample_id] = Counter()
                id_counts[sample_id].update(row_counts)
                group_alt_id_assignments += 1
                group_snv_count += row_snv_count

        stats.alternate_rows_seen += len(group_records) - 1
        stats.alternate_id_assignments_seen += group_alt_id_assignments

        if include_reference_ids:
            for sample_id in ref.ids:
                if sample_id not in id_counts:
                    id_counts[sample_id] = Counter()

        for sample_id, counts in id_counts.items():
            write_count_row(writer, gene, transcript, sample_id, counts)
            stats.output_rows_written += 1

        if diagnostics_writer is not None:
            diagnostics_writer.writerow(
                {
                    "gene": gene,
                    "transcript": transcript,
                    "reference_row": ref.row_number,
                    "reference_id_count": ref.id_count,
                    "rows_in_group": len(group_records),
                    "alternate_rows": len(group_records) - 1,
                    "alternate_id_assignments": group_alt_id_assignments,
                    "ids_written": len(id_counts),
                    "snv_count_across_ids": group_snv_count,
                    "non_atcg_mismatches": sum(skipped_non_atcg.values()) - group_skipped_before,
                }
            )


def process_file(args: argparse.Namespace) -> None:
    primary_transcripts = load_primary_transcripts(
        args.primary_transcripts,
        delimiter=args.primary_transcript_delimiter,
        id_column=args.primary_transcript_id_column,
    )
    print(f"Loaded {len(primary_transcripts)} primary transcript IDs from {args.primary_transcripts}.", file=sys.stderr)

    skipped_non_atcg = Counter()
    stats = ProcessingStats()
    matched_primary_transcripts: set[str] = set()

    with open_text(args.input) as in_fh, open(args.output, "w", newline="") as out_fh:
        dialect = detect_dialect(in_fh, args.input_delimiter)
        reader = csv.DictReader(in_fh, dialect=dialect)

        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"Input is missing required column(s): {', '.join(sorted(missing))}. "
                f"Observed columns: {fieldnames}"
            )

        writer = csv.DictWriter(
            out_fh,
            fieldnames=["gene", "transcript", "id", *MUTATION_TYPES],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        diagnostics_fh = None
        diagnostics_writer = None
        if args.diagnostics:
            diagnostics_fh = open(args.diagnostics, "w", newline="")
            diagnostics_writer = csv.DictWriter(
                diagnostics_fh,
                fieldnames=[
                    "gene",
                    "transcript",
                    "reference_row",
                    "reference_id_count",
                    "rows_in_group",
                    "alternate_rows",
                    "alternate_id_assignments",
                    "ids_written",
                    "snv_count_across_ids",
                    "non_atcg_mismatches",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            diagnostics_writer.writeheader()

        current_gene: Optional[str] = None
        records: list[SequenceRecord] = []

        try:
            for row_number, row in enumerate(reader, start=2):  # header is line 1
                stats.input_rows_seen += 1

                gene = row["gene"].strip()
                transcript = row["transcript"].strip()

                if transcript not in primary_transcripts:
                    stats.rows_skipped_by_transcript_filter += 1
                    continue

                stats.rows_kept_after_transcript_filter += 1
                matched_primary_transcripts.add(transcript)

                promoter = row["promoter"].strip().upper()
                terminator = row["terminator"].strip().upper()

                tss = tss_window(promoter, args.center)
                tts = tts_window(terminator, args.center)
                require_window_length(tss, args.window_length, gene, transcript, row_number, "promoter")
                require_window_length(tts, args.window_length, gene, transcript, row_number, "terminator")

                if current_gene is None:
                    current_gene = gene
                elif gene != current_gene:
                    write_gene_transcript_id_counts(
                        writer,
                        records,
                        args.reference_policy,
                        args.include_reference_ids,
                        skipped_non_atcg,
                        stats,
                        diagnostics_writer,
                    )
                    current_gene = gene
                    records = []

                records.append(
                    SequenceRecord(
                        gene=gene,
                        transcript=transcript,
                        row_number=row_number,
                        tss=tss,
                        tts=tts,
                        ids=parse_ids(row["group_for_cross_validation"]),
                    )
                )

            if records:
                write_gene_transcript_id_counts(
                    writer,
                    records,
                    args.reference_policy,
                    args.include_reference_ids,
                    skipped_non_atcg,
                    stats,
                    diagnostics_writer,
                )
        finally:
            if diagnostics_fh is not None:
                diagnostics_fh.close()

    stats.skipped_non_atcg = sum(skipped_non_atcg.values())
    unmatched_primary_transcripts = len(primary_transcripts - matched_primary_transcripts)

    print(
        f"Read {stats.input_rows_seen} rows; kept {stats.rows_kept_after_transcript_filter} rows "
        f"matching the primary transcript file and skipped {stats.rows_skipped_by_transcript_filter} rows.",
        file=sys.stderr,
    )
    print(
        f"Processed {stats.gene_transcript_groups_seen} gene+transcript groups, "
        f"{stats.alternate_rows_seen} alternate rows, and "
        f"{stats.alternate_id_assignments_seen} alternate ID assignments. Wrote "
        f"{stats.output_rows_written} rows to {args.output}.",
        file=sys.stderr,
    )
    if unmatched_primary_transcripts:
        print(
            f"{unmatched_primary_transcripts} loaded primary transcript IDs were not present in the input file.",
            file=sys.stderr,
        )
    if stats.references_not_first:
        print(
            f"Reference was not the first row in {stats.references_not_first} comparison groups; "
            "the row with the most IDs was used.",
            file=sys.stderr,
        )
    if stats.alternate_rows_without_ids:
        print(f"Skipped {stats.alternate_rows_without_ids} alternate rows without IDs.", file=sys.stderr)
    if stats.skipped_non_atcg:
        print(
            f"Skipped {stats.skipped_non_atcg} positions whose substitution type could not be resolved "
            "(N, unknown characters, or ambiguous IUPAC combinations such as both sides heterozygous "
            "or alternate base absent from the reference IUPAC alleles).",
            file=sys.stderr,
        )
    if args.diagnostics:
        print(f"Wrote diagnostics to {args.diagnostics}.", file=sys.stderr)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count A/T/C/G SNV classes per individual ID in truncated TSS/TTS windows for listed primary transcripts."
    )
    parser.add_argument("input", help="Input CSV/TSV file. Gzip is supported with .gz files.")
    parser.add_argument("output", help="Output TSV file with one row per included gene+transcript+ID.")
    parser.add_argument(
        "--primary-transcripts",
        required=True,
        help="File containing primary transcript IDs to include. Gzip is supported with .gz files.",
    )
    parser.add_argument(
        "--primary-transcript-id-column",
        default="auto",
        help="Column in --primary-transcripts containing transcript IDs. Default: auto; uses a transcript header if present, else first column.",
    )
    parser.add_argument(
        "--primary-transcript-delimiter",
        choices=["auto", "tab", "comma"],
        default="auto",
        help="Delimiter for --primary-transcripts. Default: auto-detect comma vs tab.",
    )
    parser.add_argument(
        "--input-delimiter",
        choices=["auto", "tab", "comma"],
        default="auto",
        help="Input delimiter. Default: auto-detect comma vs tab.",
    )
    parser.add_argument(
        "--reference-policy",
        choices=["most-ids", "first"],
        default="most-ids",
        help="How to choose the reference row within each gene+transcript group. Default: most-ids.",
    )
    parser.add_argument(
        "--include-reference-ids",
        action="store_true",
        help="Also write zero-count rows for IDs on the selected reference row. Default: omit reference IDs.",
    )
    parser.add_argument(
        "--center",
        type=int,
        default=5000,
        help="0-based center boundary of each 10000 bp sequence. Default: 5000.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=5000,
        help="Expected length after truncation. Default: 5000.",
    )
    parser.add_argument(
        "--diagnostics",
        default=None,
        help="Optional TSV path for per-gene+transcript diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        process_file(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())