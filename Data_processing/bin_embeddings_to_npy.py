#!/usr/bin/env python3
"""Convert compressed training embedding TSV files into NumPy arrays.

The input ``*.bin`` files produced by ``sequence2embedding.a2z.py`` and
``sequence2embedding.caduceus.py`` are gzip-compressed, tab-separated text.
The a2z writer currently emits an incorrect six-column header followed by
eight-column rows, so this converter deliberately ignores the header and
parses data rows by their known positions.

Metadata is joined by gene from the accompanying
``*.training.groups.<species>.tsv`` file.  The group file is authoritative for
family, TPM, and cross-validation group.

Embedding output is channel-first by default, matching the training scripts:

    tss/tts:            (N, channels, positions)
    a2z predictions:    (N, prediction_channels, positions)

Every embedding and prediction array is stored as float32.  Output files are
named, for example::

    a2z.training.data.Ath.gene.npy
    a2z.training.data.Ath.family.npy
    a2z.training.data.Ath.TPM.npy
    a2z.training.data.Ath.group.npy
    a2z.training.data.Ath.tss.npy
    a2z.training.data.Ath.tts.npy
    a2z.training.data.Ath.tss_predictions.npy
    a2z.training.data.Ath.tts_predictions.npy

Run this only after the producer has closed the corresponding gzip file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib.format import open_memmap


DATA_MARKER = ".training.data."


@dataclass(frozen=True)
class GroupMetadata:
    family: str
    tpm: np.float32
    group: str


@dataclass(frozen=True)
class InputDescription:
    rows: int
    fields: int
    stored_shape: tuple[int, ...]
    max_gene_length: int
    max_family_length: int
    max_group_length: int

    @property
    def is_a2z(self) -> bool:
        return self.fields == 8


def derive_group_path(bin_path: Path, groups_dir: Path | None) -> Path:
    """Derive the matching training-groups filename from a data filename."""
    if DATA_MARKER not in bin_path.name or not bin_path.name.endswith(".bin"):
        raise ValueError(
            f"Unexpected input name {bin_path.name!r}; expected "
            "*.training.data.<species>.bin"
        )

    group_name = bin_path.name.replace(DATA_MARKER, ".training.groups.", 1)
    group_name = group_name.removesuffix(".bin") + ".tsv"
    return (groups_dir if groups_dir is not None else bin_path.parent) / group_name


def load_group_metadata(path: Path) -> dict[str, GroupMetadata]:
    """Load and validate gene metadata from a training-groups TSV file."""
    metadata: dict[str, GroupMetadata] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene", "family", "TPM", "group"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            gene = row["gene"]
            if not gene:
                raise ValueError(f"{path}:{line_number}: empty gene")
            value = GroupMetadata(
                family=row["family"],
                tpm=np.float32(row["TPM"]),
                group=row["group"],
            )
            previous = metadata.get(gene)
            if previous is not None and previous != value:
                raise ValueError(
                    f"{path}:{line_number}: conflicting metadata for gene {gene!r}"
                )
            metadata[gene] = value

    if not metadata:
        raise ValueError(f"No metadata rows found in {path}")
    return metadata


def iter_data_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield nonempty data lines after the producer-written header."""
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"{path} is empty")
        for line_number, line in enumerate(handle, start=2):
            line = line.rstrip("\r\n")
            if line:
                yield line_number, line


def inspect_input(
    bin_path: Path,
    group_metadata: dict[str, GroupMetadata],
) -> InputDescription:
    """Count rows and establish array/string dimensions without loading JSON."""
    rows = 0
    expected_fields: int | None = None
    expected_shape: tuple[int, ...] | None = None
    max_gene_length = 1
    max_family_length = max(len(value.family) for value in group_metadata.values())
    max_group_length = max(len(value.group) for value in group_metadata.values())

    for line_number, line in iter_data_lines(bin_path):
        # Only split through the dimensions field during this inexpensive pass.
        prefix = line.split("\t", 4)
        if len(prefix) != 5:
            raise ValueError(f"{bin_path}:{line_number}: fewer than five fields")
        gene = prefix[0]
        metadata = group_metadata.get(gene)
        if metadata is None:
            raise ValueError(
                f"{bin_path}:{line_number}: gene {gene!r} is absent from the group file"
            )

        fields = line.count("\t") + 1
        if fields not in (6, 8):
            raise ValueError(
                f"{bin_path}:{line_number}: expected 6 Caduceus fields or "
                f"8 a2z fields, found {fields}"
            )
        if expected_fields is None:
            expected_fields = fields
        elif fields != expected_fields:
            raise ValueError(
                f"{bin_path}:{line_number}: field count changed from "
                f"{expected_fields} to {fields}"
            )

        shape_value = json.loads(prefix[3])
        shape = tuple(int(item) for item in shape_value)
        if len(shape) != 2 or any(item <= 0 for item in shape):
            raise ValueError(
                f"{bin_path}:{line_number}: invalid embedding shape {shape!r}"
            )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"{bin_path}:{line_number}: embedding shape changed from "
                f"{expected_shape} to {shape}"
            )

        rows += 1
        max_gene_length = max(max_gene_length, len(gene))

    if rows == 0 or expected_fields is None or expected_shape is None:
        raise ValueError(f"No embedding rows found in {bin_path}")

    return InputDescription(
        rows=rows,
        fields=expected_fields,
        stored_shape=expected_shape,
        max_gene_length=max_gene_length,
        max_family_length=max(1, max_family_length),
        max_group_length=max(1, max_group_length),
    )


def output_shape(
    rows: int,
    stored_shape: tuple[int, int],
    channel_first: bool,
) -> tuple[int, int, int]:
    sample_shape = stored_shape[::-1] if channel_first else stored_shape
    return (rows, *sample_shape)


def parse_float_array(
    value: str,
    shape: tuple[int, ...],
    *,
    path: Path,
    line_number: int,
    field_name: str,
) -> np.ndarray:
    """Parse one JSON numeric array directly into float32."""
    array = np.asarray(json.loads(value), dtype=np.float32)
    expected_size = int(np.prod(shape))
    if array.size != expected_size:
        raise ValueError(
            f"{path}:{line_number}: {field_name} contains {array.size} values; "
            f"expected {expected_size} for shape {shape}"
        )
    return array.reshape(shape)


def make_output_paths(bin_path: Path, output_dir: Path, is_a2z: bool) -> dict[str, Path]:
    prefix = bin_path.name.removesuffix(".bin")
    fields = ["gene", "family", "TPM", "group", "tss", "tts"]
    if is_a2z:
        fields.extend(["tss_predictions", "tts_predictions"])
    return {field: output_dir / f"{prefix}.{field}.npy" for field in fields}


def convert_file(
    bin_path: Path,
    *,
    groups_dir: Path | None,
    output_dir: Path | None,
    channel_first: bool,
    overwrite: bool,
) -> list[Path]:
    """Convert one gzip training-data file and return its output paths."""
    bin_path = bin_path.resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)

    group_path = derive_group_path(bin_path, groups_dir)
    if not group_path.is_file():
        raise FileNotFoundError(
            f"Matching group file not found for {bin_path}: {group_path}"
        )

    destination = (output_dir if output_dir is not None else bin_path.parent).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    metadata = load_group_metadata(group_path)
    description = inspect_input(bin_path, metadata)
    paths = make_output_paths(bin_path, destination, description.is_a2z)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist; use --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    print(
        f"Converting {bin_path.name}: {description.rows} rows, "
        f"{'a2z' if description.is_a2z else 'Caduceus'}, "
        f"stored shape {description.stored_shape}"
    )

    partial_paths = {
        name: path.with_name(f".{path.name}.partial") for name, path in paths.items()
    }
    arrays: dict[str, np.memmap] = {}
    completed = False
    try:
        arrays["gene"] = open_memmap(
            partial_paths["gene"], mode="w+", dtype=f"U{description.max_gene_length}",
            shape=(description.rows,),
        )
        arrays["family"] = open_memmap(
            partial_paths["family"], mode="w+", dtype=f"U{description.max_family_length}",
            shape=(description.rows,),
        )
        arrays["TPM"] = open_memmap(
            partial_paths["TPM"], mode="w+", dtype=np.float32,
            shape=(description.rows,),
        )
        arrays["group"] = open_memmap(
            partial_paths["group"], mode="w+", dtype=f"U{description.max_group_length}",
            shape=(description.rows,),
        )
        embedding_shape = output_shape(
            description.rows, description.stored_shape, channel_first
        )
        arrays["tss"] = open_memmap(
            partial_paths["tss"], mode="w+", dtype=np.float32,
            shape=embedding_shape,
        )
        arrays["tts"] = open_memmap(
            partial_paths["tts"], mode="w+", dtype=np.float32,
            shape=embedding_shape,
        )

        # Prediction shape is established from the first a2z row and allocated lazily.
        prediction_shape: tuple[int, int] | None = None

        row_index = 0
        for line_number, line in iter_data_lines(bin_path):
            fields = line.split("\t")
            if len(fields) != description.fields:
                raise ValueError(
                    f"{bin_path}:{line_number}: expected {description.fields} fields, "
                    f"found {len(fields)}"
                )

            gene = fields[0]
            joined = metadata[gene]
            arrays["gene"][row_index] = gene
            arrays["family"][row_index] = joined.family
            arrays["TPM"][row_index] = joined.tpm
            arrays["group"][row_index] = joined.group

            tss = parse_float_array(
                fields[4], description.stored_shape, path=bin_path,
                line_number=line_number, field_name="tss",
            )
            tts = parse_float_array(
                fields[5], description.stored_shape, path=bin_path,
                line_number=line_number, field_name="tts",
            )
            arrays["tss"][row_index] = tss.T if channel_first else tss
            arrays["tts"][row_index] = tts.T if channel_first else tts

            if description.is_a2z:
                positions = description.stored_shape[0]
                for name, field_index in (
                    ("tss_predictions", 6),
                    ("tts_predictions", 7),
                ):
                    flat = np.asarray(json.loads(fields[field_index]), dtype=np.float32)
                    if flat.size % positions != 0:
                        raise ValueError(
                            f"{bin_path}:{line_number}: {name} has {flat.size} values, "
                            f"which is not divisible by {positions} positions"
                        )
                    stored_prediction_shape = (positions, flat.size // positions)
                    sample_prediction_shape = (
                        stored_prediction_shape[::-1]
                        if channel_first else stored_prediction_shape
                    )
                    if prediction_shape is None:
                        prediction_shape = sample_prediction_shape
                        for prediction_name in (
                            "tss_predictions", "tts_predictions"
                        ):
                            arrays[prediction_name] = open_memmap(
                                partial_paths[prediction_name], mode="w+",
                                dtype=np.float32,
                                shape=(description.rows, *prediction_shape),
                            )
                    elif sample_prediction_shape != prediction_shape:
                        raise ValueError(
                            f"{bin_path}:{line_number}: prediction shape changed from "
                            f"{prediction_shape} to {sample_prediction_shape}"
                        )

                    prediction = flat.reshape(stored_prediction_shape)
                    arrays[name][row_index] = (
                        prediction.T if channel_first else prediction
                    )

            row_index += 1
            if row_index % 1000 == 0:
                print(f"  parsed {row_index}/{description.rows} rows", flush=True)

        if row_index != description.rows:
            raise RuntimeError(
                f"Input changed between passes: expected {description.rows} rows, "
                f"read {row_index}"
            )

        for array in arrays.values():
            array.flush()
        arrays.clear()

        for name, final_path in paths.items():
            os.replace(partial_paths[name], final_path)
        completed = True
    finally:
        arrays.clear()
        if not completed:
            for partial_path in partial_paths.values():
                partial_path.unlink(missing_ok=True)

    print(f"Wrote {len(paths)} arrays to {destination}")
    return list(paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert gzip-compressed *.training.data.<species>.bin files to "
            "joined NumPy arrays."
        )
    )
    parser.add_argument(
        "bin_files", nargs="+", type=Path,
        help="One or more *.training.data.<species>.bin files",
    )
    parser.add_argument(
        "--groups-dir", type=Path,
        help="Directory containing matching *.training.groups.<species>.tsv files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: directory of each input file)",
    )
    parser.add_argument(
        "--position-first", action="store_true",
        help="Preserve stored (positions, channels) order instead of channel-first",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing output .npy files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for bin_file in args.bin_files:
        convert_file(
            bin_file,
            groups_dir=args.groups_dir,
            output_dir=args.output_dir,
            channel_first=not args.position_first,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
