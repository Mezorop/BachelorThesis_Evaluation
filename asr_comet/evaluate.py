#!/usr/bin/env python3
"""Add reference-based ASR-COMET scores to output_results.tsv."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate ASR-COMET in-place.")
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        required = (
            "original_transcription",
            "asr_transcription",
            "expected_translation_of_original_audio",
        )
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError("Missing TSV columns: " + ", ".join(missing))
        return fields, list(reader)


def save(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = arguments()
    path = args.tsv.expanduser().resolve()
    fields, rows = load(path)
    for name in ("asr_comet", "asr_comet_status", "asr_comet_error"):
        if name not in fields:
            fields.append(name)

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        raise RuntimeError(
            "COMET is missing; install asr_comet/requirements.txt in its own environment."
        ) from error

    selected = rows if args.limit is None else rows[: args.limit]
    indices: list[int] = []
    data: list[dict[str, str]] = []
    for index, row in enumerate(selected):
        if not args.force and row.get("asr_comet_status") == "ok":
            continue
        if row.get("whisperx_status") != "ok" or not row.get("asr_transcription", "").strip():
            row["asr_comet_status"] = "error"
            row["asr_comet_error"] = "WhisperX stage is not complete for this row."
            continue
        indices.append(index)
        data.append(
            {
                "src": row["original_transcription"],
                "mt": row["asr_transcription"],
                "ref": row["expected_translation_of_original_audio"],
            }
        )

    if data:
        checkpoint = download_model(args.model)
        model = load_from_checkpoint(checkpoint)
        prediction: Any = model.predict(
            data,
            batch_size=args.batch_size,
            gpus=args.gpus,
            progress_bar=True,
        )
        scores = prediction.scores if hasattr(prediction, "scores") else prediction[0]
        if len(scores) != len(indices):
            raise RuntimeError("COMET returned a different number of scores than inputs.")
        for row_index, score in zip(indices, scores, strict=True):
            selected[row_index]["asr_comet"] = float(score)
            selected[row_index]["asr_comet_status"] = "ok"
            selected[row_index]["asr_comet_error"] = ""

    save(path, fields, rows)
    print(f"ASR-COMET complete: {len(indices)} rows scored with {args.model}")


if __name__ == "__main__":
    main()
