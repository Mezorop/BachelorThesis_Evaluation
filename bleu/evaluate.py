#!/usr/bin/env python3
"""Add sentence-level text BLEU scores to output_results.tsv."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate text BLEU in-place.")
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        required = ("translated_text", "expected_translation_of_original_audio")
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
    for name in ("bleu", "bleu_status", "bleu_error"):
        if name not in fields:
            fields.append(name)

    try:
        from sacrebleu.metrics import BLEU
    except ImportError as error:
        raise RuntimeError(
            "SacreBLEU is missing; install bleu/requirements.txt in its own environment."
        ) from error

    metric = BLEU(effective_order=True)
    selected = rows if args.limit is None else rows[: args.limit]
    for index, row in enumerate(selected, start=1):
        if not args.force and row.get("bleu_status") == "ok":
            continue
        try:
            row["bleu"] = float(
                metric.sentence_score(
                    row["translated_text"],
                    [row["expected_translation_of_original_audio"]],
                ).score
            )
            row["bleu_status"] = "ok"
            row["bleu_error"] = ""
        except Exception as error:
            row["bleu_status"] = "error"
            row["bleu_error"] = f"{type(error).__name__}: {error}"
        print(f"[{index}/{len(selected)}] BLEU={row.get('bleu', '')}")
    save(path, fields, rows)


if __name__ == "__main__":
    main()
