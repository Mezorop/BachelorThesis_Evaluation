from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate COMET and ASR-COMET in-place.")
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=1)
    return parser.parse_args()

#Loads the TSV file
#Check out whisperx_timing/evaluate.py for a more detailed code break down.
def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        required = (
            "original_transcription",
            "translated_text",
            "asr_transcription",
            "expected_translation_of_original_audio",
        )
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError("Missing TSV columns: " + ", ".join(missing))
        return fields, list(reader)

#Creates a temporary copy of the old tsv file with the additional new COMET scores. If the process passes then the old tsv gets replaced.
#Check out whisperx_timing/evaluate.py for a more detailed code break down.
def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
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
    fields, rows = read_tsv(path)
    for name in (
        "comet",
        "comet_status",
        "comet_error",
        "comet_and_asr_comet",
        "asr_comet_status",
        "asr_comet_error",
    ):
        if name not in fields:
            fields.append(name)

    targets: list[tuple[int, str, str, str]] = []
    data: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if row.get("comet_status") != "ok":
            translated_text = row.get("translated_text", "").strip()
            if not translated_text:
                row["comet_status"] = "error"
                row["comet_error"] = "translated_text is empty."
            else:
                targets.append((index, "comet", "comet_status", "comet_error"))
                data.append(
                    {
                        "src": row["original_transcription"],
                        "mt": translated_text,
                        "ref": row["expected_translation_of_original_audio"],
                    }
                )

        if row.get("asr_comet_status") != "ok":
            asr_transcription = row.get("asr_transcription", "").strip()
            if row.get("whisperx_status") != "ok" or not asr_transcription:
                row["asr_comet_status"] = "error"
                row["asr_comet_error"] = (
                    "WhisperX stage is not complete for this row."
                )
            else:
                targets.append(
                    (index, "comet_and_asr_comet", "asr_comet_status", "asr_comet_error")
                )
                data.append(
                    {
                        "src": row["original_transcription"],
                        "mt": asr_transcription,
                        "ref": row["expected_translation_of_original_audio"],
                    }
                )

    if data:
        from comet import download_model, load_from_checkpoint

        checkpoint = download_model(args.model)
        model = load_from_checkpoint(checkpoint)
        prediction: Any = model.predict(
            data,
            batch_size=args.batch_size,
            gpus=args.gpus,
            progress_bar=True,
        )
        scores = prediction.scores if hasattr(prediction, "scores") else prediction[0]
        if len(scores) != len(targets):
            raise RuntimeError("COMET returned a different number of scores than inputs.")
        for target, score in zip(targets, scores, strict=True):
            row_index, score_column, status_column, error_column = target
            rows[row_index][score_column] = str(float(score))
            rows[row_index][status_column] = "ok"
            rows[row_index][error_column] = ""

    write_tsv(path, fields, rows)
    comet_count = sum(score_column == "comet" for _, score_column, _, _ in targets)
    asr_comet_count = sum(
        score_column == "comet_and_asr_comet" for _, score_column, _, _ in targets
    )
    print(
        f"COMET complete: {comet_count} text rows and "
        f"{asr_comet_count} ASR rows scored with {args.model}"
    )


if __name__ == "__main__":
    main()
