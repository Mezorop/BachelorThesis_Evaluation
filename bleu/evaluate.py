from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate text BLEU in-place.")
    parser.add_argument("tsv", type=Path)
    return parser.parse_args()

#Loads the TSV file
#Check out whisperx_timing/evaluate.py for a more detailed code break down.
def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        required = ("translated_text", "expected_translation_of_original_audio")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError("Missing TSV columns: " + ", ".join(missing))
        return fields, list(reader)

#Creates a temporary copy of the old tsv file with the additional new BLEU scores. If the process passes then the old tsv gets replaced.
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
    #Adds the columns into the tsv that store the BLEU evaluation results
    for name in ("bleu", "bleu_status", "bleu_error"):
        if name not in fields:
            fields.append(name)

    from sacrebleu.metrics import BLEU

    #"metric" becomes an instance of the class "BLEU" with the configuration effective_order=True which is recommended
    #for sentence-level BLEU
    #https://docs.python.org/3.14/tutorial/classes.html#class-objects last visit: 13.11.2026
    #https://github.com/mjpost/sacrebleu/blob/v2.6.0/sacrebleu/metrics/bleu.py#L121-L122 last visit: 13.11.2026
    metric = BLEU(effective_order=True)
    for index, row in enumerate(rows, start=1):
        if row.get("bleu_status") == "ok":
            continue
        try:
            #metric.sentence_score computes the BLEU score
            #https://github.com/mjpost/sacrebleu/blob/v2.6.0/sacrebleu/metrics/base.py#L404-L416 last visit: 13.11.2026
            row["bleu"] = str(
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

        print(f"[{index}/{len(rows)}] BLEU={row.get('bleu', '')}")
    write_tsv(path, fields, rows)


if __name__ == "__main__":
    main()
