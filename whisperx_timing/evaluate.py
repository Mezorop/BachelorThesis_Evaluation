#!/usr/bin/env python3
"""Add WhisperX ASR, word timestamps, TTFA, and end offset to a result TSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "path_to_original_audio",
    "original_transcription",
    "path_to_translated_audio",
    "output_audio_start_delay",
)
OUTPUT_COLUMNS = (
    "asr_transcription",
    "source_first_word_start",
    "source_last_word_end",
    "target_first_word_start_in_audio",
    "target_last_word_end_in_audio",
    "ttfa",
    "end_offset",
    "whisperx_status",
    "whisperx_error",
)


@dataclass(frozen=True)
class WordTiming:
    start: float
    end: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WhisperX and calculate TTFA/end offset in-place."
    )
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--source-language", default="de")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError("Missing TSV columns: " + ", ".join(missing))
        return list(reader.fieldnames), list(reader)


def write_tsv_atomic(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
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


def words_from_alignment(result: dict[str, Any]) -> list[WordTiming]:
    raw_words = result.get("word_segments")
    if raw_words is None:
        raw_words = [
            word
            for segment in result.get("segments", [])
            for word in segment.get("words", [])
        ]
    words = [
        WordTiming(float(word["start"]), float(word["end"]))
        for word in raw_words
        if word.get("start") is not None and word.get("end") is not None
    ]
    return sorted(words, key=lambda word: (word.start, word.end))


class WhisperXEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import whisperx
        except ImportError as error:
            raise RuntimeError(
                "WhisperX is missing; install whisperx_timing/requirements.txt "
                "in its dedicated environment."
            ) from error

        self.whisperx = whisperx
        self.args = args
        self.aligners: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.asr_model = whisperx.load_model(
            args.model,
            args.device,
            compute_type=args.compute_type,
            language=args.target_language,
            download_root=str(args.model_dir) if args.model_dir else None,
        )

    def aligner(self, language: str) -> tuple[Any, dict[str, Any]]:
        if language not in self.aligners:
            self.aligners[language] = self.whisperx.load_align_model(
                language_code=language,
                device=self.args.device,
                model_dir=str(self.args.model_dir) if self.args.model_dir else None,
            )
        return self.aligners[language]

    def align_segments(
        self, audio: Any, segments: list[dict[str, Any]], language: str
    ) -> list[WordTiming]:
        model, metadata = self.aligner(language)
        aligned = self.whisperx.align(
            segments,
            model,
            metadata,
            audio,
            self.args.device,
            return_char_alignments=False,
        )
        words = words_from_alignment(aligned)
        if not words:
            raise ValueError(f"WhisperX aligned no words for language {language!r}.")
        return words

    def align_reference(self, audio_path: Path, text: str, language: str) -> list[WordTiming]:
        if not text.strip():
            raise ValueError("Source transcription is empty.")
        audio = self.whisperx.load_audio(str(audio_path))
        duration = len(audio) / 16_000
        return self.align_segments(
            audio,
            [{"start": 0.0, "end": duration, "text": text}],
            language,
        )

    def transcribe_target(self, audio_path: Path) -> tuple[str, list[WordTiming]]:
        audio = self.whisperx.load_audio(str(audio_path))
        result = self.asr_model.transcribe(audio, batch_size=self.args.batch_size)
        segments = list(result.get("segments", []))
        text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        )
        if not text:
            raise ValueError("WhisperX transcribed no target speech.")
        language = str(result.get("language") or self.args.target_language)
        return text, self.align_segments(audio, segments, language)


def main() -> int:
    args = parse_args()
    path = args.tsv.expanduser().resolve()
    fieldnames, rows = read_tsv(path)
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    evaluator = WhisperXEvaluator(args)
    selected_count = len(rows) if args.limit is None else min(args.limit, len(rows))
    failures = 0

    for index, row in enumerate(rows[:selected_count], start=1):
        if not args.force and row.get("whisperx_status") == "ok":
            print(f"[{index}/{selected_count}] already complete")
            continue
        try:
            source_audio = Path(row["path_to_original_audio"]).expanduser()
            target_audio = Path(row["path_to_translated_audio"]).expanduser()
            if not source_audio.is_file():
                raise FileNotFoundError(f"Source audio not found: {source_audio}")
            if not target_audio.is_file():
                raise FileNotFoundError(f"Target audio not found: {target_audio}")

            source_words = evaluator.align_reference(
                source_audio, row["original_transcription"], args.source_language
            )
            asr_text, target_words = evaluator.transcribe_target(target_audio)
            delay = float(row["output_audio_start_delay"])
            if delay < 0:
                raise ValueError("output_audio_start_delay must not be negative.")

            ttfa = delay + target_words[0].start - source_words[0].start
            end_offset = delay + target_words[-1].end - source_words[-1].end
            row.update(
                {
                    "asr_transcription": asr_text,
                    "source_first_word_start": source_words[0].start,
                    "source_last_word_end": source_words[-1].end,
                    "target_first_word_start_in_audio": target_words[0].start,
                    "target_last_word_end_in_audio": target_words[-1].end,
                    "ttfa": ttfa,
                    "end_offset": end_offset,
                    "whisperx_status": "ok",
                    "whisperx_error": "",
                }
            )
            print(f"[{index}/{selected_count}] TTFA={ttfa:.3f}, end={end_offset:.3f}")
        except Exception as error:
            failures += 1
            row["whisperx_status"] = "error"
            row["whisperx_error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{selected_count}] ERROR: {row['whisperx_error']}")
        write_tsv_atomic(path, fieldnames, rows)

    print(f"WhisperX timing complete: {selected_count - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
