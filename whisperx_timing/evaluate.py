from __future__ import annotations

import argparse
import csv
import os
import tempfile
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
    "source_first_word_start", #TTFA calculation
    "source_last_word_end", #End Offset calculation
    "target_first_word_start_in_audio", #TTFA calculation
    "target_last_word_end_in_audio", #End Offset calculation
    "ttfa",
    "end_offset",
    "whisperx_status",
    "whisperx_error", #Debugging
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WhisperX and calculate TTFA/end offset in-place."
    )
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--source-language", default="de")
    parser.add_argument("--target-language", default="en")
    #The Whisper Model to use
    #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/asr.py#L342 last visit: 11.11.2026
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        #DictReader leaves out the fieldnames (1st row) https://docs.python.org/3/library/csv.html#csv.DictReader last visit: 09.11.26
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        #Checks if all headers in REQUIRED_COLUMNS are present
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError("Missing TSV columns: " + ", ".join(missing))

        #Returns a list with the headers and a list of the instances
        return list(reader.fieldnames), list(reader)


def write_tsv_atomic(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    #Creates a temporary file that doesn't interfere with the existing tsv file till it's fully build up
    #"prefix" sets the beginning of the file name. "suffix" the end of the file name.
    #https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp last visit: 09.11.2026
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:

        #Creates a file object named "handle" out of the temporary file
        #https://docs.python.org/3/library/os.html#os.fdopen last visit: 09.11.2026
        #https://docs.python.org/3/builtins/functions.html#open last visit: 09.11.2026
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            #buffer gets cleared
            #https://docs.python.org/3.14/library/io.html#io.IOBase.flush last visit: 09.11.2026
            handle.flush()
            #os.fsync(fd) Force writes the file fd (file descriptor) onto the disk.
            #https://docs.python.org/3.14/library/os.html#os.fsync last visit: 09.11.2026
            #Therefore the file descriptor of the stream is given via handle.fileno() .
            #https://docs.python.org/3.14/library/io.html#io.IOBase.fileno last visit: 09.11.2026
            os.fsync(handle.fileno())
        #Replaces the old model result file with the extended (TTFA, End-Offset, ASR transcription) result + evaluation file.
        #https://docs.python.org/3.14/library/os.html#os.replace last visit: 09.11.2026
        os.replace(temporary_name, path)
    except BaseException:
        try:
            #Removes the file path
            #https://docs.python.org/3.14/library/os.html#os.unlink last visit: 11.11.2026
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def timed_word_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    #Let's "SingleWordSegment"'s of "word_segments" pass that include start and end attributes
    #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/alignment.py#L414 last visit: 11.11.2026
    #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/schema.py#L11-L18 last visit: 11.11.2026
    words = [
        word
        for word in result["word_segments"]
        if word.get("start") is not None and word.get("end") is not None
    ]
    return words


class WhisperXEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        import whisperx

        self.whisperx = whisperx
        self.args = args
        #Prepares the datastructure for storing the two alignment models.
        #dict[key(type "str"), value(being a tuple of the WhisperX model and an embedded dictionary with model metadata)]
        #http://docs.python.org/3.14/builtins/stdtypes.html#dict last visit: 12.11.2026
        self.aligners: dict[str, tuple[Any, dict[str, Any]]] = {}
        #Loads the WhisperModel version given in "args.model" (see run_pipeline.sh)
        #https://github.com/m-bain/whisperX/blob/b904f02b7e1f26fd10fb04840d655ff27e88e464/whisperx/asr.py#L474 last visit 12.11.2026
        self.asr_model = whisperx.load_model(
            args.model,
            args.device,
            compute_type=args.compute_type,
            language=args.target_language,
        )

    def aligner(self, language: str) -> tuple[Any, dict[str, Any]]:
        if language not in self.aligners:
            #Loads the alignment model of the specific language if not yet loaded in the aligners dictionary.
            #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/alignment.py#L80 last visit 12.11.2026
            self.aligners[language] = self.whisperx.load_align_model(
                language_code=language,
                device=self.args.device,
            )
        return self.aligners[language]

    def align_segments(
        self, audio: Any, segments: list[dict[str, Any]], language: str
    ) -> list[dict[str, Any]]:
        model, metadata = self.aligner(language)
        #Timestamps the words in the audio with the help of the transcript
        #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/alignment.py#L117 last visit: 12.11.2026
        aligned = self.whisperx.align(
            segments,
            model,
            metadata,
            audio,
            self.args.device,
            return_char_alignments=False,
        )
        words = timed_word_segments(aligned)
        if not words:
            raise ValueError(f"WhisperX aligned no words for language {language!r}.")
        return words

    #Timestamping the original audio with the help of the original transcript
    def align_reference(
        self, audio_path: Path, text: str, language: str
    ) -> list[dict[str, Any]]:
        #load_audio resamples any non 16kHz based dataset sample rate to 16kHz via FFMPEG
        #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/audio.py#L25-L60 last visit: 13.11.2026
        audio = self.whisperx.load_audio(str(audio_path))
        #All samples/sample rate => duration of audio in seconds
        duration = len(audio) / 16_000
        return self.align_segments(
            audio,
            [{"start": 0.0, "end": duration, "text": text}],
            language,
        )

    #Transcribing and timestamping the translated audio
    def transcribe_target(
        self, audio_path: Path
    ) -> tuple[str, list[dict[str, Any]]]:
        audio = self.whisperx.load_audio(str(audio_path))
        #.transcribe(...) returns "TranscriptionResult" which is a list of "SingleSegments" which consist of start, end, text and avg_logprobe
        #https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/asr.py#L197-L209 last visit: 12.11.2026
        #https://github.com/m-bain/whisperX/blob/main/whisperx/schema.py#L65 last visit: 12.11.2026
        #https://github.com/m-bain/whisperX/blob/main/whisperx/schema.py#L30 last visit: 12.11.2026
        result = self.asr_model.transcribe(audio, batch_size=self.args.batch_size)
        segments = list(result.get("segments", []))
        #Concatenates words with a " " separator
        #https://docs.python.org/3.14/builtins/stdtypes.html#str.join last visit: 12.11.2026
        text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        )
        if not text:
            raise ValueError("WhisperX transcribed no target speech.")
        language = str(result["language"])
        return text, self.align_segments(audio, segments, language)


def main() -> int:
    args = parse_args()
    path = args.tsv.expanduser().resolve()
    fieldnames, rows = read_tsv(path)
    #Appends new evaluation columns to the tsv file.
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    evaluator = WhisperXEvaluator(args)
    row_count = len(rows)
    failures = 0

    for index, row in enumerate(rows, start=1):
        #Leaves out already evaluated translations
        if row.get("whisperx_status") == "ok":
            print(f"[{index}/{row_count}] already complete")
            continue
        try:
            source_audio = Path(row["path_to_original_audio"]).expanduser()
            target_audio = Path(row["path_to_translated_audio"]).expanduser()
            if not source_audio.is_file():
                raise FileNotFoundError(f"Source audio not found: {source_audio}")
            if not target_audio.is_file():
                raise FileNotFoundError(f"Target audio not found: {target_audio}")

            #Timestamps the original audio
            source_words = evaluator.align_reference(
                source_audio, row["original_transcription"], args.source_language
            )

            #Transcribes and timestamps the translated audio
            asr_text, target_words = evaluator.transcribe_target(target_audio)
            delay = float(row["output_audio_start_delay"])
            if delay < 0:
                raise ValueError("output_audio_start_delay must not be negative.")

            source_first_start = float(source_words[0]["start"])
            source_last_end = float(source_words[-1]["end"])
            target_first_start = float(target_words[0]["start"])
            target_last_end = float(target_words[-1]["end"])

            ttfa = delay + target_first_start - source_first_start
            end_offset = delay + target_last_end - source_last_end
            row.update(
                {
                    "asr_transcription": asr_text,
                    "source_first_word_start": source_first_start,
                    "source_last_word_end": source_last_end,
                    "target_first_word_start_in_audio": target_first_start,
                    "target_last_word_end_in_audio": target_last_end,
                    "ttfa": ttfa,
                    "end_offset": end_offset,
                    "whisperx_status": "ok",
                    "whisperx_error": "",
                }
            )
            print(f"[{index}/{row_count}] TTFA={ttfa:.3f}, end={end_offset:.3f}")
        except Exception as error:
            failures += 1
            row["whisperx_status"] = "error"
            row["whisperx_error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{row_count}] ERROR: {row['whisperx_error']}")
        write_tsv_atomic(path, fieldnames, rows)

    print(f"WhisperX timing complete: {row_count - failures} ok, {failures} failed")

    return 1 if failures else 0


if __name__ == "__main__":
    #main() returns either 1 "abnormal termination" or 0 "successful termination"
    #https://docs.python.org/3.14/builtins/exceptions.html#SystemExit last visit: 11.11.2026
    #Exception is raised by sys.exit() - where 0 and any nonzero value is defined
    #https://docs.python.org/3.14/library/sys.html#sys.exit last visit: 11.11.2026
    raise SystemExit(main())
