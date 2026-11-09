# BachelorThesis Evaluation

Dieses Repository ergänzt die von den Speech-to-Speech-Modellen erzeugte
`output_results.tsv` schrittweise. Jede Evaluationsstufe besitzt genau eine
Python-Datei und ein eigenes virtuelles Environment. Dadurch müssen WhisperX,
SacreBLEU und COMET nicht auf einen künstlich gemeinsamen Abhängigkeitsstand
gepinnt werden.

## Pipeline

```text
output_results.tsv
  -> WhisperX + TTFA/End Offset
  -> Text-BLEU
  -> ASR-BLEU
  -> ASR-COMET
  -> vollständig ergänzte output_results.tsv
```

| Stufe | Python-Datei | Environment | Ergänzte Hauptspalten |
|---|---|---|---|
| WhisperX/Timing | `whisperx_timing/evaluate.py` | `whisperx` | `asr_transcription`, `ttfa`, `end_offset` und Wortgrenzen |
| BLEU | `bleu/evaluate.py` | `bleu` | `bleu` |
| ASR-BLEU | `asr_bleu/evaluate.py` | `asr-bleu` | `asr_bleu` |
| ASR-COMET | `asr_comet/evaluate.py` | `asr-comet` | `asr_comet` |

Alle bestehenden Spalten bleiben erhalten. Jede Stufe schreibt atomar zurück
in dieselbe TSV. Die teure WhisperX-Stufe speichert nach jeder Zeile, überspringt
bereits erfolgreiche Zeilen und kann nach einem abgebrochenen Job fortgesetzt
werden. Zusätzliche Status- und Fehlerspalten machen unvollständige Zeilen
sichtbar.

## Definition der Zeitmetriken

WhisperX richtet das bekannte deutsche Quelltranskript an der Quellaufnahme aus.
Für die übersetzte Aufnahme wird zuerst ASR ausgeführt und anschließend das
erkannte englische Transkript auf Wortebene ausgerichtet. Damit beziehen sich
Anfang und Ende auf tatsächlich erkannte Zielsprache in der Audiodatei.

Die Zeitstempel beider WAV-Dateien beginnen bei null. Hibikis
`output_audio_start_delay` verschiebt die Zielaufnahme auf die Zeitachse des
Quellstreams:

```text
TTFA = output_audio_start_delay
       + target_first_word_start_in_audio
       - source_first_word_start

End Offset = output_audio_start_delay
             + target_last_word_end_in_audio
             - source_last_word_end
```

## Offizielle Implementierungen

- WhisperX folgt dem offiziellen Ablauf `load_model` -> `transcribe` ->
  `load_align_model` -> `align` aus dem
  [WhisperX-Repository](https://github.com/m-bain/whisperX#python-usage).
- BLEU und ASR-BLEU verwenden `sacrebleu.metrics.BLEU` aus dem
  [offiziellen SacreBLEU-Projekt](https://github.com/mjpost/sacreBLEU). Die
  TSV-Werte sind Satz-BLEU-Werte auf der SacreBLEU-Skala von 0 bis 100;
  `effective_order=True` verhindert irreführende Nullwerte bei sehr kurzen
  Sätzen.
- ASR-COMET verwendet `download_model`, `load_from_checkpoint` und `predict`
  entsprechend der
  [offiziellen COMET-Python-API](https://github.com/Unbabel/COMET#scoring-within-python).
  `src` ist das deutsche Quelltranskript, `mt` das WhisperX-Transkript der
  übersetzten Audiodatei und `ref` die englische Referenzübersetzung.

## Installation auf VSC-5

Nach dem Laden derselben Python- und FFmpeg-Module wie im SLURM-Job:

```bash
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6

cd "$DATA/BachelorThesis-Evaluation"
./setup_environments.sh
```

Standardmäßig entstehen die Environments unter
`$DATA/venvs/bachelor-thesis-evaluation/`. Ein anderer Zielordner kann als
Argument an `setup_environments.sh` übergeben oder über
`EVALUATION_VENV_ROOT` gesetzt werden. Für eine standortspezifische
PyTorch/CUDA-Installation können die Pakete im WhisperX- und COMET-Environment
vor dem Lauf angepasst werden, ohne die übrigen Stufen zu beeinflussen.

## Proof of Concept und vollständiger Lauf

Zuerst zehn Zeilen prüfen:

```bash
export EVALUATION_LIMIT=10
./run_pipeline.sh "$DATA/unchecked_model_results/hibiki_zero/output_results.tsv"
```

Danach `EVALUATION_LIMIT` entfernen. Bereits erfolgreich ausgewertete Zeilen
werden wiederverwendet:

```bash
unset EVALUATION_LIMIT
sbatch evaluation_pipeline.slurm
```

Wichtige optionale Umgebungsvariablen sind `WHISPERX_MODEL`,
`WHISPERX_BATCH_SIZE`, `WHISPERX_COMPUTE_TYPE`, `COMET_MODEL`,
`COMET_BATCH_SIZE`, `SOURCE_LANGUAGE`, `TARGET_LANGUAGE` und
`EVALUATION_DEVICE`.

## Erwartete Eingangsspalten

Die von `hibiki_zero_script.py` erzeugte TSV enthält bereits:

- `path_to_original_audio`
- `original_transcription`
- `expected_translation_of_original_audio`
- `translated_text`
- `path_to_translated_audio`
- `output_audio_start_delay`
- die zunächst leeren Ergebnisfelder `ttfa`, `end_offset`, `bleu`,
  `asr_bleu` und `asr_comet`

Die Audiopfade müssen auf dem Rechenknoten gültig sein. WhisperX und COMET
laden ihre Modellgewichte beim ersten Lauf; dafür müssen die Caches gefüllt
oder vom Rechenknoten aus erreichbar sein.
