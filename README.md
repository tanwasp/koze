# Koze

A transcription tool for Mauritian Creole (Kreol Morisien) audio. *Koze* is Kreol for "talk".

## Status

Planning. Nothing is implemented yet.

## Approach (draft)

Mauritian Creole is a low-resource language, so off-the-shelf speech recognition is weak on it. Options to evaluate:

1. **Baseline:** run a multilingual speech-to-text model (e.g. Whisper, Meta MMS) and measure word error rate on a small set of hand-transcribed Kreol clips. Whisper has no Kreol option, so try French as the source language; MMS may support Kreol directly.
2. **Fine-tune:** fine-tune Whisper or MMS on Kreol audio/transcript pairs. Possible data sources: hand-labelled recordings, radio/TV with subtitles, Bible recordings with text, and corrections made in this tool.
3. **Post-process:** normalize output to the standard orthography (*Lortograf Kreol Morisien*, 2011), since raw output will tend toward French spelling.

## Planned features

- Upload an audio file, get a timestamped transcript
- Edit the transcript in the browser; save corrections as training data
- Optional English translation
- Export as plain text or SRT/VTT subtitles

## Open questions

- Where it runs: a local Python CLI first, or a web app with a GPU backend?
- How much labelled Kreol audio can be gathered?
