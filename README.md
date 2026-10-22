# Koze

A transcription tool for Mauritian Creole (Kreol Morisien) audio. *Koze* is Kreol for "talk".

## Status

- **Web app** (`web/`): tap to talk, get a Kreol transcript. Meta's Omnilingual ASR CTC 300M model runs in the browser, on WebGPU when available and otherwise on WebAssembly. The model downloads once (about 350 MB) and is cached. Audio never leaves the device. It is deployed to GitHub Pages by `.github/workflows/pages.yml`.
- **Model harness** (`harness/`): a Python/Gradio app that compares every Omnilingual ASR model and size, scores against reference transcripts, and logs runs. It runs on Colab or a Mac.

### How the web model is made

`tools/export_onnx.py` converts the CTC model to ONNX and makes 8-bit copies. `tools/verify_onnx.py` checks them against Meta's PyTorch pipeline on synthesised Kreol speech: the fp32 graph must match it, and the 8-bit file that ships must change at most 3% of characters. `tools/build_site.py` splits the chosen file into 40 MB parts for Pages. CI runs all three on every PR and deploys on `main`. The check results appear in the job summary.

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
