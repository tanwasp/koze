# Koze

A transcription tool for Mauritian Creole (Kreol Morisien) audio. *Koze* is Kreol for "talk".

## Status

- **Web app** (`web/`): tap to talk, get a Kreol transcript. Meta's Omnilingual ASR CTC models run in the browser, on WebGPU when available and otherwise on WebAssembly. There are two sizes: 300M (about 350 MB, hosted with the site) and 1B (about 1 GB). GitHub Pages caps a site at 1 GB, so the 1B file is attached to the repo's `models` GitHub Release. Browsers can't fetch release files from another site (no CORS headers), so you download the file once and choose it in the page. The page recommends a size from what the browser reveals and a speed check, downloads the model once and caches it. Audio never leaves the device. It is deployed to GitHub Pages by `.github/workflows/pages.yml`.
- **Model harness** (`harness/`): a Python/Gradio app that compares every Omnilingual ASR model and size, scores against reference transcripts, and logs runs. It runs on Colab or a Mac.

### How the web model is made

`tools/export_onnx.py` converts a CTC model to ONNX and makes 8-bit copies. `tools/verify_onnx.py` checks them against Meta's PyTorch pipeline on synthesised Kreol speech: the fp32 graph must match it, and the 8-bit file that ships must change at most 3% of characters. `tools/build_site.py pack` splits the chosen file into 40 MB parts with a SHA-256 for each, and `build_site.py site` assembles the site. CI runs all of this on every PR and deploys on `main`. The check results appear in the job summary.

The 1B model has its own job, which converts and checks it the same way and attaches it to the `models` release using the workflow's built-in token, so no extra credentials are needed. If the repository has an `HF_REPO` variable (for example `tanwasp/koze-models`) and an `HF_TOKEN` secret with write access to it, the job uploads to Hugging Face instead with `tools/publish_hf.py`, and the page downloads 1B directly with no file step.

### Choosing a size

Browsers reveal little about hardware on purpose. `navigator.deviceMemory` is rounded and capped at 8 GB, and Safari and Firefox don't report it. WebGPU names the GPU family but not its memory. So the page uses the hardware only to rule the large model out (phone or tablet, no WebGPU, under 8 GB reported). After a model loads it times it on 6 seconds of audio, and it recommends 1B only if 1B would transcribe a minute of audio in under about 15 seconds. 1B's speed is estimated from the measured model scaled by file size.

### Security

- `models.json` is served from the site and pins a SHA-256 for every model part and the vocabulary. The page checks each part as it downloads, before using or caching it, so neither the model host nor a file someone picks can swap in a different model.
- No third-party code at runtime. onnxruntime-web and the fonts are copied into the site at build time from npm tarballs checked against pinned integrity hashes (`VENDOR` in `tools/build_site.py`).
- A Content-Security-Policy allows only the site's own scripts, styles and fonts, and network access only to the site and Hugging Face. The policy is in `web/index.html`.
- GitHub Actions are pinned to commit SHAs.
- There is no login. Anyone with the URL can use the page, but that costs nothing because everything runs on their own device. Recordings and history stay in the browser.

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
