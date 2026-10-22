# Kreol ASR harness

A small web app for trying Meta's Omnilingual ASR on Kreol Morisien. You can record or upload a clip, run several model sizes side by side, paste what was actually said to get WER and CER, and keep a log of every run in CSV with the audio.

## Option A: Colab (free GPU, any computer or phone)

1. Upload `kreol_asr_harness_colab.ipynb` to Google Drive, then right-click it and choose **Open with → Google Colaboratory**. You can also go to colab.research.google.com and use **File → Upload notebook**.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Choose **Runtime → Run all**, then open the `https://….gradio.live` link printed at the bottom. The link is public for 72 hours, so don't share it.

## Option B: your Mac (Apple Silicon) or a Linux box

```bash
cd harness
./run_local.sh            # first run installs into .venv, then opens http://127.0.0.1:7860
./run_local.sh --device cpu   # if the Apple GPU path errors
./run_local.sh --mock         # UI only, no models (instant)
```

On a Mac, the script installs Homebrew's `python@3.11` and `libsndfile` if they're missing. Use Python 3.11: Meta's package declares `<=3.12`, which pip reads as excluding 3.12.1 and later.

## Which models to try

| Family | Sizes | What it is |
|---|---|---|
| CTC | 300M, 1B, 3B, 7B | Fast, maps audio straight to letters, ignores the language hint |
| LLM | 300M, 1B, 3B, 7B | Adds a text decoder that uses the `mfe_Latn` hint. Slower, usually more accurate |
| LLM Unlimited | 300M, 1B, 3B, 7B | Same as LLM with no 40 s limit, so long audio isn't split |

These are rough limits, estimated from model size and not measured:

- **Free Colab T4 (15 GB):** everything up to 3B.
- **M3 Max with 64 GB:** all sizes fit.
- **MacBook Air with 16 GB:** 300M and 1B. 3B is tight.

Start with **CTC 300M + LLM 1B**.

## What the app does

- Non-Unlimited models only accept clips under 40 s, so longer audio is split automatically at the quietest point.
- **Simulate phone line** turns the clip into 8 kHz G.711 audio, so you can see how much accuracy a phone call would cost. You can listen to the result.
- **Batch** tab: drop in several voice notes. You can add references as `filename|transcript`, one per line.
- **Log** tab: every run is saved to `runs/runs.csv` and `runs/audio/`. On Colab it goes to `MyDrive/kreol-asr-runs` if Drive is mounted.

## Caveats

- WER here uses only light normalisation: lowercase, punctuation stripped, apostrophes unified. Kreol spelling variants count as errors. For real numbers, use the eval repo's normalisation tables.
- The Apple GPU (MPS) path is untested. If a model fails there, the app retries it on the CPU automatically.
- Built against `omnilingual-asr==0.2.0` (Python 3.10 or 3.11).

## Files

`app.py` is the UI. `asr_core.py` holds the audio prep, splitting, phone simulation, scoring and model cache. `test_core.py` holds the tests (`pip install pytest && pytest`). `kreol_asr_harness_colab.ipynb` is a self-contained copy of both files for Colab.
