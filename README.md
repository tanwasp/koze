# Koze

A transcription tool for Mauritian Creole (Kreol Morisien) audio. *Koze* is Kreol for "talk".

## Status

- **Web app** (`web/`): tap to talk, get a Kreol transcript. Meta's Omnilingual ASR CTC models run in the browser, on WebGPU when available and otherwise on WebAssembly. There are two sizes: 300M (about 350 MB, hosted with the site) and 1B (about 1 GB, hosted on Hugging Face because GitHub Pages caps a site at 1 GB). The page recommends a size from what the browser reveals and a speed check, downloads the model once and caches it. Audio never leaves the device. It is deployed to GitHub Pages by `.github/workflows/pages.yml`.
- **Model harness** (`harness/`): a Python/Gradio app that compares every Omnilingual ASR model and size, scores against reference transcripts, and logs runs. It runs on Colab or a Mac.

### Collecting Kreol speech

Anyone can use the transcriber without an account. After a transcription, a signed-out visitor sees an invitation to help. A signed-in contributor sees a form to play the clip back, correct the transcript, rate it, tag what went wrong and which languages were spoken, and share it.

- **Sign-in** is a passwordless email code (Supabase Auth). There are no passwords to store, leak or reset, and the same code creates the account.
- **Consent** is recorded on the account with its version and time. It covers being 18+, what is sent and why, an opt-in for open release, and optional speaker details. Uploads also need "everyone who speaks agreed". Each clip keeps a copy of the terms it was shared under.
- **Roles:** everyone starts as `contributor`. Set `role` to `collector` (a volunteer or paid transcriber) or `admin` (can read every clip) in the `profiles` table in the Supabase dashboard. Nobody can change their own role.
- **Account page:** contributors see their clips, minutes and corrections, can play or delete any clip, and can delete the whole account.

Each clip stores:
- the 16 kHz mono WAV the model heard;
- the model's output, untouched;
- the corrected text, exactly as typed;
- whether the contributor aimed for Grafi Larmoni spelling;
- the feedback;
- the model and backend.

The page holds only the project's public key, and `supabase/schema.sql` scopes every request to the signed-in person with row-level security. `tools/export_dataset.py` runs on your own machine with the service-role key. It:
- downloads everything as a Hugging Face audiofolder dataset, with a hashed contributor id per clip;
- optionally pushes it;
- writes a per-contributor summary for paying collectors;
- deletes orphaned audio.

To turn it on:
1. Create a Supabase project (free tier) and run `supabase/schema.sql` in its SQL editor.
2. Authentication > Emails: set up custom SMTP (for example Resend's free tier). Supabase's built-in email only reaches your own team's addresses, a few emails an hour.
3. Authentication > Emails > Templates > Magic link: make the body show the code, for example `<p>Your Koze code is <b>{{ .Token }}</b></p>`, so people get a code instead of a link.
4. In this repo, go to Settings > Secrets and variables > Actions > Variables. Set `SUPABASE_URL` (like `https://abcd.supabase.co`) and `SUPABASE_KEY`. Use the publishable or anon key, never the secret/service-role key; the build refuses those.
5. Re-run the Pages workflow. Without these variables the site works as before, with no sign-in.

The free tier holds about 1 GB of audio, roughly 8 hours. It also pauses a project after a week with no requests; unpause it from the dashboard and nothing is lost. Export to Hugging Face before it fills.

### Choosing a size

Browsers reveal little about hardware on purpose. `navigator.deviceMemory` is rounded and capped at 8 GB, and Safari and Firefox don't report it. WebGPU names the GPU family but not its memory. So the page uses the hardware only to rule the large model out (phone or tablet, no WebGPU, under 8 GB reported). After a model loads it times it on 6 seconds of audio, and it recommends 1B only if 1B would transcribe a minute of audio in under about 15 seconds. 1B's speed is estimated from the measured model scaled by file size.

### Security

- `models.json` is served from the site and pins a SHA-256 for every model part and the vocabulary. The page checks each part as it downloads, before using or caching it, so the model host (Hugging Face) can't swap in a different file.
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
- Optional English translation
- Export as plain text or SRT/VTT subtitles

## Open questions

- Where it runs: a local Python CLI first, or a web app with a GPU backend?
- How much labelled Kreol audio can be gathered?
