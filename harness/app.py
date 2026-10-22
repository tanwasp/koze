"""Kreol ASR web harness: record or upload audio, run any omnilingual-asr model
side by side, score against a reference, and log every run.

    python app.py            # real models (GPU strongly recommended)
    python app.py --share    # also print a public gradio.live link
    python app.py --mock     # fake models, to try the UI without a GPU
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from asr_core import DEFAULT_MODELS, LANGS, MODELS, SR, ModelBank, score, simulate_phone, to_mono_16k

COLUMNS = ["model", "transcript", "WER", "CER", "secs", "RTF", "chunks", "note"]
LOG_FIELDS = ["time", "clip", "source", "duration_s", "phone_sim", "lang", "model", "transcript",
              "reference", "wer", "cer", "secs", "load_secs", "chunks", "error", "notes"]


def device_info(device: str) -> str:
    if device == "cuda":
        import torch

        p = torch.cuda.get_device_properties(0)
        return f"Running on GPU: {p.name}, {p.total_memory / 2**30:.0f} GB."
    if device == "mps":
        return "Running on Apple GPU (MPS), falling back to CPU per model if an op is unsupported."
    return "Running on CPU: start with the 300M models."


def build(bank: ModelBank, log_dir: Path) -> gr.Blocks:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "audio").mkdir(exist_ok=True)
    log_path = log_dir / "runs.csv"

    def log_rows(rows: list[dict]) -> None:
        new = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)

    def run_one(wav, clip_name, source, models, lang, phone, reference, notes):
        dur = len(wav) / SR
        if phone:
            wav = simulate_phone(wav)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        saved = log_dir / "audio" / f"{stamp}_{Path(clip_name).stem}{'_phone' if phone else ''}.wav"
        sf.write(saved, wav, SR)
        table, logs = [], []
        for label in models:
            r = bank.transcribe(label, wav, LANGS[lang])
            s = score(reference, r.text) if reference.strip() and not r.error else {}
            note = r.error or (f"first load {r.load_seconds:.0f}s" if r.load_seconds >= 1 else "")
            table.append([label, r.text, _pct(s.get("wer")), _pct(s.get("cer")), round(r.seconds, 2),
                          round(r.seconds / dur, 3) if dur else None, r.chunks, note])
            logs.append({"time": stamp, "clip": saved.name, "source": source, "duration_s": round(dur, 2),
                         "phone_sim": phone, "lang": LANGS[lang], "model": MODELS[label], "transcript": r.text,
                         "reference": reference, "wer": s.get("wer"), "cer": s.get("cer"),
                         "secs": round(r.seconds, 3), "load_secs": round(r.load_seconds, 1),
                         "chunks": r.chunks, "error": r.error, "notes": notes})
        log_rows(logs)
        return table, dur, saved

    def transcribe(audio, models, lang, phone, reference, notes):
        if audio is None:
            raise gr.Error("Record or upload a clip first.")
        if not models:
            raise gr.Error("Tick at least one model.")
        sr, data = audio
        wav = to_mono_16k(sr, data)
        if len(wav) < SR * 0.3:
            raise gr.Error("Clip is shorter than 0.3 s.")
        table, dur, saved = run_one(wav, "clip", "single", models, lang, phone, reference, notes)
        summary = f"{dur:.1f}s clip, saved as `{saved.name}`. {device_info(bank.device)}"
        if phone:
            return table, summary, (SR, wav_for_playback(saved))
        return table, summary, None

    def transcribe_batch(files, models, lang, phone, refs_text):
        if not files:
            raise gr.Error("Upload one or more audio files.")
        if not models:
            raise gr.Error("Tick at least one model.")
        refs = parse_refs(refs_text)
        rows = []
        for f in files:
            path = Path(f if isinstance(f, str) else f.name)
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            wav = to_mono_16k(sr, data)
            ref = refs.get(path.name, refs.get(path.stem, ""))
            table, _, _ = run_one(wav, path.name, "batch", models, lang, phone, ref, "")
            rows += [[path.name] + r for r in table]
        return rows, str(log_path)

    with gr.Blocks(title="Kreol ASR harness") as demo:
        gr.Markdown(
            "## Kreol ASR harness\n"
            "Test Meta's Omnilingual ASR on Kreol Morisien. Pick models, record or upload audio, "
            "and compare. Paste what was actually said to get WER/CER. Every run is logged.  \n"
            f"_{device_info(bank.device)} The first run of each model downloads and loads it, which can take a few minutes._"
        )
        with gr.Row():
            models = gr.CheckboxGroup(list(MODELS), value=DEFAULT_MODELS, label="Models",
                                      info="CTC: fast, no language hint. LLM: uses the hint below. "
                                           "Unlimited: no 40 s cap. 7B needs a big GPU.")
        with gr.Row():
            lang = gr.Dropdown(list(LANGS), value=list(LANGS)[0], label="Language hint (LLM models only)")
            phone = gr.Checkbox(False, label="Simulate phone line (8 kHz G.711)",
                                info="Degrades the clip like a phone call before transcribing.")

        with gr.Tab("Single clip"):
            audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip (any length)")
            reference = gr.Textbox(label="Reference transcript (optional)", lines=2,
                                   placeholder="What was actually said, in Kreol. Leave empty to skip scoring.")
            notes = gr.Textbox(label="Notes (optional, saved to log)", placeholder="speaker, setting, code-switching…")
            go = gr.Button("Transcribe", variant="primary")
            summary = gr.Markdown()
            out = gr.Dataframe(headers=COLUMNS, wrap=True, interactive=False)
            phone_audio = gr.Audio(label="What the model heard (phone-simulated)", interactive=False)
            go.click(transcribe, [audio, models, lang, phone, reference, notes], [out, summary, phone_audio])

        with gr.Tab("Batch"):
            files = gr.File(file_count="multiple", file_types=["audio"], label="Audio files")
            refs = gr.Textbox(label="References (optional)", lines=4,
                              placeholder="one per line:  filename<TAB or |>transcript")
            go_b = gr.Button("Transcribe all", variant="primary")
            out_b = gr.Dataframe(headers=["file"] + COLUMNS, wrap=True, interactive=False)
            log_file = gr.File(label="Full run log (CSV)")
            go_b.click(transcribe_batch, [files, models, lang, phone, refs], [out_b, log_file])

        with gr.Tab("Log"):
            gr.Markdown(f"Runs are appended to `{log_path}`; clips are saved in `{log_dir / 'audio'}`.")
            show = gr.Button("Refresh")
            log_table = gr.Dataframe(wrap=True, interactive=False)
            log_dl = gr.File(label="Download log")
            show.click(lambda: (read_log(log_path), str(log_path) if log_path.exists() else None),
                       None, [log_table, log_dl])

        with gr.Row():
            unload = gr.Button("Unload models (free GPU memory)", size="sm")
            status = gr.Markdown()
            unload.click(lambda: (bank.unload_all(), "Unloaded.")[1], None, status)
    return demo


def wav_for_playback(path: Path) -> np.ndarray:
    data, _ = sf.read(path, dtype="float32")
    return data


def parse_refs(text: str) -> dict[str, str]:
    refs = {}
    for line in (text or "").splitlines():
        for sep in ("\t", "|"):
            if sep in line:
                name, ref = line.split(sep, 1)
                refs[name.strip()] = ref.strip()
                break
    return refs


def read_log(path: Path) -> list[list]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[-200:] if len(rows) <= 1 else [rows[0]] + rows[1:][-200:]


def _pct(x):
    return None if x is None else f"{100 * x:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="fake models, for trying the UI")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--log-dir", default=os.environ.get("KREOL_LOG_DIR", "runs"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--dtype", default="auto", help="auto, bf16, fp16 or fp32")
    a = p.parse_args()
    demo = build(ModelBank(mock=a.mock, device=a.device, dtype=a.dtype), Path(a.log_dir))
    demo.queue(default_concurrency_limit=1).launch(share=a.share, inbrowser=not a.share, server_name="127.0.0.1" if not a.share else "0.0.0.0", server_port=a.port,
                                                   allowed_paths=[str(Path(a.log_dir).resolve())])


if __name__ == "__main__":
    main()
