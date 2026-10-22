"""Check the exported ONNX models against Meta's own PyTorch pipeline on Kreol speech,
and pick which file the web app ships.

    python tools/verify_onnx.py --model-dir build/model --report build/report.json

Test audio is synthesised with facebook/mms-tts-mfe (Kreol TTS; fine for testing,
its licence is non-commercial), plus noisy copies. The reference for each clip is
what the PyTorch pipeline transcribes, so this measures conversion loss only, not
how good the model is at Kreol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_onnx import ctc_greedy  # noqa: E402

SENTENCES = [
    "Bonzour, ki manyer ou ete zordi?",
    "Mo pe al lakaz mo mama apre travay.",
    "Ki ler bis la pou pase dan vilaz la?",
    "Mo bizin pey mo bil elektrisite avan lafin lemwa.",
    "Zot ti dir mwa ki lapli pou tonbe dime gramatin.",
    "Eski ou kapav rapel mwa pli tar, mersi.",
    "Nou pe rod enn plas pou manze dan Port Louis.",
    "Sa bann zanfan la pe zwe foutborl lor plaz.",
]
# Ship an 8-bit file only if it changes at most this share of characters vs PyTorch.
MAX_CER = 0.03


def tts_clips() -> list[tuple[str, np.ndarray]]:
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-mfe")
    tts = VitsModel.from_pretrained("facebook/mms-tts-mfe").eval()
    assert tts.config.sampling_rate == 16000
    rng = np.random.default_rng(0)
    clips = []
    for i, text in enumerate(SENTENCES):
        torch.manual_seed(i)
        with torch.no_grad():
            wav = tts(**tok(text, return_tensors="pt")).waveform[0].numpy().astype(np.float32)
        clips.append((f"tts{i}", wav))
        noisy = wav + rng.normal(0, 0.02, wav.shape).astype(np.float32)
        clips.append((f"tts{i}+noise", noisy))
    return clips


def cer(ref: str, hyp: str) -> float:
    import jiwer

    return 0.0 if ref == hyp else (jiwer.cer(ref, hyp) if ref else 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card", default="omniASR_CTC_300M_v2")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    d = Path(a.model_dir)

    import onnxruntime as ort
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=a.card, device="cpu", dtype=torch.float32)
    vocab = json.loads((d / "vocab.json").read_text())
    variants = [n for n in ("model.onnx", "model.int8.onnx", "model.int8pc.onnx") if (d / n).exists()]
    sessions = {n: ort.InferenceSession(str(d / n)) for n in variants}

    rows = []
    for name, wav in tts_clips():
        ref = pipe.transcribe([{"waveform": wav, "sample_rate": 16000}], batch_size=1)[0]
        row = {"clip": name, "seconds": round(len(wav) / 16000, 1), "pytorch": ref}
        for n, sess in sessions.items():
            row[n] = ctc_greedy(sess.run(None, {"waveform": wav[None, :]})[0], vocab)
        rows.append(row)

    summary = {}
    for n in variants:
        summary[n] = {
            "exact": sum(r[n] == r["pytorch"] for r in rows) / len(rows),
            "cer_vs_pytorch": float(np.mean([cer(r["pytorch"], r[n]) for r in rows])),
        }
    # The fp32 graph must reproduce PyTorch, otherwise the export or the decoder is wrong.
    ok32 = summary["model.onnx"]["cer_vs_pytorch"] <= 0.005
    ship = next((n for n in ("model.int8.onnx", "model.int8pc.onnx")
                 if summary.get(n, {}).get("cer_vs_pytorch", 1) <= MAX_CER), None)
    report = {"card": a.card, "summary": summary, "ship": ship, "fp32_matches": ok32, "rows": rows}
    Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))

    md = [f"### ONNX conversion check: {a.card}", "", "| file | exact match | CER vs PyTorch |", "|---|---|---|"]
    md += [f"| {n} | {s['exact']:.0%} | {s['cer_vs_pytorch']:.2%} |" for n, s in summary.items()]
    md += ["", f"Shipping: **{ship}**", "", "| clip | PyTorch | " + " | ".join(variants) + " |",
           "|---|---|" + "---|" * len(variants)]
    md += [f"| {r['clip']} | {r['pytorch']} | " + " | ".join(r[n] for n in variants) + " |" for r in rows]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if not ok32 or ship is None:
        sys.exit("Conversion check failed: see the table above.")


if __name__ == "__main__":
    main()
