"""Export Meta's MMS Kreol Morisien text-to-speech model (facebook/mms-tts-mfe, VITS,
about 36M parameters) to ONNX for the web app, and check the export.

    python tools/export_tts.py --out build/tts --asr-dir build/model
    python tools/export_tts.py --random --out /tmp/tts   # random weights, no download

Writes:
  tts.onnx                 input_ids [1, L] int64, noise_scale, noise_scale_duration,
                           speaking_rate (float32 scalars) -> waveform [1, N] (16 kHz)
  tts.json                 tokenizer table and defaults, read by web/tts.js
  tokenizer-cases.json     texts and the ids the Hugging Face tokenizer gives them;
                           tools/check_tts_tokenizer.mjs checks web/tts.js against these
  report.json              the checks below

Checks: with both noise scales at 0 the ONNX graph must reproduce PyTorch, and (with
--asr-dir) the shipped speech recognition model must understand what it says.

The model weights are CC-BY-NC 4.0 (non-commercial), unlike the ASR model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = "facebook/mms-tts-mfe"
SENTENCES = [
    "Bonzour, kouma ou ete? Mo bien kontan trouv ou.",
    "Zordi lapli pe tonbe depi gramatin, me aswar pou ena soley.",
    "Mo pe al bazar Port-Louis pou aste legim ek pwason.",
    "Eski to kapav ed mwa ranz sa program-la avan Lindi?",
    "Lekol finn ferme akoz siklonn, tou zanfan res lakaz.",
    "Kreol Morisien se enn lalang ki tou Morisien konpran.",
]
# Text the tokenizer must handle the same way in Python and JavaScript.
TOKENIZER_CASES = SENTENCES + [
    "ÉTÉ, Été, été! L'enn ek lot.",
    "Mo ena 3 zanfan ek 12 rupi",
    "  espas   an   plis  ",
    "Tab\tek\nnouvo lign",
    "Zot dir: «ki ler?» — “mwa” çava œuf naïf",
    "",
]
# Round trip through the speech recogniser: share of characters it may get wrong.
MAX_ROUNDTRIP_CER = 0.25


class TtsForExport(nn.Module):
    """VITS with the sampling knobs as graph inputs instead of baked-in constants, so the
    page can choose the speed, and the check can turn the randomness off."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, noise_scale, noise_scale_duration, speaking_rate):
        m = self.model
        m.noise_scale, m.noise_scale_duration, m.speaking_rate = noise_scale, noise_scale_duration, speaking_rate
        return m(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).waveform


def load(random: bool):
    from transformers import VitsConfig, VitsModel, VitsTokenizer

    if random:
        # Same architecture as the MMS TTS models, with a made-up Latin vocabulary.
        vocab = {"<pad>": 0} | {c: i + 1 for i, c in enumerate(" '-abcdefghijklmnopqrstuvwxyz")}
        d = Path(os.environ.get("TMPDIR", "/tmp")) / "koze-tts-random-vocab"
        d.mkdir(parents=True, exist_ok=True)
        (d / "vocab.json").write_text(json.dumps(vocab))
        tok = VitsTokenizer(str(d / "vocab.json"), add_blank=True, normalize=True, phonemize=False, is_uroman=False)
        torch.manual_seed(0)
        model = VitsModel(VitsConfig(vocab_size=len(vocab), sampling_rate=16000))
        return model.eval(), tok
    from transformers import AutoTokenizer

    return VitsModel.from_pretrained(REPO).eval(), AutoTokenizer.from_pretrained(REPO)


def tokenizer_table(tok, cfg) -> dict:
    if getattr(tok, "phonemize", False) or getattr(tok, "is_uroman", False):
        raise SystemExit("This tokenizer needs phonemes or uroman, which web/tts.js does not implement")
    vocab = tok.get_vocab()
    return {
        "model": REPO,
        "licence": "CC-BY-NC-4.0",
        "sampling_rate": cfg.sampling_rate,
        "vocab": vocab,
        "blank_id": 0,
        "add_blank": bool(tok.add_blank),
        "normalize": bool(tok.normalize),
        "noise_scale": float(cfg.noise_scale),
        "noise_scale_duration": float(cfg.noise_scale_duration),
        "speaking_rate": float(cfg.speaking_rate),
    }


def fold_accents(tok, text: str) -> str:
    """What web/tts.js adds before tokenising: a lowercase letter the model has no token
    for is replaced by its unaccented form."""
    vocab = tok.get_vocab()
    return "".join(c if c in vocab else unicodedata.normalize("NFKD", c) for c in tok.normalize_text(text)) \
        if tok.normalize else text


def scalars(noise: float, noise_dur: float, rate: float) -> dict:
    f = lambda x: np.array(x, dtype=np.float32)  # noqa: E731
    return {"noise_scale": f(noise), "noise_scale_duration": f(noise_dur), "speaking_rate": f(rate)}


def export(model, path: Path) -> None:
    ids = torch.tensor([[0, 5, 0, 9, 0, 3, 0]], dtype=torch.long)
    args = (ids,) + tuple(torch.tensor(x, dtype=torch.float32) for x in (0.667, 0.8, 1.0))
    torch.onnx.export(
        TtsForExport(model).eval(), args, str(path), dynamo=False, opset_version=17,
        input_names=["input_ids", "noise_scale", "noise_scale_duration", "speaking_rate"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}},
    )


def asr(asr_dir: Path, report: Path | None):
    """The speech recogniser file the site ships, as a function from audio to text."""
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).parent))
    from export_onnx import ctc_greedy

    name = json.loads(report.read_text())["ship"] if report else "model.onnx"
    sess = ort.InferenceSession(str(asr_dir / name))
    vocab = json.loads((asr_dir / "vocab.json").read_text())
    return lambda wav: ctc_greedy(sess.run(None, {"waveform": wav[None, :].astype(np.float32)})[0], vocab)


def plain(text: str) -> str:
    """Lowercase, no accents or punctuation: what both sides of the round trip are compared on."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c if c.isalnum() else " " for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--random", action="store_true", help="random weights, to test the export without downloading")
    p.add_argument("--asr-dir", help="a converted ASR model dir; checks the speech is understood")
    p.add_argument("--asr-report", help="verify_onnx.py's report for --asr-dir, naming the file the site ships")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load(a.random)
    table = tokenizer_table(tok, model.config)
    (out / "tts.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    cases = [{"text": t, "ids": tok(fold_accents(tok, t))["input_ids"]} for t in TOKENIZER_CASES]
    (out / "tokenizer-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))

    path = out / "tts.onnx"
    export(model, path)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    hear = asr(Path(a.asr_dir), Path(a.asr_report) if a.asr_report else None) if a.asr_dir else None
    rows = []
    for text in SENTENCES:
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            ref = TtsForExport(model).eval()(ids, *(torch.tensor(x) for x in (0.0, 0.0, 1.0)))[0].numpy()
        got = sess.run(None, {"input_ids": ids.numpy()} | scalars(0, 0, 1))[0][0]
        same_len = len(got) == len(ref)
        diff = float(np.abs(got - ref).max()) if same_len else float("inf")
        # With the default randomness, as the page plays it.
        sampled = sess.run(None, {"input_ids": ids.numpy()} | scalars(
            table["noise_scale"], table["noise_scale_duration"], table["speaking_rate"]))[0][0]
        slow = sess.run(None, {"input_ids": ids.numpy()} | scalars(0, 0, 0.75))[0][0]
        row = {"text": text, "seconds": round(len(sampled) / table["sampling_rate"], 2), "same_length": same_len,
               "max_abs_diff": diff, "slow_ratio": round(len(slow) / max(1, len(got)), 2)}
        if hear:
            import jiwer

            heard = hear(sampled)
            row |= {"heard": heard, "cer": float(jiwer.cer(plain(text), plain(heard)))}
        rows.append(row)

    ok_match = all(r["same_length"] and r["max_abs_diff"] < 1e-3 for r in rows)
    ok_rate = all(r["slow_ratio"] > 1.2 for r in rows)
    mean_cer = float(np.mean([r["cer"] for r in rows])) if hear else None
    ok_heard = mean_cer is None or a.random or mean_cer <= MAX_ROUNDTRIP_CER
    report = {"model": "random" if a.random else REPO, "bytes": path.stat().st_size, "matches_pytorch": ok_match,
              "speed_works": ok_rate, "roundtrip_cer": mean_cer, "rows": rows}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    md = [f"### Text-to-speech export check: {report['model']}", "",
          f"ONNX file: {report['bytes'] / 2**20:.0f} MB. Matches PyTorch with noise off: **{ok_match}**. "
          f"Speed input works: **{ok_rate}**."
          + (f" Speech recogniser's character error rate on the output: **{mean_cer:.1%}**." if mean_cer is not None else ""),
          "", "| text | seconds | max diff vs PyTorch | heard by the ASR model |", "|---|---|---|---|"]
    md += [f"| {r['text']} | {r['seconds']} | {r['max_abs_diff']:.1e} | {r.get('heard', '')} |" for r in rows]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if not (ok_match and ok_rate and ok_heard):
        sys.exit("Text-to-speech export check failed: see the table above.")


if __name__ == "__main__":
    main()
