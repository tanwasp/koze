"""Export an Omnilingual ASR CTC model to ONNX for in-browser inference.

    python tools/export_onnx.py --card omniASR_CTC_300M_v2 --out web/model
    python tools/export_onnx.py --random --out /tmp/m   # random weights, no download

Writes:
  model.onnx        fp32 graph: waveform [1, T] (16 kHz, raw) -> logits [1, F, V]
  model.int8.onnx   8-bit weights, about a quarter of the size
  model.int8pc.onnx 8-bit with per-channel scales (fallback)
  vocab.json        token id -> piece, for CTC decoding in JavaScript
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path

import torch
from torch import nn


class CtcForExport(nn.Module):
    """Raw waveform in, CTC logits out. Normalisation matches the omnilingual-asr
    inference pipeline: layer norm over the whole clip."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        from fairseq2.nn import BatchLayout

        # Same as layer_norm over the clip, written out so the length stays dynamic.
        mean = wav.mean(dim=-1, keepdim=True)
        var = ((wav - mean) ** 2).mean(dim=-1, keepdim=True)
        wav = (wav - mean) / torch.sqrt(var + 1e-5)
        layout = BatchLayout(tuple(wav.shape), seq_lens=None)
        logits, _ = self.model(wav, layout)
        return logits


def load(card: str, random: bool):
    if random:
        from fairseq2.models.wav2vec2.asr.hub import get_wav2vec2_asr_model_hub

        hub = get_wav2vec2_asr_model_hub()
        arch = card.split("_CTC_")[1].lower()  # omniASR_CTC_300M_v2 -> 300m_v2
        model = hub.create_new_model(hub.get_arch_config(arch), device=torch.device("cpu"), dtype=torch.float32)
        return model, None
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.models.hub import load_model

    model = load_model(card, device=torch.device("cpu"), dtype=torch.float32)
    return model, load_tokenizer(card)


def export_vocab(tokenizer, size: int, path: Path) -> None:
    """Write the raw SentencePiece pieces. The web app decodes like the omnilingual-asr
    pipeline: collapse repeated ids, drop control symbols, join, '▁' -> space."""
    sp = tokenizer._model  # fairseq2 RawSentencePieceTokenizer keeps the SentencePieceModel here
    n = sp.vocabulary_size
    pieces = [sp.index_to_token(i) if i < n else "" for i in range(size)]
    vi = tokenizer.vocab_info
    control = sorted({i for i in (vi.pad_idx, vi.bos_idx, vi.eos_idx) if i is not None}
                     | {i for i, p in enumerate(pieces) if p in ("<pad>", "<s>", "</s>", "")})
    path.write_text(json.dumps({"pieces": pieces, "control": control, "unk": vi.unk_idx}, ensure_ascii=False))


def ctc_greedy(logits, vocab: dict) -> str:
    """Python twin of the JavaScript decoder in web/app.js, used to verify it."""
    ids = logits.argmax(-1).reshape(-1).tolist()
    control = set(vocab["control"])
    out, prev = [], None
    for i in ids:
        if i != prev and i not in control:
            out.append(vocab["pieces"][i])
        prev = i
    return "".join(out).replace("\u2581", " ").strip()


PARITY_SECONDS = (1.3, 4.0, 11.7, 30.0)


def parity_inputs():
    g = torch.Generator().manual_seed(0)
    return {f"{s}s": torch.randn(1, int(16000 * s), generator=g) * 0.1 for s in PARITY_SECONDS}


def check_parity(refs: dict, out: Path) -> dict:
    """Compare PyTorch outputs (computed before the model was freed) against each ONNX
    file on noise clips of several lengths. argmax_agree is how often the per-frame CTC
    argmax agrees (1.0 = identical decoding). One ONNX session is loaded at a time."""
    import gc

    import numpy as np
    import onnxruntime as ort

    inputs = parity_inputs()
    report = {k: {} for k in refs}
    for name in ("model.onnx", "model.int8.onnx", "model.int8pc.onnx"):
        if not (out / name).exists():
            continue
        sess = ort.InferenceSession(str(out / name))
        for k, ref in refs.items():
            got = sess.run(None, {"waveform": inputs[k].numpy()})[0]
            assert got.shape == ref.shape, (name, got.shape, ref.shape)
            report[k][name] = {"max_abs_diff": float(np.abs(got - ref).max()),
                               "argmax_agree": float((got.argmax(-1) == ref.argmax(-1)).mean())}
        del sess
        gc.collect()
    return report


def quantize(src: Path, dst: Path, per_channel: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=per_channel)


def export(card: str, random: bool, out: Path, opset: int, check: bool) -> int:
    model, tokenizer = load(card, random)
    model.eval()
    wrapped = CtcForExport(model).eval()
    dummy = torch.randn(1, 16000 * 4)
    with torch.no_grad():
        torch.onnx.export(
            wrapped, (dummy,), str(out / "model.onnx"),
            input_names=["waveform"], output_names=["logits"],
            dynamic_axes={"waveform": {1: "samples"}, "logits": {1: "frames"}},
            opset_version=opset, dynamo=False,
        )
    vocab_size = int(model.final_proj.weight.shape[0])
    if tokenizer is not None:
        export_vocab(tokenizer, vocab_size, out / "vocab.json")
    if check:
        import numpy as np

        with torch.no_grad():
            np.savez(out / "parity_refs.npz", **{k: wrapped(x).numpy() for k, x in parity_inputs().items()})
    return vocab_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card", default="omniASR_CTC_300M_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--random", action="store_true", help="random weights (pipeline test only)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check", action="store_true", help="compare ONNX outputs with PyTorch")
    p.add_argument("--step", choices=["all", "export", "int8", "int8pc", "check"], default="all", help=argparse.SUPPRESS)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.step == "export":
        print(json.dumps({"vocab_size": export(a.card, a.random, out, a.opset, a.check)}))
    elif a.step in ("int8", "int8pc"):
        # int8pc: per-channel scales, the fallback if per-tensor 8-bit loses too much accuracy.
        quantize(out / "model.onnx", out / f"model.{a.step}.onnx", per_channel=a.step == "int8pc")
    elif a.step == "check":
        import numpy as np

        refs = dict(np.load(out / "parity_refs.npz"))
        print(json.dumps({"parity": check_parity(refs, out)}, indent=1))
    else:
        # Each step runs in its own process so memory is fully returned between them:
        # for the 1B model, PyTorch holds about 5 GB and the quantiser peaks near 11 GB,
        # which together don't fit on a 16 GB machine.
        base = [sys.executable, __file__, "--card", a.card, "--out", str(out), "--opset", str(a.opset)]
        base += ["--random"] * a.random + ["--check"] * a.check
        steps = ["export", "int8", "int8pc"] + ["check"] * a.check
        for step in steps:
            subprocess.run(base + ["--step", step], check=True)
        print(json.dumps({"files": sorted(f.name for f in out.glob("*.onnx"))}))


if __name__ == "__main__":
    main()
