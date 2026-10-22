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


def check_parity(wrapped: nn.Module, out: Path, seconds=(1.3, 4.0, 11.7, 30.0)) -> dict:
    """Compare PyTorch against both ONNX files on noise clips of several lengths.
    Reports how often the per-frame CTC argmax agrees (1.0 = identical decoding)."""
    import numpy as np
    import onnxruntime as ort

    names = [n for n in ("model.onnx", "model.int8.onnx", "model.int8pc.onnx") if (out / n).exists()]
    sessions = {n: ort.InferenceSession(str(out / n)) for n in names}
    g = torch.Generator().manual_seed(0)
    report = {}
    for secs in seconds:
        x = torch.randn(1, int(16000 * secs), generator=g) * 0.1
        with torch.no_grad():
            ref = wrapped(x).numpy()
        row = {}
        for name, sess in sessions.items():
            got = sess.run(None, {"waveform": x.numpy()})[0]
            assert got.shape == ref.shape, (name, got.shape, ref.shape)
            row[name] = {"max_abs_diff": float(np.abs(got - ref).max()),
                         "argmax_agree": float((got.argmax(-1) == ref.argmax(-1)).mean())}
        report[f"{secs}s"] = row
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card", default="omniASR_CTC_300M_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--random", action="store_true", help="random weights (pipeline test only)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check", action="store_true", help="compare ONNX outputs with PyTorch")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(a.card, a.random)
    model.eval()
    wrapped = CtcForExport(model).eval()
    dummy = torch.randn(1, 16000 * 4)
    with torch.no_grad():
        torch.onnx.export(
            wrapped, (dummy,), str(out / "model.onnx"),
            input_names=["waveform"], output_names=["logits"],
            dynamic_axes={"waveform": {1: "samples"}, "logits": {1: "frames"}},
            opset_version=a.opset, dynamo=False, external_data=False,
        )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    # Fallback if per-tensor 8-bit loses too much accuracy (see verify_onnx.py).
    quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8pc.onnx"), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=True)

    vocab_size = int(model.final_proj.weight.shape[0])
    if tokenizer is not None:
        export_vocab(tokenizer, vocab_size, out / "vocab.json")
    result = {"vocab_size": vocab_size, "files": sorted(f.name for f in out.iterdir())}
    if a.check:
        result["parity"] = check_parity(wrapped, out)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
