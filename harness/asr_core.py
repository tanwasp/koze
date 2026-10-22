"""Core helpers for the Kreol ASR web harness: audio prep, splitting, phone
simulation, scoring and a cached wrapper around Meta's omnilingual-asr models.

Everything except `ModelBank` runs without the model package, so it can be
tested anywhere.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

import numpy as np

SR = 16_000
MAX_CLIP_S = 40.0  # hard limit in omnilingual-asr for non-Unlimited models
TARGET_CHUNK_S = 30.0  # aim for chunks well under the limit

# UI label -> model card name, from the omnilingual-asr 0.2.0 cards.
MODELS = {
    "CTC 300M": "omniASR_CTC_300M_v2",
    "CTC 1B": "omniASR_CTC_1B_v2",
    "CTC 3B": "omniASR_CTC_3B_v2",
    "CTC 7B": "omniASR_CTC_7B_v2",
    "LLM 300M": "omniASR_LLM_300M_v2",
    "LLM 1B": "omniASR_LLM_1B_v2",
    "LLM 3B": "omniASR_LLM_3B_v2",
    "LLM 7B": "omniASR_LLM_7B_v2",
    "LLM Unlimited 300M": "omniASR_LLM_Unlimited_300M_v2",
    "LLM Unlimited 1B": "omniASR_LLM_Unlimited_1B_v2",
    "LLM Unlimited 3B": "omniASR_LLM_Unlimited_3B_v2",
    "LLM Unlimited 7B": "omniASR_LLM_Unlimited_7B_v2",
}
DEFAULT_MODELS = ["CTC 300M", "LLM 1B"]

LANGS = {
    "Kreol Morisien (mfe_Latn)": "mfe_Latn",
    "French (fra_Latn)": "fra_Latn",
    "English (eng_Latn)": "eng_Latn",
    "Bhojpuri (bho_Deva)": "bho_Deva",
    "No hint": None,
}


def is_unlimited(card: str) -> bool:
    return "Unlimited" in card


def is_ctc(card: str) -> bool:
    return "_CTC_" in card


# ---------------------------------------------------------------- audio prep


def to_mono_16k(sr: int, wav: np.ndarray) -> np.ndarray:
    """Gradio gives (sr, int16|float array [, channels]). Return float32 mono 16 kHz."""
    wav = np.asarray(wav)
    if wav.dtype.kind == "i":
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    elif wav.dtype.kind == "u":
        wav = (wav.astype(np.float32) - 128) / 128
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        # Gradio uses (samples, channels); guard against (channels, samples).
        wav = wav.mean(axis=1) if wav.shape[0] > wav.shape[1] else wav.mean(axis=0)
    if sr != SR:
        wav = resample(wav, sr, SR)
    return wav


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    out = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr_in, sr_out)
    return out.numpy().astype(np.float32)


def simulate_phone(wav: np.ndarray) -> np.ndarray:
    """Rough G.711 phone line: 300–3400 Hz band, 8 kHz, 8-bit mu-law, back to 16 kHz."""
    import torch
    import torchaudio.functional as AF

    x = torch.from_numpy(np.ascontiguousarray(wav))
    x = AF.highpass_biquad(x, SR, 300.0)
    x = AF.lowpass_biquad(x, SR, 3400.0)
    x = AF.resample(x, SR, 8000)
    x = x.clamp(-1, 1)
    x = AF.mu_law_decoding(AF.mu_law_encoding(x, 256), 256)
    x = AF.resample(x, 8000, SR)
    return x.numpy().astype(np.float32)


def split_on_pauses(
    wav: np.ndarray, sr: int = SR, target_s: float = TARGET_CHUNK_S, max_s: float = MAX_CLIP_S - 2
) -> list[tuple[int, int]]:
    """Split into chunks of at most `max_s`, cutting at the quietest 50 ms frame
    between `target_s * 0.6` and `max_s`. Returns (start, end) sample indices."""
    n = len(wav)
    if n <= max_s * sr:
        return [(0, n)]
    hop = int(0.05 * sr)
    frames = n // hop
    energy = np.array([np.sqrt(np.mean(wav[i * hop : (i + 1) * hop] ** 2) + 1e-12) for i in range(frames)])
    bounds, start = [], 0
    while n - start > max_s * sr:
        lo = start + int(target_s * 0.6 * sr)
        hi = start + int(max_s * sr)
        f_lo, f_hi = lo // hop, min(hi // hop, frames)
        cut = (f_lo + int(np.argmin(energy[f_lo:f_hi]))) * hop if f_hi > f_lo else hi
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


# ---------------------------------------------------------------- scoring


def normalize(text: str) -> str:
    """Light normalisation for scoring: lowercase, unify apostrophes, drop
    punctuation, collapse spaces. Kreol spelling variants are NOT merged here;
    use the eval repo's variants.tsv for that."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score(reference: str, hypothesis: str) -> dict:
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return {}
    return {"wer": jiwer.wer(ref, hyp or "<empty>"), "cer": jiwer.cer(ref, hyp or " ")}


# ---------------------------------------------------------------- models


@dataclass
class Result:
    label: str
    text: str
    seconds: float
    load_seconds: float
    chunks: int
    error: str | None = None


class ModelBank:
    """Loads omnilingual-asr pipelines lazily and keeps them on the GPU."""

    def __init__(self, mock: bool = False, device: str = "auto", dtype: str = "auto"):
        self.mock = mock
        self.device = pick_device(device)
        self.dtype = dtype
        self.loaded: dict[str, object] = {}

    def _get(self, card: str):
        if card in self.loaded:
            return self.loaded[card], 0.0
        t0 = time.perf_counter()
        if self.mock:
            pipe = MockPipeline(card)
        else:
            pipe = self._load(card, self.device)
        self.loaded[card] = pipe
        return pipe, time.perf_counter() - t0

    def _load(self, card: str, device: str):
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        # bf16 on NVIDIA; fp32 elsewhere, which is the safe choice on CPU and MPS.
        dtype = dtypes.get(self.dtype) or (torch.bfloat16 if device == "cuda" else torch.float32)
        return ASRInferencePipeline(model_card=card, device=device, dtype=dtype)

    def unload_all(self) -> None:
        self.loaded.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(self, label: str, wav: np.ndarray, lang: str | None) -> Result:
        card = MODELS[label]
        try:
            pipe, load_s = self._get(card)
        except Exception as e:  # noqa: BLE001 - surface any load error in the UI
            return Result(label, "", 0.0, 0.0, 0, f"Load failed: {type(e).__name__}: {e}")
        spans = [(0, len(wav))] if is_unlimited(card) else split_on_pauses(wav)
        chunks = [{"waveform": wav[a:b], "sample_rate": SR} for a, b in spans]
        langs = None if is_ctc(card) or lang is None else [lang] * len(chunks)
        t0 = time.perf_counter()
        try:
            try:
                texts = pipe.transcribe(chunks, lang=langs, batch_size=1)
            except Exception:
                if self.mock or str(getattr(pipe, "device", "")) != "mps":
                    raise
                # Some fairseq2 ops may be missing on Apple's GPU: retry this model on CPU.
                pipe = self.loaded[card] = self._load(card, "cpu")
                t0 = time.perf_counter()
                texts = pipe.transcribe(chunks, lang=langs, batch_size=1)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if "out of memory" in msg.lower():
                msg += " — try a smaller model or click 'Unload models'."
            return Result(label, "", time.perf_counter() - t0, load_s, len(chunks), msg)
        return Result(label, " ".join(t.strip() for t in texts), time.perf_counter() - t0, load_s, len(chunks))


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MockPipeline:
    """Stands in for a real model so the UI can be tried without a GPU."""

    def __init__(self, card: str):
        self.card = card
        self.device = "cpu"

    def transcribe(self, chunks, lang=None, batch_size=1):
        return [f"[{self.card} mock: {len(c['waveform']) / SR:.1f}s, lang={lang[0] if lang else None}]" for c in chunks]
