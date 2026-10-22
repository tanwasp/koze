import numpy as np

from asr_core import MAX_CLIP_S, SR, ModelBank, normalize, score, simulate_phone, split_on_pauses, to_mono_16k


def speechy(seconds, gap_every=7.0):
    """Noise bursts with 0.4 s silences every `gap_every` seconds."""
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.3, int(seconds * SR)).astype(np.float32)
    for t in np.arange(gap_every, seconds, gap_every):
        wav[int(t * SR) : int((t + 0.4) * SR)] = 0
    return wav


def test_short_clip_not_split():
    assert split_on_pauses(np.zeros(SR * 10, np.float32)) == [(0, SR * 10)]


def test_long_clip_split_under_limit_and_at_pauses():
    wav = speechy(150)
    spans = split_on_pauses(wav)
    assert spans[0][0] == 0 and spans[-1][1] == len(wav)
    assert all(b == c for (_, b), (c, _) in zip(spans, spans[1:]))
    assert all((b - a) / SR < MAX_CLIP_S for a, b in spans)
    for _, b in spans[:-1]:  # every cut lands inside a silence
        assert np.abs(wav[b : b + 100]).max() == 0


def test_to_mono_16k_int16_stereo_44k():
    data = (np.ones((44100, 2)) * 16000).astype(np.int16)
    out = to_mono_16k(44100, data)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - SR) < 5
    assert 0.4 < out[1000] < 0.6


def test_phone_sim_keeps_length_and_removes_highs():
    t = np.arange(SR) / SR
    hi = np.sin(2 * np.pi * 6000 * t).astype(np.float32) * 0.5
    out = simulate_phone(hi)
    assert abs(len(out) - len(hi)) < 5
    assert np.abs(out[2000:-2000]).max() < 0.1


def test_normalize_and_score():
    assert normalize("Bonzour, ki manyer?  Mo  BIEN!") == "bonzour ki manyer mo bien"
    assert normalize("li’nn al") == "li'nn al"
    s = score("mo pe al lakaz", "mo pe al lakaz")
    assert s["wer"] == 0 and s["cer"] == 0
    assert abs(score("mo pe al lakaz", "mo al lakaz")["wer"] - 0.25) < 1e-9
    assert score("", "anything") == {}


def test_mock_bank_splits_except_unlimited():
    bank = ModelBank(mock=True)
    wav = speechy(90)
    r = bank.transcribe("CTC 300M", wav, "mfe_Latn")
    assert r.error is None and r.chunks >= 3 and "lang=None" in r.text  # CTC ignores lang
    r = bank.transcribe("LLM 1B", wav, "mfe_Latn")
    assert "lang=mfe_Latn" in r.text
    r = bank.transcribe("LLM Unlimited 1B", wav, "mfe_Latn")
    assert r.chunks == 1
