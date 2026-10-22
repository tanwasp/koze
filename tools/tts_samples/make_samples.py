"""Synthesise Kreol sample sentences with facebook/mms-tts-mfe so their quality can be judged by ear.

    python tools/tts_samples/make_samples.py --out tools/tts_samples/out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

# Grafi Larmoni spelling.
SENTENCES = [
    "Bonzour, kouma ou ete? Mo bien kontan trouv ou.",
    "Zordi lapli pe tonbe depi gramatin, me aswar pou ena soley.",
    "Mo pe al bazar Port-Louis pou aste legim ek pwason.",
    "Eski to kapav ed mwa ranz sa program-la avan Lindi?",
    "Kan mo ti piti, mo gran-mer ti rakont mwa zistwar Ti-Zan.",
    "Lekol finn ferme akoz siklonn, tou zanfan res lakaz.",
    "Kreol Morisien se enn lalang ki tou Morisien konpran.",
    "Ki ler li ete? Mo panse mo pou reta pou travay!",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-mfe")
    model = VitsModel.from_pretrained("facebook/mms-tts-mfe").eval()
    rate = model.config.sampling_rate
    index = []
    for i, text in enumerate(SENTENCES):
        torch.manual_seed(0)
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            wav = model(**ids).waveform[0].numpy().astype(np.float32)
        name = f"mfe_{i + 1:02d}.wav"
        sf.write(args.out / name, wav, rate)
        # What the model actually reads: characters outside its vocabulary are dropped.
        seen = tok.decode(ids["input_ids"][0])
        index.append({"file": name, "text": text, "model_sees": seen, "seconds": round(len(wav) / rate, 2)})
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
