"""Assemble the GitHub Pages site: the web app plus the chosen model, split into
parts small enough for any static host.

    python tools/build_site.py --model-dir build/model --report build/report.json --out site
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PART_BYTES = 40 * 2**20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src, out = Path(a.model_dir), Path(a.out)
    report = json.loads(Path(a.report).read_text())

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(Path(__file__).parent.parent / "web", out)
    (out / "model").mkdir()

    model = src / report["ship"]
    data = model.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    parts = []
    for i in range(0, len(data), PART_BYTES):
        name = f"model/{sha[:12]}.part{i // PART_BYTES:02d}"
        (out / name).write_bytes(data[i : i + PART_BYTES])
        parts.append(name)
    shutil.copy(src / "vocab.json", out / "model" / "vocab.json")
    manifest = {
        "card": report["card"],
        "file": report["ship"],
        "bytes": len(data),
        "sha256": sha,
        "parts": parts,
        "vocab": "model/vocab.json",
        "check": report["summary"],
    }
    (out / "model" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / ".nojekyll").write_text("")
    print(json.dumps({k: v for k, v in manifest.items() if k != "parts"} | {"parts": len(parts)}, indent=1))


if __name__ == "__main__":
    main()
