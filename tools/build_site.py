"""Assemble the GitHub Pages site: the web app, its pinned third-party files, and the
model registry (web/models.json) with a SHA-256 for every model part.

    # pack one converted model into parts + manifest.json (also used for the Hugging Face upload)
    python tools/build_site.py pack --model-dir build/model --report build/report.json --out site/model

    # pack the text-to-speech model written by tools/export_tts.py
    python tools/build_site.py pack-tts --model-dir build/tts --out build/tts-pack

    # build the site; --remote adds a model whose parts live elsewhere, pinned by its manifest
    python tools/build_site.py site --local site-model-dir --remote build/1b/manifest.json=https://huggingface.co/... \
        --tts build/tts-pack --out site

The page trusts models.json (served from this site) and checks every downloaded part
against it, so a model host can't swap in a different file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
PART_BYTES = 40 * 2**20

# Third-party files served from this site instead of a CDN. Each tarball is checked
# against the integrity hash npm published for it.
VENDOR = {
    "onnxruntime-web@1.27.0": {
        "integrity": "sha512-ogDLsqIozHZwifPuN37OproAo0byX6t43/bP8GzeZWBWD6MOGExswFAx3up4NS/vvWBOg2u2PXomDt3rMmdQSg==",
        "files": {
            "dist/ort.webgpu.min.mjs": "vendor/ort/ort.webgpu.min.mjs",
            "dist/ort-wasm-simd-threaded.asyncify.mjs": "vendor/ort/ort-wasm-simd-threaded.asyncify.mjs",
            "dist/ort-wasm-simd-threaded.asyncify.wasm": "vendor/ort/ort-wasm-simd-threaded.asyncify.wasm",
        },
    },
    "@fontsource/inter@5.3.0": {
        "integrity": "sha512-RofMylZmjlJEfELXeNHFWBRcSs75rGU/6bV2S2jfnvv/3rPXPGe0LgUJTklcHZ9lM4OZmAVFhcJPnACfb91A3g==",
        "files": {f"files/inter-latin-{w}-normal.woff2": f"vendor/fonts/inter-{w}.woff2" for w in (400, 500, 600)}
        | {"LICENSE": "vendor/fonts/LICENSE-inter"},
    },
    "@fontsource/instrument-serif@5.3.0": {
        "integrity": "sha512-mDiaIg0u67sYV59fie92Wz4sM8UiVlbL7fLxnFPCKkX15DMASEnQTREbaP5S5/3DCcsoAOcQ0sV9E4AeY4QqYQ==",
        "files": {
            "files/instrument-serif-latin-400-normal.woff2": "vendor/fonts/instrument-serif-400.woff2",
            "files/instrument-serif-latin-400-italic.woff2": "vendor/fonts/instrument-serif-400-italic.woff2",
            "LICENSE": "vendor/fonts/LICENSE-instrument-serif",
        },
    },
}

# What the picker says about each size. Keys are model card names.
ABOUT = {
    "omniASR_CTC_300M_v2": {"id": "300m", "name": "Standard", "params": "300M",
                            "blurb": "Fast on any recent laptop or phone."},
    "omniASR_CTC_1B_v2": {"id": "1b", "name": "Large", "params": "1B",
                          "blurb": "More accurate. Wants a laptop or desktop with a good GPU."},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split(data: bytes, out: Path) -> list[dict]:
    """Write data to out in PART_BYTES parts named after its hash."""
    digest = sha256(data)
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(0, len(data), PART_BYTES):
        chunk = data[i : i + PART_BYTES]
        name = f"{digest[:12]}.part{i // PART_BYTES:02d}"
        (out / name).write_bytes(chunk)
        parts.append({"path": name, "bytes": len(chunk), "sha256": sha256(chunk)})
    return parts


def pack(model_dir: Path, report_path: Path, out: Path) -> dict:
    """Split the shipped model into parts and write manifest.json next to them.
    Paths in the manifest are relative to the manifest."""
    report = json.loads(report_path.read_text())
    data = (model_dir / report["ship"]).read_bytes()
    digest = sha256(data)
    parts = split(data, out)
    vocab = (model_dir / "vocab.json").read_bytes()
    (out / "vocab.json").write_bytes(vocab)
    manifest = {
        "card": report["card"],
        "file": report["ship"],
        "bytes": len(data),
        "sha256": digest,
        "parts": parts,
        "vocab": {"path": "vocab.json", "sha256": sha256(vocab)},
        "check": report["summary"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def pack_tts(model_dir: Path, out: Path) -> dict:
    """Same layout as pack(), for the text-to-speech model and its tokenizer table."""
    data = (model_dir / "tts.onnx").read_bytes()
    parts = split(data, out)
    table = (model_dir / "tts.json").read_bytes()
    (out / "tts.json").write_bytes(table)
    manifest = {
        "card": json.loads(table)["model"],
        "licence": json.loads(table)["licence"],
        "bytes": len(data),
        "sha256": sha256(data),
        "parts": parts,
        "table": {"path": "tts.json", "sha256": sha256(table)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def vendor(out: Path, cache: Path) -> None:
    notices = []
    for spec, v in VENDOR.items():
        name, version = spec.rsplit("@", 1)
        tgz = cache / f"{name.replace('/', '_')}-{version}.tgz"
        if not tgz.exists():
            url = f"https://registry.npmjs.org/{name}/-/{name.split('/')[-1]}-{version}.tgz"
            cache.mkdir(parents=True, exist_ok=True)
            tgz.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        data = tgz.read_bytes()
        got = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
        if got != v["integrity"]:
            tgz.unlink()
            raise SystemExit(f"{spec}: tarball does not match the pinned integrity hash")
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for src, dst in v["files"].items():
                (out / dst).parent.mkdir(parents=True, exist_ok=True)
                (out / dst).write_bytes(tar.extractfile(f"package/{src}").read())
            licence = json.loads(tar.extractfile("package/package.json").read()).get("license", "see package")
        notices.append(f"{spec}  ({licence})  {v['integrity']}")
    (out / "vendor" / "NOTICE.txt").write_text("Third-party files served by this site, from npm:\n\n" + "\n".join(notices) + "\n")


def model_entry(manifest: dict, base: str) -> dict:
    about = ABOUT.get(manifest["card"], {"id": manifest["card"], "name": manifest["card"], "params": "?", "blurb": ""})
    return about | {"card": manifest["card"], "base": base} | {
        k: manifest[k] for k in ("bytes", "sha256", "parts", "vocab")}


def build_site(local: list[Path], remote: list[str], tts: Path | None, out: Path, cache: Path) -> None:
    staged = {}
    for d in local + ([tts] if tts else []):  # copy before wiping out, in case a model dir lives inside it
        staged[d] = (json.loads((d / "manifest.json").read_text()), {p.name: p.read_bytes() for p in d.iterdir()})
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    vendor(out, cache)

    voice = None
    if tts:
        manifest, files = staged.pop(tts)
        (out / "tts").mkdir()
        for name, data in files.items():
            (out / "tts" / name).write_bytes(data)
        voice = {"base": "tts/"} | {k: manifest[k] for k in ("card", "licence", "bytes", "sha256", "parts", "table")}

    models = []
    for i, (manifest, files) in enumerate(staged.values()):
        # The first local model keeps the old model/ path, so existing browser caches stay valid.
        sub = "model" if i == 0 else f"model-{manifest['card']}"
        (out / sub).mkdir()
        for name, data in files.items():
            (out / sub / name).write_bytes(data)
        models.append(model_entry(manifest, f"{sub}/"))
    for r in remote:
        path, base = r.split("=", 1)
        models.append(model_entry(json.loads(Path(path).read_text()), base.rstrip("/") + "/"))
    models.sort(key=lambda m: m["bytes"])
    (out / "models.json").write_text(json.dumps({"models": models} | ({"tts": voice} if voice else {}), indent=1))
    (out / ".nojekyll").write_text("")
    print(json.dumps([{k: m[k] for k in ("id", "card", "bytes", "base")} | {"parts": len(m["parts"])}
                      for m in models], indent=1))
    if voice:
        print(f"text to speech: {voice['card']}, {voice['bytes']} bytes in {len(voice['parts'])} parts")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack")
    pk.add_argument("--model-dir", required=True)
    pk.add_argument("--report", required=True)
    pk.add_argument("--out", required=True)
    pt = sub.add_parser("pack-tts")
    pt.add_argument("--model-dir", required=True)
    pt.add_argument("--out", required=True)
    st = sub.add_parser("site")
    st.add_argument("--local", action="append", default=[], help="a directory written by `pack`")
    st.add_argument("--remote", action="append", default=[], help="MANIFEST_PATH=BASE_URL")
    st.add_argument("--tts", help="a directory written by `pack-tts`")
    st.add_argument("--out", required=True)
    st.add_argument("--cache", default="build/vendor")
    a = p.parse_args()
    if a.cmd == "pack":
        m = pack(Path(a.model_dir), Path(a.report), Path(a.out))
        print(json.dumps({k: v for k, v in m.items() if k != "parts"} | {"parts": len(m["parts"])}, indent=1))
    elif a.cmd == "pack-tts":
        m = pack_tts(Path(a.model_dir), Path(a.out))
        print(json.dumps({k: v for k, v in m.items() if k != "parts"} | {"parts": len(m["parts"])}, indent=1))
    else:
        build_site([Path(d) for d in a.local], a.remote, Path(a.tts) if a.tts else None, Path(a.out), Path(a.cache))


if __name__ == "__main__":
    main()
