"""Upload a packed model (from `build_site.py pack`) to a Hugging Face model repo, for
sizes too big to sit next to the site on GitHub Pages (which caps a site at 1 GB).

    HF_TOKEN=... python tools/publish_hf.py --dir build/large-pack --repo user/koze-models --path omniASR_CTC_1B_v2

The web app does not trust this host: models.json on the site pins a SHA-256 for every
part, and the page rejects anything that doesn't match.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CARD = """---
license: apache-2.0
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: facebook/omniASR-CTC-1B
---
# Koze model files

ONNX conversions of Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
CTC models (Apache 2.0), with 8-bit weights, split into parts for the
[Koze](https://github.com/tanwasp/koze) web app. Built and checked against Meta's PyTorch
pipeline by the repo's GitHub Actions workflow.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--path", required=True)
    a = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    local = json.loads((Path(a.dir) / "manifest.json").read_text())
    try:
        remote = json.loads(Path(hf_hub_download(a.repo, f"{a.path}/manifest.json", token=api.token)).read_text())
    except EntryNotFoundError:
        remote = None
    if remote and remote["sha256"] == local["sha256"]:
        print(f"{a.repo}/{a.path} already has {local['sha256'][:12]}, nothing to upload")
        return
    if not api.file_exists(a.repo, "README.md"):
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=a.repo)
    # Old parts stay: part names start with the model hash, and the live site may still
    # point at them until its next deploy.
    api.upload_folder(folder_path=a.dir, path_in_repo=a.path, repo_id=a.repo,
                      commit_message=f"{local['card']} {local['sha256'][:12]}")
    print(f"Uploaded {local['card']} ({local['bytes']} bytes) to {a.repo}/{a.path}")


if __name__ == "__main__":
    main()
