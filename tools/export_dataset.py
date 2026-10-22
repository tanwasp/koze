"""Download the shared clips from Supabase into a Hugging Face "audiofolder" dataset:
audio/<id>.wav plus metadata.jsonl, which `datasets.load_dataset("audiofolder", ...)` reads.

    SUPABASE_URL=https://abcd.supabase.co SUPABASE_SERVICE_KEY=... \\
        python tools/export_dataset.py --out build/dataset [--open-only] [--clean] [--push user/koze-kreol]

The service-role key can read and delete everything, so it stays on your machine; never
put it in the site or in a repository variable. --open-only keeps just the clips whose
contributors allowed an open release, and --push requires it unless you add --private.
--clean deletes audio that has no clip row (withdrawn clips, failed shares) and is over
an hour old.

Text is exported exactly as contributors typed it (`text`), next to the model's output
(`model_text`). Spelling normalisation belongs in the training pipeline, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BUCKET = "clips"
PAGE = 1000
# Columns that go in the dataset. The withdraw token hash stays behind.
COLUMNS = ["id", "created_at", "source", "duration_s", "model_id", "model_card", "backend", "model_text", "text",
           "spelling", "languages", "rating", "issues", "comment", "speaker", "consent_version", "open_release"]


class Supabase:
    def __init__(self, url: str, key: str):
        self.url, self.key = url.rstrip("/"), key

    def request(self, method: str, path: str, body: object = None) -> bytes:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "apikey": self.key, "authorization": f"Bearer {self.key}", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.read()

    def rows(self, open_only: bool) -> list[dict]:
        out = []
        while True:
            q = {"select": ",".join(COLUMNS), "order": "created_at", "limit": PAGE, "offset": len(out)}
            if open_only:
                q["open_release"] = "is.true"
            page = json.loads(self.request("GET", "/rest/v1/clips?" + urllib.parse.urlencode(q)))
            out += page
            if len(page) < PAGE:
                return out

    def objects(self) -> list[dict]:
        out = []
        while True:
            page = json.loads(self.request("POST", f"/storage/v1/object/list/{BUCKET}",
                                           {"prefix": "", "limit": PAGE, "offset": len(out)}))
            out += page
            if len(page) < PAGE:
                return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--open-only", action="store_true", help="only clips allowed in an open release")
    p.add_argument("--clean", action="store_true", help="delete orphaned audio over an hour old")
    p.add_argument("--push", help="Hugging Face dataset repo to upload to (needs HF_TOKEN)")
    p.add_argument("--private", action="store_true", help="make the pushed dataset private")
    a = p.parse_args()
    if a.push and not (a.open_only or a.private):
        sys.exit("--push publishes clips; add --open-only, or --private for a private dataset")
    sb = Supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    rows = sb.rows(a.open_only)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    keep = set()
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            name = f"{r['id']}.wav"
            keep.add(name)
            dst = out / "audio" / name
            if not dst.exists():
                dst.write_bytes(sb.request("GET", f"/storage/v1/object/{BUCKET}/{name}"))
            f.write(json.dumps({"file_name": f"audio/{name}", **r, "edited": r["text"] != r["model_text"]},
                               ensure_ascii=False) + "\n")
    # Drop local copies of clips that were withdrawn since the last export.
    for stale in (out / "audio").glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    hours = sum(r["duration_s"] for r in rows) / 3600
    print(f"{len(rows)} clips, {hours:.2f} h of audio, {sum(r['text'] != r['model_text'] for r in rows)} corrected")

    if a.clean:
        all_ids = {f"{r['id']}.wav" for r in (rows if not a.open_only else sb.rows(False))}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        orphans = [o["name"] for o in sb.objects() if o["name"] not in all_ids
                   and datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")) < cutoff]
        for i in range(0, len(orphans), 100):
            sb.request("DELETE", f"/storage/v1/object/{BUCKET}", {"prefixes": orphans[i : i + 100]})
        print(f"deleted {len(orphans)} orphaned audio files")

    if a.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(a.push, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=a.push, repo_type="dataset", folder_path=out, delete_patterns=["audio/*.wav"],
                          commit_message=f"Export {len(rows)} clips")
        print(f"pushed to https://huggingface.co/datasets/{a.push}")


if __name__ == "__main__":
    main()
