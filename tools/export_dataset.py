"""Download the shared clips from Supabase into a Hugging Face "audiofolder" dataset:
audio/<id>.wav plus metadata.jsonl, which `datasets.load_dataset("audiofolder", ...)` reads.

    SUPABASE_URL=https://abcd.supabase.co SUPABASE_SERVICE_KEY=... \\
        python tools/export_dataset.py --out build/dataset [--open-only] [--clean] [--push user/koze-kreol] \
            [--contributors build/contributors.csv]

The service-role key can read and delete everything, so it stays on your machine; never
put it in the site or in a repository variable. --open-only keeps just the clips whose
contributors allowed an open release, and --push requires it unless you add --private.
--clean deletes audio that has no clip row (deleted clips and accounts, failed shares)
and is over an hour old. --contributors writes a per-person summary (email, role, clips,
minutes, corrected) for paying or thanking collectors; it holds emails, so keep it private.

Each clip gets a `contributor` id: a hash of the account id, so clips can be grouped by
person (for example to keep a speaker out of both train and test) without exposing it.

Text is exported exactly as contributors typed it (`text`), next to the model's output
(`model_text`). Spelling normalisation belongs in the training pipeline, not here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
COLUMNS = ["id", "created_at", "user_id", "source", "duration_s", "model_id", "model_card", "backend", "model_text", "text",
           "spelling", "languages", "rating", "issues", "comment", "own_voice", "speaker", "consent_version", "open_release"]


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

    def list(self, prefix: str) -> list[dict]:
        out = []
        while True:
            page = json.loads(self.request("POST", f"/storage/v1/object/list/{BUCKET}",
                                           {"prefix": prefix, "limit": PAGE, "offset": len(out)}))
            out += page
            if len(page) < PAGE:
                return out

    def objects(self) -> list[dict]:
        """Every audio file, as {name: "<user>/<clip>.wav", created_at}. Folders list with no id."""
        return [{**o, "name": f"{d['name']}/{o['name']}"}
                for d in self.list("") if d.get("id") is None
                for o in self.list(d["name"] + "/") if o.get("id")]

    def profiles(self) -> list[dict]:
        return json.loads(self.request("GET", "/rest/v1/profiles?select=id,display_name,role,created_at"))

    def emails(self) -> dict[str, str]:
        out, page = {}, 1
        while True:
            users = json.loads(self.request("GET", f"/auth/v1/admin/users?page={page}&per_page={PAGE}"))["users"]
            out |= {u["id"]: u.get("email", "") for u in users}
            if len(users) < PAGE:
                return out
            page += 1


def contributor_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--open-only", action="store_true", help="only clips allowed in an open release")
    p.add_argument("--clean", action="store_true", help="delete orphaned audio over an hour old")
    p.add_argument("--push", help="Hugging Face dataset repo to upload to (needs HF_TOKEN)")
    p.add_argument("--private", action="store_true", help="make the pushed dataset private")
    p.add_argument("--contributors", help="write a per-person CSV here (has emails: keep it out of the dataset)")
    a = p.parse_args()
    if a.push and not (a.open_only or a.private):
        sys.exit("--push publishes clips; add --open-only, or --private for a private dataset")
    sb = Supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    rows = sb.rows(a.open_only)
    out = Path(a.out)
    if a.contributors and Path(a.contributors).resolve().is_relative_to(out.resolve()):
        sys.exit("--contributors must be outside --out, so emails never end up in the dataset")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    keep = set()
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            name = f"{r['id']}.wav"
            keep.add(name)
            dst = out / "audio" / name
            if not dst.exists():
                dst.write_bytes(sb.request("GET", f"/storage/v1/object/{BUCKET}/{r['user_id']}/{name}"))
            meta = {k: v for k, v in r.items() if k != "user_id"}
            f.write(json.dumps({"file_name": f"audio/{name}", "contributor": contributor_id(r["user_id"]), **meta,
                                "edited": r["text"] != r["model_text"]}, ensure_ascii=False) + "\n")
    # Drop local copies of clips that were withdrawn since the last export.
    for stale in (out / "audio").glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    hours = sum(r["duration_s"] for r in rows) / 3600
    print(f"{len(rows)} clips, {hours:.2f} h of audio, {sum(r['text'] != r['model_text'] for r in rows)} corrected")

    if a.clean:
        all_ids = {f"{r['user_id']}/{r['id']}.wav" for r in (rows if not a.open_only else sb.rows(False))}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        orphans = [o["name"] for o in sb.objects() if o["name"] not in all_ids
                   and datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")) < cutoff]
        for i in range(0, len(orphans), 100):
            sb.request("DELETE", f"/storage/v1/object/{BUCKET}", {"prefixes": orphans[i : i + 100]})
        print(f"deleted {len(orphans)} orphaned audio files")

    if a.contributors:
        every = rows if not a.open_only else sb.rows(False)
        emails = sb.emails()
        with open(a.contributors, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["contributor", "email", "display_name", "role", "joined", "clips", "minutes", "corrected"])
            for pr in sb.profiles():
                mine = [r for r in every if r["user_id"] == pr["id"]]
                w.writerow([contributor_id(pr["id"]), emails.get(pr["id"], ""), pr["display_name"] or "", pr["role"],
                            pr["created_at"][:10], len(mine), round(sum(r["duration_s"] for r in mine) / 60, 1),
                            sum(r["text"] != r["model_text"] for r in mine)])
        print(f"wrote {a.contributors}")

    if a.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(a.push, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=a.push, repo_type="dataset", folder_path=out, delete_patterns=["audio/*.wav"],
                          commit_message=f"Export {len(rows)} clips")
        print(f"pushed to https://huggingface.co/datasets/{a.push}")


if __name__ == "__main__":
    main()
