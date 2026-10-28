"""
Download the audio tracks for a list of company song IDs (read-only: list + download).

    lyrics2audio/bin/python data/download_tracks.py --env /path/to/.env \
        --slugs data/company_add_2026-10-25.txt

Track keys come from ~/Downloads/company-s3/songs.csv (column newest_track); files are
saved under ~/Downloads/company-s3/<key>, where data/export_company.py looks for them.
Credentials: the .env used for s3.mjs (any KEY=VALUE lines whose names contain BUCKET,
REGION, ACCESS_KEY/KEY_ID and SECRET). Nothing is written to the bucket.
"""
import argparse
import csv
import os
from pathlib import Path

import boto3


def read_env(path):
    vals = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    pick = lambda *words: next((v for k, v in vals.items() if all(w in k.upper() for w in words)), None)
    return {"bucket": pick("BUCKET"), "region": pick("REGION"),
            "key": pick("ACCESS_KEY") or pick("KEY_ID"), "secret": pick("SECRET")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="path to the .env with the S3 credentials")
    ap.add_argument("--slugs", required=True, help="text file with song IDs (space or newline separated)")
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "company-s3"))
    args = ap.parse_args()

    cfg = read_env(args.env)
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(f"could not find {missing} in {args.env}")
    s3 = boto3.client("s3", region_name=cfg["region"], aws_access_key_id=cfg["key"],
                      aws_secret_access_key=cfg["secret"])
    src = Path(args.src).expanduser()
    with open(src / "songs.csv", encoding="utf-8") as f:
        keys = {r["slug"]: r["newest_track"] for r in csv.DictReader(f)}
    slugs = Path(args.slugs).read_text().split()

    done = skipped = 0
    total_mb = 0.0
    for i, slug in enumerate(slugs, 1):
        key = keys.get(slug)
        if not key:
            print(f"[{i}/{len(slugs)}] {slug}: no track listed in songs.csv -- skipped")
            continue
        out = src / key
        size = s3.head_object(Bucket=cfg["bucket"], Key=key)["ContentLength"]
        if out.exists() and out.stat().st_size == size:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        s3.download_file(cfg["bucket"], key, str(tmp))
        os.replace(tmp, out)
        done += 1
        total_mb += size / 1e6
        print(f"[{i}/{len(slugs)}] {slug}: {key} ({size / 1e6:.1f} MB)", flush=True)
    print(f"done: {done} downloaded ({total_mb:.0f} MB), {skipped} already there")


if __name__ == "__main__":
    main()
