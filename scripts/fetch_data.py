#!/usr/bin/env python3.12
"""Download the benchmark dataset from Zenodo into $SW_DATA_DIR (or --dest).

    python3.12 scripts/fetch_data.py            # -> $SW_DATA_DIR or ./data/goes_data/processed
    python3.12 scripts/fetch_data.py --dest /path/to/processed

Verifies each file's MD5 and skips files already present and intact.
Dataset DOI: 10.5281/zenodo.22245285
"""
import argparse
import hashlib
import json
import os
import urllib.request

ZENODO_RECORD = "22245285"
API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("SW_DATA_DIR", "./data/goes_data/processed"))
    dest = ap.parse_args().dest
    os.makedirs(dest, exist_ok=True)
    with urllib.request.urlopen(API) as r:
        record = json.load(r)
    for f in record["files"]:
        name, url = f["key"], f["links"]["self"]
        want = f["checksum"].split(":")[-1]
        out = os.path.join(dest, name)
        if os.path.exists(out) and md5(out) == want:
            print("ok (cached):", name)
            continue
        print(f"downloading {name} ({f['size'] / 1e6:.0f} MB) ...")
        urllib.request.urlretrieve(url, out)
        print(("  verified: " if md5(out) == want else "  CHECKSUM MISMATCH: ") + name)
    print("done ->", dest)


if __name__ == "__main__":
    main()
