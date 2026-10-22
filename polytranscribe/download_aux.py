"""Fetch the two compact auxiliary corpora for a five-stem prototype."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .prepare import safe_extract


DATASETS = {
    "vocadito": {
        "url": "https://zenodo.org/records/5578807/files/vocadito.zip?download=1",
        "file": "vocadito.zip",
        "md5": "dea40fd18f14d899643c4ba221b33a46",
    },
    "babyslakh": {
        "url": "https://zenodo.org/records/4603844/files/babyslakh_16k.zip?download=1",
        "file": "babyslakh_16k.zip",
        "md5": "ea1797fc57689a0e33c759c17a2292f5",
    },
}


def fetch(name: str, root: Path) -> Path:
    spec = DATASETS[name]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec["file"]
    extract_to = root / name
    marker = extract_to / ".ready.json"
    if marker.is_file():
        return extract_to
    if not archive.is_file():
        temporary = archive.with_name(archive.name + ".part")
        digest = hashlib.md5()
        total = 0
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            with temporary.open("wb") as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
                    digest.update(block)
                    total += len(block)
                    if total // (50 * 1024 * 1024) != (total - len(block)) // (50 * 1024 * 1024):
                        print(f"{name}: {total / 1024**2:.0f} MiB", flush=True)
        if digest.hexdigest() != spec["md5"]:
            raise ValueError(f"Checksum mismatch for {name}")
        temporary.replace(archive)
    else:
        digest = hashlib.md5()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != spec["md5"]:
            raise ValueError(f"Checksum mismatch for {archive}")
    extract_to.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, extract_to)
    marker.write_text(json.dumps({"source": spec["url"], "md5": spec["md5"]}), encoding="utf-8")
    return extract_to


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("vocadito", "babyslakh", "both"), default="both")
    args = parser.parse_args()
    names = DATASETS if args.dataset == "both" else (args.dataset,)
    for name in names:
        print(f"{name}: {fetch(name, args.output)}")


if __name__ == "__main__":
    main()
