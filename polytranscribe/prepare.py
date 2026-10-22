"""Cache GuitarSet from an attached folder or from the official ZIP archives."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import find_pairs, frame_targets, note_events
from .features import FEATURE_VERSION, load_audio, pitch_features


FILES = {
    "annotation.zip": "b39b78e63d3446f2e54ddb7a54df9b10",
    "audio_mono-mic.zip": "275966d6610ac34999b58426beb119c3",
}
RECORD = "https://zenodo.org/records/3371780/files/"


def checked_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def safe_extract(archive: Path, root: Path) -> None:
    root = root.resolve()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path in {archive}: {item.filename}")
        source.extractall(root)


def acquire(root: Path, annotation_zip: Path | None, audio_zip: Path | None, download: bool) -> None:
    for name, provided in (("annotation.zip", annotation_zip), ("audio_mono-mic.zip", audio_zip)):
        archive = provided or root / name
        if not archive.exists():
            if not download:
                raise FileNotFoundError(f"Provide {name} or enable --download")
            print(f"Downloading {name} from official Zenodo record", flush=True)
            with urllib.request.urlopen(RECORD + name + "?download=1", timeout=60) as response:
                with archive.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
        checked_md5(archive, FILES[name])
        print(f"Extracting {name}", flush=True)
        safe_extract(archive, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/guitarset"))
    parser.add_argument("--source-root", type=Path,
                        help="Existing extracted GuitarSet, e.g. /kaggle/input/guitarset")
    parser.add_argument("--annotation-zip", type=Path)
    parser.add_argument("--audio-zip", type=Path)
    parser.add_argument("--download", action="store_true", help="Requires Kaggle Internet enabled")
    args = parser.parse_args()
    cfg = Config()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.source_root:
        if args.download or args.annotation_zip or args.audio_zip:
            parser.error("--source-root cannot be combined with ZIP/download options")
        source_root = args.source_root
    else:
        acquire(args.root, args.annotation_zip, args.audio_zip, args.download)
        source_root = args.root
    pairs = find_pairs(source_root)
    cache = args.root / "cache"
    cache.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = Counter()
    overlaps = 0
    for i, (annotation, audio, split) in enumerate(pairs, 1):
        audio_tensor = load_audio(audio, cfg)
        x = pitch_features(audio_tensor, cfg, device)
        events = note_events(annotation)
        y, onset = frame_targets(events, x.shape[-1], cfg)
        overlaps += int((y.sum(axis=0) > 1).sum())
        np.savez_compressed(cache / f"{annotation.stem}.npz", x=x, y=y, onset=onset,
                            events=np.asarray(events, dtype=np.float32).reshape(-1, 3))
        counts[split] += 1
        if i % 30 == 0:
            print(f"Cached {i}/{len(pairs)}", flush=True)
    manifest = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                "task": "pitched", "stem": "guitar",
                "counts": dict(counts), "overlap_frames": overlaps,
                "source": str(source_root.resolve()), "input": "microphone mono",
                "split": "player 00-03/04/05"}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
