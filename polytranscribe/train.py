"""Train and evaluate one ablation arm on player-held-out GuitarSet."""

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import Config
from .data import player_split
from .features import FEATURE_VERSION
from .metrics import (decode_drums, decode_notes, frame_scores, frame_scores_by_polyphony,
                      note_scores, paper_voicing_gate)
from .model import build_transcriber, predict_in_chunks


CHECKPOINT_FORMAT = 2


def manifest_identity(manifest: dict) -> str:
    """Bind a resumable run to its source, selection and split assignments."""
    fields = {key: manifest.get(key) for key in
              ("feature_version", "task", "stem", "source", "max_slakh_tracks",
               "counts", "file_splits", "split", "config")}
    serialized = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after the new copy was completely written."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@lru_cache(maxsize=16)
def read_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        return archive["x"].astype(np.float32), archive["y"].astype(np.float32), \
            archive["onset"].astype(np.float32), archive["events"]


class WindowDataset(Dataset):
    def __init__(self, files: list[Path], cfg: Config):
        self.cfg = cfg
        self.windows = []
        for path in files:
            with np.load(path) as archive:
                length = archive["x"].shape[-1]
            self.windows.extend((path, start) for start in range(0, length, cfg.window_frames))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        path, start = self.windows[index]
        x, y, onset, _ = read_cache(path)
        stop = start + self.cfg.window_frames
        def window(array):
            part = array[..., start:stop]
            return np.pad(part, [(0, 0)] * (part.ndim - 1) + [(0, self.cfg.window_frames - part.shape[-1])])
        return torch.from_numpy(window(x)), torch.from_numpy(window(y)), torch.from_numpy(window(onset))


def split_files(cache: Path, cfg: Config) -> dict[str, list[Path]]:
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare first: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature cache is outdated; run prepare again")
    for key in ("sample_rate", "n_fft", "n_mels", "hop_length", "midi_min", "midi_max"):
        if manifest["config"][key] != getattr(cfg, key):
            raise ValueError(f"Cache configuration mismatch: {key}")
    files = {split: [] for split in ("train", "val", "test")}
    file_splits = manifest.get("file_splits")
    if file_splits is not None:
        for stem, split in sorted(file_splits.items()):
            path = cache / f"{stem}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            files[split].append(path)
    else:
        for path in sorted(cache.glob("*.npz")):
            files[player_split(path.stem)].append(path)
    expected = manifest["counts"] if file_splits is not None else {"train": 240, "val": 60, "test": 60}
    if {split: len(group) for split, group in files.items()} != expected:
        raise ValueError(f"Incomplete cache: expected {expected}, got { {k: len(v) for k,v in files.items()} }")
    return files


@torch.no_grad()
def predictions(model: nn.Module, files: list[Path], device: torch.device,
                fixed_gate: bool = False):
    model.eval()
    outputs = []
    for path in files:
        x, y, _, events = read_cache(path)
        note, onset = predict_in_chunks(model, torch.from_numpy(x[None]), device)
        note_prob = note[0].cpu().numpy()
        if fixed_gate:
            note_prob = paper_voicing_gate(note_prob, x)
        outputs.append((note_prob, torch.sigmoid(onset[0]).cpu().numpy(),
                        y.astype(bool), [tuple(map(float, event)) for event in events]))
    return outputs


def summarize(outputs, threshold: float, cfg: Config, task: str = "pitched") -> dict:
    frame_counts = {"tp": 0, "fp": 0, "fn": 0}
    polyphonic_counts = {"tp": 0, "fp": 0, "fn": 0, "frames": 0}
    note_counts = {"tp": 0, "fp": 0, "fn": 0}
    onset_counts = {"tp": 0, "fp": 0, "fn": 0}
    for note, onset, target, reference in outputs:
        frame = frame_scores(note >= threshold, target)
        polyphonic = frame_scores_by_polyphony(note >= threshold, target)
        estimated = (decode_drums(note, cfg, threshold=threshold) if task == "drums"
                     else decode_notes(note, onset, cfg, threshold=threshold))
        notes = note_scores(estimated, reference, offset_ratio=None if task == "drums" else 0.2)
        onsets = note_scores(estimated, reference, offset_ratio=None)
        for key in frame_counts:
            frame_counts[key] += frame[key]
            polyphonic_counts[key] += polyphonic[key]
            note_counts[key] += notes[key]
            onset_counts[key] += onsets[key]
        polyphonic_counts["frames"] += polyphonic["frames"]
    def ratios(counts):
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**counts, "precision": precision, "recall": recall, "f1": f1}
    return {"frame": ratios(frame_counts),
            "polyphonic_frame": ratios(polyphonic_counts),
            "note": ratios(note_counts), "onset_note": ratios(onset_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/guitarset/cache"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--variant", required=True,
                        choices=("baseline", "gated", "advanced_plain", "advanced_gated", "advanced"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--resume", type=Path, help="Previous *_last.pt, including from /kaggle/input")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save resumable state after this many batches")
    parser.add_argument("--max-runtime-minutes", type=float, default=0,
                        help="Stop cleanly at a batch boundary before a session limit; 0 disables")
    args = parser.parse_args()
    if (args.epochs < 1 or (args.batch_size is not None and args.batch_size < 1)
            or (args.window_frames is not None and args.window_frames < 16)
            or args.checkpoint_every < 1 or args.max_runtime_minutes < 0):
        parser.error("epochs, batch size and checkpoint interval must be positive; runtime must be nonnegative")
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    cache_identity = manifest_identity(manifest)
    advanced = args.variant.startswith("advanced")
    cfg = replace(Config(**manifest["config"]), epochs=args.epochs,
                  batch_size=args.batch_size or (4 if advanced else 16),
                  window_frames=args.window_frames or (256 if advanced else manifest["config"]["window_frames"]))
    task = manifest.get("task", "pitched")
    stem = manifest.get("stem", "guitar")
    if task not in ("pitched", "drums"):
        raise ValueError(f"Unknown training task: {task}")
    if task == "drums" and args.variant not in ("baseline", "advanced"):
        parser.error("Drums support baseline or advanced; harmonic gating is for pitched stems")
    if task == "pitched" and args.variant == "advanced":
        parser.error("Pitched stems use advanced_plain or advanced_gated")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = split_files(args.cache, cfg)
    dataset = WindowDataset(files["train"], cfg)
    model = build_transcriber(task, args.variant, cfg.pitches).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    onset_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0, device=device))
    args.output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_checkpoint = None
    history = []
    checkpoint = args.output / f"{args.variant}.pt"
    last_checkpoint = args.output / f"{args.variant}_last.pt"
    status_path = args.output / f"{args.variant}_status.json"
    start_epoch, start_batch, epoch_loss_sum, epoch_loss_count = 1, 0, 0.0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("checkpoint_format") != CHECKPOINT_FORMAT or saved.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Resume checkpoint format or feature version is incompatible")
        legacy_without_identity = saved.get("cache_identity") is None and stem in ("guitar", "vocals")
        if not legacy_without_identity and saved.get("cache_identity") != cache_identity:
            raise ValueError("Resume checkpoint was trained on a different cache or dataset")
        if saved["variant"] != args.variant:
            raise ValueError("Resume checkpoint belongs to a different variant")
        if saved.get("task", "pitched") != task or saved.get("stem", "guitar") != stem:
            raise ValueError("Resume checkpoint belongs to a different stem or task")
        for key, value in cfg.to_dict().items():
            if key != "epochs" and saved["config"][key] != value:
                raise ValueError(f"Resume configuration differs in {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, start_batch = saved["epoch"], saved["next_batch"]
        epoch_loss_sum, epoch_loss_count = saved["epoch_loss_sum"], saved["epoch_loss_count"]
        best, best_checkpoint = saved["best_f1"], saved["best_checkpoint"]
        history = saved["history"]
        random.setstate(saved["rng_python"])
        np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"].cpu())
        if device.type == "cuda" and saved.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        if best_checkpoint is not None:
            atomic_torch_save(best_checkpoint, checkpoint)
        print(f"Resuming {args.variant} at epoch {start_epoch}, batch {start_batch}", flush=True)

    def save_progress(epoch: int, next_batch: int, loss_sum: float, loss_count: int) -> None:
        payload = {
            "checkpoint_format": CHECKPOINT_FORMAT, "feature_version": FEATURE_VERSION,
            "cache_identity": cache_identity,
            "variant": args.variant, "task": task, "stem": stem,
            "config": cfg.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "next_batch": next_batch,
            "epoch_loss_sum": loss_sum, "epoch_loss_count": loss_count,
            "best_f1": best, "best_checkpoint": best_checkpoint, "history": history,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        atomic_torch_save(payload, last_checkpoint)

    def write_status(state: str, epoch: int, batch: int) -> None:
        status_path.write_text(json.dumps({"status": state, "epoch": epoch,
                                           "next_batch": batch,
                                           "checkpoint": str(last_checkpoint)}, indent=2),
                               encoding="utf-8")

    started = time.monotonic()
    for epoch in range(start_epoch, cfg.epochs + 1):
        order_generator = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(dataset), generator=order_generator).tolist()
        total_batches = (len(order) + cfg.batch_size - 1) // cfg.batch_size
        first_batch = start_batch if epoch == start_epoch else 0
        if first_batch > total_batches:
            raise ValueError("Resume batch exceeds the number of training batches")
        remaining_order = order[first_batch * cfg.batch_size:]
        loader = DataLoader(Subset(dataset, remaining_order), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
        model.train()
        loss_sum = epoch_loss_sum if epoch == start_epoch else 0.0
        loss_count = epoch_loss_count if epoch == start_epoch else 0
        for batch, (x, y, onset) in enumerate(loader, start=first_batch):
            x, y, onset = x.to(device), y.to(device), onset.to(device)
            optimizer.zero_grad(set_to_none=True)
            note_prob, onset_logits = model(x)
            weights = 1.0 + (9.0 if task == "drums" else 4.0) * y
            loss = torch.nn.functional.binary_cross_entropy(note_prob, y, weight=weights)
            if task == "pitched":
                loss = loss + 0.25 * onset_loss(onset_logits, onset)
            loss.backward()
            if advanced:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            loss_count += 1
            next_batch = batch + 1
            timed_out = args.max_runtime_minutes > 0 and (time.monotonic() - started) / 60 >= args.max_runtime_minutes
            if next_batch % args.checkpoint_every == 0 or timed_out:
                save_progress(epoch, next_batch, loss_sum, loss_count)
            if timed_out:
                write_status("paused", epoch, next_batch)
                print(f"Paused at epoch {epoch}, batch {next_batch}; resume from {last_checkpoint}", flush=True)
                return
        val_outputs = predictions(model, files["val"], device)
        options = [(t, summarize(val_outputs, t, cfg, task)) for t in (0.25, 0.35, 0.45, 0.55, 0.65)]
        threshold, val = max(options, key=lambda pair: (pair[1]["note"]["f1"],
                                                          pair[1]["frame"]["f1"]))
        row = {"epoch": epoch, "loss": loss_sum / max(loss_count, 1), "threshold": threshold,
               "val_frame_f1": val["frame"]["f1"], "val_note_f1": val["note"]["f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["note"]["f1"] > best:
            best = val["note"]["f1"]
            best_checkpoint = {"model": copy.deepcopy(model.state_dict()),
                               "config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                               "cache_identity": cache_identity,
                               "variant": args.variant, "task": task, "stem": stem,
                               "threshold": threshold, "epoch": epoch}
            atomic_torch_save(best_checkpoint, checkpoint)
        save_progress(epoch + 1, 0, 0.0, 0)
        write_status("in_progress", epoch + 1, 0)
    if best_checkpoint is None:
        raise ValueError("No completed epoch available for evaluation")
    model.load_state_dict(best_checkpoint["model"])
    threshold = best_checkpoint["threshold"]
    val = summarize(predictions(model, files["val"], device), threshold, cfg, task)
    test = summarize(predictions(model, files["test"], device), threshold, cfg, task)
    result = {"variant": args.variant, "task": task, "stem": stem,
              "best_epoch": best_checkpoint["epoch"], "threshold": threshold,
              "validation": val, "test": test, "history": history,
              "split": manifest.get("split", "dataset-defined train/validation/test")}
    (args.output / f"{args.variant}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "test": test}, indent=2))
    if args.variant in ("baseline", "advanced_plain") and task == "pitched":
        fixed_val_outputs = predictions(model, files["val"], device, fixed_gate=True)
        fixed_options = [(t, summarize(fixed_val_outputs, t, cfg, task)) for t in (0.25, 0.35, 0.45, 0.55, 0.65)]
        fixed_threshold, fixed_val = max(fixed_options, key=lambda pair: (pair[1]["note"]["f1"],
                                                                     pair[1]["frame"]["f1"]))
        fixed_test = summarize(predictions(model, files["test"], device, fixed_gate=True),
                               fixed_threshold, cfg, task)
        prefix = "paper_gate" if args.variant == "baseline" else "advanced_paper_gate"
        fixed_result = {"variant": prefix, "best_epoch": best_checkpoint["epoch"],
                        "threshold": fixed_threshold, "validation": fixed_val, "test": fixed_test,
                        "split": result["split"], "gate_percentile": 10}
        (args.output / f"{prefix}.json").write_text(json.dumps(fixed_result, indent=2), encoding="utf-8")
        fixed_checkpoint = {**best_checkpoint, "threshold": fixed_threshold, "gate_mode": "paper_fixed"}
        atomic_torch_save(fixed_checkpoint, args.output / f"{prefix}.pt")
        print(json.dumps({"variant": prefix, "test": fixed_test}, indent=2))
    write_status("complete", cfg.epochs + 1, 0)


if __name__ == "__main__":
    main()
