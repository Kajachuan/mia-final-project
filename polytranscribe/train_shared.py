"""Train one stem-conditioned pitched transcriber on four balanced datasets."""

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config
from .features import FEATURE_VERSION
from .model import AdvancedPitchTranscriber, build_transcriber
from .train import (CHECKPOINT_FORMAT, WindowDataset, atomic_torch_save,
                    manifest_identity, predictions, split_files, summarize)


STEMS = AdvancedPitchTranscriber.STEMS
THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65)


def load_data(caches: dict[str, Path], batch_size: int, window_frames: int,
              epochs: int, learning_rate: float):
    manifests, configs, splits, datasets = {}, {}, {}, {}
    for stem in STEMS:
        cache = caches[stem]
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("feature_version") != FEATURE_VERSION or manifest.get("stem") != stem:
            raise ValueError(f"Incompatible {stem} cache at {cache}")
        cfg = replace(Config(**manifest["config"]), batch_size=batch_size,
                      window_frames=window_frames, epochs=epochs,
                      learning_rate=learning_rate)
        files = split_files(cache, cfg)
        dataset = WindowDataset(files["train"], cfg)
        if not dataset:
            raise ValueError(f"No training windows for {stem}")
        manifests[stem], configs[stem], splits[stem], datasets[stem] = manifest, cfg, files, dataset
    return manifests, configs, splits, datasets


def shared_cache_identity(manifests: dict, batch_size: int, window_frames: int,
                          steps_per_stem: int, learning_rate: float) -> str:
    fields = {"manifests": {stem: manifest_identity(manifests[stem]) for stem in STEMS},
              "batch_size": batch_size, "window_frames": window_frames,
              "steps_per_stem": steps_per_stem, "learning_rate": learning_rate}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def training_schedule(datasets: dict, steps_per_stem: int, batch_size: int,
                      seed: int, epoch: int):
    """Fixed balanced order makes mid-epoch resume independent of loader state."""
    generator = torch.Generator().manual_seed(seed + epoch)
    for _ in range(steps_per_stem):
        for index in torch.randperm(len(STEMS), generator=generator).tolist():
            stem = STEMS[index]
            choices = torch.randint(len(datasets[stem]), (batch_size,), generator=generator)
            yield stem, choices.tolist()


def evaluate(model, splits, configs, device):
    result, thresholds = {}, {}
    for stem in STEMS:
        outputs = predictions(model, splits[stem]["val"], device, stem=stem)
        threshold, scores = max(((threshold, summarize(outputs, threshold, configs[stem]))
                                 for threshold in THRESHOLDS),
                                key=lambda pair: (pair[1]["note"]["f1"],
                                                  pair[1]["frame"]["f1"]))
        result[stem] = scores
        thresholds[stem] = threshold
    macro = sum(result[stem]["note"]["f1"] for stem in STEMS) / len(STEMS)
    return macro, thresholds, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guitar-cache", type=Path, required=True)
    parser.add_argument("--aux-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("shared_plain", "shared_gated"), required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=256)
    parser.add_argument("--steps-per-stem", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-runtime-minutes", type=float, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.window_frames < 16
            or args.steps_per_stem < 1 or args.learning_rate <= 0
            or args.checkpoint_every < 1 or args.max_runtime_minutes < 0):
        parser.error("Training sizes and learning rate must be positive")
    caches = {stem: args.guitar_cache if stem == "guitar" else args.aux_cache / stem
              for stem in STEMS}
    manifests, configs, splits, datasets = load_data(
        caches, args.batch_size, args.window_frames, args.epochs, args.learning_rate)
    cache_identity = shared_cache_identity(manifests, args.batch_size, args.window_frames,
                                           args.steps_per_stem, args.learning_rate)
    seed = configs["guitar"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transcriber("pitched", args.variant, configs["guitar"].pitches).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    onset_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0, device=device))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.variant}.pt"
    last_checkpoint = args.output / f"{args.variant}_last.pt"
    status_path = args.output / f"{args.variant}_status.json"
    start_epoch, start_step, loss_sum, loss_count = 1, 0, 0.0, 0
    best, best_checkpoint, history = -1.0, None, []
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (saved.get("checkpoint_format") != CHECKPOINT_FORMAT
                or saved.get("feature_version") != FEATURE_VERSION
                or saved.get("cache_identity") != cache_identity
                or saved.get("variant") != args.variant):
            raise ValueError("Shared resume checkpoint differs from this model or dataset")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, start_step = saved["epoch"], saved["next_step"]
        loss_sum, loss_count = saved["loss_sum"], saved["loss_count"]
        best, best_checkpoint, history = saved["best_f1"], saved["best_checkpoint"], saved["history"]
        random.setstate(saved["rng_python"])
        np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"].cpu())
        if device.type == "cuda" and saved.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        if best_checkpoint is not None:
            atomic_torch_save(best_checkpoint, checkpoint)
        print(f"Resuming {args.variant} at epoch {start_epoch}, step {start_step}", flush=True)

    def save_progress(epoch, next_step, running_loss, batches):
        atomic_torch_save({
            "checkpoint_format": CHECKPOINT_FORMAT, "feature_version": FEATURE_VERSION,
            "cache_identity": cache_identity, "variant": args.variant,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "next_step": next_step,
            "loss_sum": running_loss, "loss_count": batches,
            "best_f1": best, "best_checkpoint": best_checkpoint, "history": history,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, last_checkpoint)

    def write_status(state, epoch, next_step):
        status_path.write_text(json.dumps({"status": state, "epoch": epoch,
                                           "next_step": next_step,
                                           "checkpoint": str(last_checkpoint)}, indent=2),
                               encoding="utf-8")

    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        schedule = list(training_schedule(datasets, args.steps_per_stem,
                                          args.batch_size, seed, epoch))
        first_step = start_step if epoch == start_epoch else 0
        if first_step > len(schedule):
            raise ValueError("Resume step exceeds the epoch schedule")
        model.train()
        running_loss = loss_sum if epoch == start_epoch else 0.0
        batches = loss_count if epoch == start_epoch else 0
        for step in range(first_step, len(schedule)):
            stem, indices = schedule[step]
            examples = [datasets[stem][i] for i in indices]
            x, y, onset = (torch.stack([example[part] for example in examples]).to(device)
                           for part in range(3))
            optimizer.zero_grad(set_to_none=True)
            note_prob, onset_logits = model(x, stem=stem)
            loss = torch.nn.functional.binary_cross_entropy(
                note_prob, y, weight=1.0 + 4.0 * y)
            loss = loss + 0.25 * onset_loss(onset_logits, onset)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
            next_step = step + 1
            timed_out = (args.max_runtime_minutes > 0 and
                         (time.monotonic() - started) / 60 >= args.max_runtime_minutes)
            if next_step % args.checkpoint_every == 0 or timed_out:
                save_progress(epoch, next_step, running_loss, batches)
            if timed_out:
                write_status("paused", epoch, next_step)
                print(f"Paused at epoch {epoch}, step {next_step}", flush=True)
                return
        model.eval()
        macro, thresholds, validation = evaluate(model, splits, configs, device)
        row = {"epoch": epoch, "loss": running_loss / max(1, batches),
               "val_macro_note_f1": macro,
               "val_note_f1_by_stem": {stem: validation[stem]["note"]["f1"] for stem in STEMS}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if macro > best:
            best = macro
            best_checkpoint = {"model": copy.deepcopy(model.state_dict()),
                               "feature_version": FEATURE_VERSION, "task": "pitched",
                               "variant": args.variant, "stem": "shared",
                               "configs": {stem: configs[stem].to_dict() for stem in STEMS},
                               "thresholds": thresholds, "cache_identity": cache_identity,
                               "epoch": epoch}
            atomic_torch_save(best_checkpoint, checkpoint)
        save_progress(epoch + 1, 0, 0.0, 0)
        write_status("in_progress", epoch + 1, 0)
    if best_checkpoint is None:
        raise ValueError("No completed epoch available")
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    _, _, validation = evaluate(model, splits, configs, device)
    scores = {}
    for stem in STEMS:
        outputs = predictions(model, splits[stem]["test"], device, stem=stem)
        scores[stem] = summarize(outputs, best_checkpoint["thresholds"][stem], configs[stem])
    report = {"variant": args.variant, "best_epoch": best_checkpoint["epoch"],
              "validation_macro_note_f1": best, "validation": validation,
              "test": scores, "history": history, "thresholds": best_checkpoint["thresholds"]}
    (args.output / f"{args.variant}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant,
                      "test_note_f1": {stem: scores[stem]["note"]["f1"] for stem in STEMS}},
                     indent=2), flush=True)
    write_status("complete", args.epochs + 1, 0)


if __name__ == "__main__":
    main()
