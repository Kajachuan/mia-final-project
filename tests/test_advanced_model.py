import torch
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from polytranscribe.model import (AdvancedDrumTranscriber, AdvancedPitchTranscriber,
                                   build_transcriber, predict_in_chunks)
from polytranscribe.config import Config
from polytranscribe.features import FEATURE_VERSION


def test_advanced_pitch_model_and_gate_ablation():
    torch.manual_seed(7)
    audio = torch.randn(1, 10, 43, 48)
    changed_gate = audio.clone()
    changed_gate[:, 3:6] += 3
    plain = AdvancedPitchTranscriber(gated=False).eval()
    gated = AdvancedPitchTranscriber(gated=True).eval()
    with torch.no_grad():
        plain_before, onset = plain(audio)
        plain_after, _ = plain(changed_gate)
        gated_before, _ = gated(audio)
        gated_after, _ = gated(changed_gate)
    assert plain_before.shape == onset.shape == (1, 43, 48)
    assert torch.allclose(plain_before, plain_after)
    assert not torch.allclose(gated_before, gated_after)


def test_advanced_model_training_and_chunked_inference():
    model = build_transcriber("pitched", "advanced_gated", 43)
    model.train()
    notes, onsets = model(torch.randn(1, 10, 43, 32))
    notes.mean().backward()
    assert model.note_head.weight.grad is not None
    assert model.onset_head.weight.grad is not None
    assert model.gate[-1].weight.grad is not None
    model.eval()
    with torch.no_grad():
        notes, onsets = predict_in_chunks(model, torch.randn(1, 10, 43, 300),
                                          torch.device("cpu"))
    assert notes.shape == onsets.shape == (1, 43, 300)
    assert torch.isfinite(notes).all()


def test_advanced_drum_model():
    model = build_transcriber("drums", "advanced", 9)
    assert isinstance(model, AdvancedDrumTranscriber)
    probabilities, logits = model(torch.randn(1, 1, 128, 48))
    assert probabilities.shape == logits.shape == (1, 9, 48)
    probabilities.mean().backward()
    assert model.head.weight.grad is not None


def test_advanced_training_and_paper_gate_checkpoint(tmp_path, monkeypatch):
    from polytranscribe import train
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg = Config()
    file_splits = {}
    for split in ("train", "val", "test"):
        name = f"{split}_track"
        file_splits[name] = split
        x = np.zeros((10, cfg.pitches, 32), dtype=np.float16)
        x[5, :, 5:] = 1
        y = np.zeros((cfg.pitches, 32), dtype=np.uint8)
        y[0, 8:20] = 1
        np.savez(cache / f"{name}.npz", x=x, y=y, onset=y,
                 events=np.asarray([(8 * cfg.frame_seconds, 20 * cfg.frame_seconds, 40)],
                                   dtype=np.float32))
    manifest = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                "task": "pitched", "stem": "guitar", "file_splits": file_splits,
                "counts": {"train": 1, "val": 1, "test": 1}}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(cache), "--output", str(output),
                                      "--variant", "advanced_plain", "--epochs", "1",
                                      "--batch-size", "1", "--window-frames", "32"])
    train.main()
    assert (output / "advanced_plain.pt").exists()
    assert (output / "advanced_paper_gate.pt").exists()
    saved = torch.load(output / "advanced_plain_last.pt", weights_only=False)
    assert saved["checkpoint_format"] == train.CHECKPOINT_FORMAT
    assert saved["cache_identity"] == train.manifest_identity(manifest)
    manifest["source"] = "another-dataset"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(cache), "--output", str(output),
                                      "--variant", "advanced_plain", "--epochs", "2",
                                      "--batch-size", "1", "--window-frames", "32",
                                      "--resume", str(output / "advanced_plain_last.pt")])
    with pytest.raises(ValueError, match="different cache or dataset"):
        train.main()


def test_external_midi_benchmark_uses_shared_metrics(tmp_path, monkeypatch):
    from polytranscribe import benchmark_external
    from polytranscribe.infer import write_midi

    annotation = tmp_path / "05_example.jams"
    annotation.write_text(json.dumps({"annotations": [
        {"namespace": "note_midi", "data": [{"time": 0.1, "duration": 0.2, "value": 40}]},
        *({"namespace": "note_midi", "data": []} for _ in range(5))]}), encoding="utf-8")
    audio = tmp_path / "05_example_mic.wav"
    sf.write(audio, np.zeros(22050, dtype=np.float32), 22050)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    write_midi([(0.1, 0.3, 40)], prediction_dir / "05_example.mid")
    monkeypatch.setattr(benchmark_external, "find_pairs", lambda root: [(annotation, audio, "test")])
    report = benchmark_external.evaluate(tmp_path, prediction_dir, tmp_path / "scores.json")
    assert report["scores"]["note"]["f1"] == 1.0
