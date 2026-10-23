import json
import sys

import numpy as np
import soundfile as sf
import torch

from polytranscribe import pipeline, train_shared
from polytranscribe.features import FEATURE_VERSION
from polytranscribe.infer import transcribe_stem
from polytranscribe.model import build_transcriber, predict_in_chunks
from polytranscribe.stems import config_for_stem


def test_shared_model_conditions_on_stem_and_supports_pitch_ranges():
    model = build_transcriber("pitched", "shared_gated", 43).eval()
    x = torch.randn(1, 10, 43, 32)
    with torch.no_grad():
        guitar, _ = model(x, stem="guitar")
        bass, _ = model(x, stem="bass")
        longer, _ = predict_in_chunks(model, torch.randn(1, 10, 88, 270),
                                      torch.device("cpu"), stem="piano")
    assert guitar.shape == bass.shape == (1, 43, 32)
    assert not torch.allclose(guitar, bass)
    assert longer.shape == (1, 88, 270)


def test_shared_training_resume_and_pipeline_checkpoint_selection(tmp_path, monkeypatch):
    guitar_cache = tmp_path / "guitar"
    aux_cache = tmp_path / "aux"
    caches = {stem: guitar_cache if stem == "guitar" else aux_cache / stem
              for stem in train_shared.STEMS}
    for stem, cache in caches.items():
        cache.mkdir(parents=True)
        cfg = config_for_stem(stem)
        splits = {}
        for split in ("train", "val", "test"):
            name = f"{split}_{stem}"
            splits[name] = split
            x = np.zeros((10, cfg.pitches, 16), dtype=np.float16)
            x[0, 0, 3:10] = 1
            y = np.zeros((cfg.pitches, 16), dtype=np.uint8)
            y[0, 3:10] = 1
            onset = np.zeros_like(y)
            onset[0, 3:5] = 1
            np.savez(cache / f"{name}.npz", x=x, y=y, onset=onset,
                     events=np.array([[3 * cfg.frame_seconds, 10 * cfg.frame_seconds,
                                       cfg.midi_min]], dtype=np.float32))
        manifest = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                    "stem": stem, "task": "pitched", "source": stem,
                    "counts": {split: 1 for split in ("train", "val", "test")},
                    "file_splits": splits}
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "models"
    base = ["train_shared", "--guitar-cache", str(guitar_cache),
            "--aux-cache", str(aux_cache), "--output", str(output),
            "--variant", "shared_gated", "--epochs", "1", "--batch-size", "1",
            "--window-frames", "16", "--steps-per-stem", "1",
            "--checkpoint-every", "1"]
    monkeypatch.setattr(sys, "argv", base + ["--max-runtime-minutes", "0.000000001"])
    train_shared.main()
    resume = output / "shared_gated_last.pt"
    assert torch.load(resume, weights_only=False)["next_step"] == 1
    monkeypatch.setattr(sys, "argv", base + ["--resume", str(resume)])
    train_shared.main()
    saved = torch.load(output / "shared_gated.pt", weights_only=False)
    assert saved["stem"] == "shared"
    assert set(saved["configs"]) == set(train_shared.STEMS)
    assert set(saved["thresholds"]) == set(train_shared.STEMS)
    assert json.loads((output / "shared_gated_status.json").read_text())["status"] == "complete"
    uninterrupted = tmp_path / "uninterrupted"
    full_command = base.copy()
    full_command[full_command.index(str(output))] = str(uninterrupted)
    monkeypatch.setattr(sys, "argv", full_command)
    train_shared.main()
    full_weights = torch.load(uninterrupted / "shared_gated.pt", weights_only=False)["model"]
    assert all(torch.allclose(saved["model"][name], full_weights[name]) for name in full_weights)
    drums = output / "drums"
    drums.mkdir()
    torch.save({"stem": "drums", "task": "drums"}, drums / "advanced.pt")
    paths = pipeline.checkpoint_paths(output)
    assert all(paths[stem] == output / "shared_gated.pt" for stem in train_shared.STEMS)
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(10000, dtype=np.float32), 22050)
    for stem in ("guitar", "piano"):
        midi = tmp_path / f"{stem}.mid"
        transcribe_stem(audio, output / "shared_gated.pt", midi, stem)
        assert midi.exists()
