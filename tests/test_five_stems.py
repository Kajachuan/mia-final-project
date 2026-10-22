import json
import sys

import mido
import numpy as np
import soundfile as sf
import torch
import pytest
import yaml

from polytranscribe import prepare_five, train
from polytranscribe.features import FEATURE_VERSION, drum_features, pitch_features
from polytranscribe.infer import write_midi
from polytranscribe.metrics import decode_drums
from polytranscribe.model import DrumTranscriber
from polytranscribe.stems import DRUM_MIDI, TARGET_STEMS, config_for_stem


def test_stem_configs_and_drum_decoder():
    assert set(TARGET_STEMS) == {"vocals", "drums", "bass", "guitar", "piano"}
    assert config_for_stem("vocals").midi_min == 36
    assert config_for_stem("bass").midi_min == 28
    assert config_for_stem("piano").midi_max == 108
    cfg = config_for_stem("drums")
    probabilities = np.zeros((len(DRUM_MIDI), 32), dtype=np.float32)
    probabilities[0, 10] = probabilities[1, 10] = 0.9
    events = decode_drums(probabilities, cfg)
    assert len(events) == 2
    assert {event[2] for event in events} == {36, 38}


def test_drum_midi_uses_percussion_channel(tmp_path):
    path = tmp_path / "drums.mid"
    write_midi([(0.1, 0.2, 36)], path, stem="drums")
    midi = mido.MidiFile(path)
    note_on = next(message for message in midi.tracks[0] if message.type == "note_on")
    assert note_on.channel == 9


def test_pitched_and_drum_features_and_model():
    seconds = 0.25
    cfg = config_for_stem("piano")
    audio = torch.randn(round(cfg.sample_rate * seconds)) * 0.01
    x = pitch_features(audio, cfg, torch.device("cpu"))
    assert x.shape[0:2] == (6, cfg.pitches)
    assert np.isfinite(x).all()
    drums_cfg = config_for_stem("drums")
    mel = drum_features(audio, drums_cfg, torch.device("cpu"))
    assert mel.shape[0:2] == (1, drums_cfg.n_mels)
    model = DrumTranscriber()
    probabilities, logits = model(torch.from_numpy(mel.astype(np.float32))[None])
    assert probabilities.shape == logits.shape == (1, len(DRUM_MIDI), mel.shape[-1])


def test_vocadito_note_format(tmp_path):
    note_file = tmp_path / "vocadito_1_notesA1.csv"
    note_file.write_text("0.1,440.0,0.2\n0.5,220.0,0.1\n", encoding="utf-8")
    events = prepare_five.vocadito_notes(note_file)
    assert [event[2] for event in events] == [69, 57]
    assert [event[0] for event in events] == pytest.approx([0.1, 0.5])
    assert [event[1] for event in events] == pytest.approx([0.3, 0.6])


def test_prepare_slakh_instrument_caches(tmp_path):
    root = tmp_path / "slakh"
    for split in ("train", "validation", "test"):
        track = root / split / f"Track_{split}"
        (track / "stems").mkdir(parents=True)
        (track / "MIDI").mkdir()
        metadata = {"audio_dir": "stems", "midi_dir": "MIDI", "stems": {}}
        for number, (stem, midi_pitch, program) in enumerate(
                (("Bass", 40, 33), ("Piano", 60, 0), ("Drums", 36, 0)), 1):
            stem_id = f"S{number:02d}"
            sf.write(track / "stems" / f"{stem_id}.wav", np.zeros(5000, dtype=np.float32), 22050)
            write_midi([(0.03, 0.13, midi_pitch)], track / "MIDI" / f"{stem_id}.mid")
            metadata["stems"][stem_id] = {"inst_class": stem, "is_drum": stem == "Drums",
                                           "program_num": program,
                                           "audio_rendered": False, "midi_saved": False}
        (track / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    output = tmp_path / "cache"
    manifests = prepare_five.prepare_slakh(root, output, 0, torch.device("cpu"))
    for stem in ("bass", "piano", "drums"):
        assert manifests[stem]["counts"] == {"train": 1, "val": 1, "test": 1}
        assert len(list((output / stem).glob("*.npz"))) == 3


def test_drum_training_checkpoint(tmp_path, monkeypatch):
    cache = tmp_path / "drums"
    cache.mkdir()
    cfg = config_for_stem("drums")
    file_splits = {}
    for split in ("train", "val", "test"):
        name = f"{split}_drum"
        file_splits[name] = split
        x = np.zeros((1, cfg.n_mels, 32), dtype=np.float16)
        y = np.zeros((len(DRUM_MIDI), 32), dtype=np.uint8)
        y[0, 10] = 1
        np.savez(cache / f"{name}.npz", x=x, y=y, onset=y,
                 events=np.asarray([(10 * cfg.frame_seconds, 10 * cfg.frame_seconds + 0.1, 36)],
                                   dtype=np.float32))
    manifest = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                "task": "drums", "stem": "drums", "file_splits": file_splits,
                "counts": {"train": 1, "val": 1, "test": 1}}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(cache), "--output", str(output),
                                      "--variant", "baseline", "--epochs", "1", "--batch-size", "1"])
    train.main()
    saved = torch.load(output / "baseline.pt", weights_only=False)
    assert saved["task"] == saved["stem"] == "drums"
    assert not (output / "paper_gate.pt").exists()
