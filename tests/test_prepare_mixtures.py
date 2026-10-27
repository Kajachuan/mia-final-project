import json

import numpy as np
import soundfile as sf
import torch
import yaml

from polytranscribe import prepare_five, prepare_mixtures
from polytranscribe.infer import write_midi
from polytranscribe.stems import config_for_stem


def test_mixture_cache_uses_demucs_estimates_and_can_resume(tmp_path, monkeypatch):
    slakh = tmp_path / "slakh"
    for split in ("train", "validation", "test"):
        track = slakh / split / f"Track_{split}"
        (track / "MIDI").mkdir(parents=True)
        sf.write(track / "mix.wav", np.full(6615, 0.1, dtype=np.float32), 22050)
        metadata = {"midi_dir": "MIDI", "stems": {}}
        for index, (stem, pitch) in enumerate(
                (("Guitar", 40), ("Bass", 36), ("Piano", 60), ("Drums", 36)), 1):
            source = f"S{index:02d}"
            write_midi([(0.05, 0.20, pitch)], track / "MIDI" / f"{source}.mid")
            metadata["stems"][source] = {"inst_class": stem, "is_drum": stem == "Drums"}
        (track / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    vocals = tmp_path / "vocadito"
    vocals.mkdir()
    for index in range(1, 41):
        name = f"vocadito_{index}"
        sf.write(vocals / f"{name}.wav", np.full(4410, 0.1, dtype=np.float32), 22050)
        (vocals / f"{name}_notesA1.csv").write_text("0.05,440.0,0.1\n", encoding="utf-8")
    calls = []

    class FakeSeparator:
        samplerate = 22050

        def eval(self):
            return self

    def fake_separate(model, waveform, rate, device):
        calls.append((len(waveform), rate))
        source = torch.from_numpy(waveform.T.copy())
        return {stem: source for stem in ("vocals", "drums", "bass", "other", "guitar", "piano")}

    def fake_pitch(audio, cfg, device):
        return np.zeros((10, cfg.pitches, 32), dtype=np.float16)

    def fake_drums(audio, cfg, device):
        return np.zeros((1, cfg.n_mels, 32), dtype=np.float16)

    monkeypatch.setattr(prepare_mixtures, "separate_waveform", fake_separate)
    monkeypatch.setattr(prepare_five, "pitch_features", fake_pitch)
    monkeypatch.setattr(prepare_five, "drum_features", fake_drums)
    output = tmp_path / "mixed"
    prepare_mixtures.prepare(slakh, vocals, output, 0, 0.3,
                             separator=FakeSeparator(), device=torch.device("cpu"))
    assert len(calls) == 43  # three Slakh mixes and 40 vocal plus accompaniment mixes
    for stem in ("vocals", "bass", "guitar", "piano", "drums"):
        manifest = json.loads((output / stem / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["input_domain"] == "demucs_estimate_from_mix"
        assert all(manifest["counts"][split] > 0 for split in ("train", "val", "test"))
    with np.load(output / "guitar" / "train_Track_train.npz") as archive:
        assert archive["x"].shape[0] == 10
        assert archive["events"][0, 2] == config_for_stem("guitar").midi_min
    with np.load(output / "vocals" / "train_vocadito_1.npz") as archive:
        assert archive["events"][0, 2] == 69
    continued = tmp_path / "continued"
    prepare_mixtures.prepare(slakh, vocals, continued, 0, 0.3, resume_root=output,
                             separator=FakeSeparator(), device=torch.device("cpu"))
    assert len(calls) == 43
    assert (continued / "piano" / "test_Track_test.npz").is_file()
    paused = tmp_path / "paused"
    prepare_mixtures.prepare(slakh, vocals, paused, 0, 0.3,
                             separator=FakeSeparator(), device=torch.device("cpu"),
                             max_runtime_minutes=1e-9)
    assert json.loads((paused / "mix_status.json").read_text())["status"] == "paused"
    prepare_mixtures.prepare(slakh, vocals, paused, 0, 0.3, resume_root=output,
                             separator=FakeSeparator(), device=torch.device("cpu"))
    assert json.loads((paused / "mix_status.json").read_text())["status"] == "complete"
