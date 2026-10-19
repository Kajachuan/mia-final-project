import numpy as np
import soundfile as sf
import torch

from polytranscribe.config import Config
from polytranscribe.features import load_audio, pitch_features
from polytranscribe.infer import write_midi
from polytranscribe.metrics import paper_voicing_gate
from polytranscribe.model import PolyphonicTranscriber
from polytranscribe import pipeline, separate


def test_audio_to_model_and_midi(tmp_path):
    cfg = Config()
    sr = cfg.sample_rate
    time = np.arange(sr, dtype=np.float32) / sr
    chord = 0.2 * np.sin(2 * np.pi * 82.4 * time) + 0.2 * np.sin(2 * np.pi * 123.47 * time)
    audio_path = tmp_path / "chord.wav"
    sf.write(audio_path, chord, sr)
    audio = load_audio(audio_path, cfg)
    x = pitch_features(audio, cfg, torch.device("cpu"))
    assert x.shape == (6, cfg.pitches, 1 + sr // cfg.hop_length)
    assert np.isfinite(x).all()
    batch = torch.from_numpy(x.astype(np.float32))[None]
    for variant in ("baseline", "gated"):
        model = PolyphonicTranscriber(variant)
        note, onset = model(batch)
        assert note.shape == onset.shape == (1, cfg.pitches, x.shape[-1])
        assert ((note > 0) & (note < 1)).all()
        note.mean().backward()
        assert model.note_head.weight.grad is not None
        if variant == "gated":
            assert model.gate_head.weight.grad is not None
    fixed = paper_voicing_gate(note[0].detach().numpy(), x.astype(np.float32))
    assert fixed.shape == (cfg.pitches, x.shape[-1])
    assert (fixed == 0).any()
    midi_path = tmp_path / "chord.mid"
    write_midi([(0.1, 0.5, 40), (0.1, 0.5, 47)], midi_path)
    import mido
    midi = mido.MidiFile(midi_path)
    assert sum(message.type == "note_on" for message in midi.tracks[0]) == 2


def test_separation_precedes_transcription(tmp_path, monkeypatch):
    mixture = tmp_path / "song.wav"
    sf.write(mixture, np.zeros(4410, dtype=np.float32), 44100)
    checkpoint = tmp_path / "gated.pt"
    checkpoint.write_bytes(b"fixture")
    calls = []

    class FakeModel:
        sources = separate.STEMS
        samplerate = 44100
        audio_channels = 2

        def eval(self):
            return self

    def fake_get_model(name):
        assert name == "htdemucs_6s"
        calls.append("separate")
        return FakeModel()

    def fake_apply_model(model, mix, **kwargs):
        assert mix.shape == (1, 2, 4410)
        return torch.zeros((1, 6, 2, 4410))

    def fake_transcribe(audio, ckpt, midi):
        assert calls == ["separate"]
        assert audio.name == "guitar.wav"
        assert ckpt == checkpoint
        calls.append("transcribe")
        midi.write_bytes(b"midi")
        return 2

    monkeypatch.setattr(separate, "get_model", fake_get_model)
    monkeypatch.setattr(separate, "apply_model", fake_apply_model)
    monkeypatch.setattr(pipeline, "transcribe_stem", fake_transcribe)
    result = pipeline.transcribe_mix(mixture, checkpoint, tmp_path / "result")
    assert calls == ["separate", "transcribe"]
    assert result["note_count"] == 2
    assert result["transcribed_stem"] == "guitar"
    assert result["separator"] == "htdemucs_6s"
    assert set(result["stems"]) == set(separate.STEMS)
    assert (tmp_path / "result" / "guitar.mid").exists()
    assert (tmp_path / "result" / "pipeline.json").exists()


def test_mel_chroma_branch_changes_only_gated_model():
    torch.manual_seed(3)
    inputs = torch.randn(1, 6, 43, 24)
    changed = inputs.clone()
    changed[:, 3:] += 2.0
    baseline = PolyphonicTranscriber("baseline").eval()
    gated = PolyphonicTranscriber("gated").eval()
    with torch.no_grad():
        baseline_before, _ = baseline(inputs)
        baseline_after, _ = baseline(changed)
        gated_before, _ = gated(inputs)
        gated_after, _ = gated(changed)
    assert torch.allclose(baseline_before, baseline_after)
    assert not torch.allclose(gated_before, gated_after)


def test_baseline_training_resumes_from_batch_and_emits_paper_gate(tmp_path, monkeypatch):
    import json
    import sys
    from polytranscribe import train

    files = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.npz"
        frames = 560 if split == "train" else 48
        features = np.zeros((6, 43, frames), dtype=np.float16)
        features[5, :, 8:] = 1.0
        notes = np.zeros((43, frames), dtype=np.uint8)
        notes[0, 10:30] = 1
        onsets = np.zeros_like(notes)
        onsets[0, 9:12] = 1
        np.savez(path, x=features, y=notes, onset=onsets,
                 events=np.array([[10 * Config().frame_seconds, 30 * Config().frame_seconds, 40]],
                                 dtype=np.float32))
        files[split] = [path]
    monkeypatch.setattr(train, "split_files", lambda cache, cfg: files)
    output = tmp_path / "first_run"
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(tmp_path),
                                      "--output", str(output), "--variant", "baseline",
                                      "--epochs", "1", "--batch-size", "1",
                                      "--checkpoint-every", "1", "--max-runtime-minutes", "0.000000001"])
    train.main()
    resume = output / "baseline_last.pt"
    saved = torch.load(resume, weights_only=False)
    assert (saved["epoch"], saved["next_batch"]) == (1, 1)
    assert saved["optimizer"]["state"]

    continued = tmp_path / "continued_run"
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(tmp_path),
                                      "--output", str(continued), "--variant", "baseline",
                                      "--epochs", "1", "--batch-size", "1", "--resume", str(resume)])
    train.main()
    assert (continued / "baseline.pt").exists()
    assert (continued / "paper_gate.pt").exists()
    assert json.loads((continued / "paper_gate.json").read_text())["variant"] == "paper_gate"
    assert json.loads((continued / "baseline_status.json").read_text())["status"] == "complete"

    uninterrupted = tmp_path / "uninterrupted_run"
    monkeypatch.setattr(sys, "argv", ["train", "--cache", str(tmp_path),
                                      "--output", str(uninterrupted), "--variant", "baseline",
                                      "--epochs", "1", "--batch-size", "1"])
    train.main()
    resumed_weights = torch.load(continued / "baseline.pt", weights_only=False)["model"]
    full_weights = torch.load(uninterrupted / "baseline.pt", weights_only=False)["model"]
    assert all(torch.allclose(resumed_weights[name], full_weights[name]) for name in full_weights)
