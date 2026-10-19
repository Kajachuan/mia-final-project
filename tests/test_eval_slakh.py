import json
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
import torch
import yaml

from polytranscribe import eval_slakh
from polytranscribe.config import Config
from polytranscribe.features import FEATURE_VERSION
from polytranscribe.infer import write_midi
from polytranscribe.metrics import frame_scores_by_polyphony, note_scores


def test_midi_reader_honors_tempo_changes(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=40, velocity=80, time=480))
    track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))
    track.append(mido.Message("note_on", note=40, velocity=0, time=480))
    path = tmp_path / "tempo.mid"
    midi.save(path)
    start, end, pitch = eval_slakh.midi_events(path)[0]
    assert (start, end, pitch) == (0.5, 1.5, 40)


def test_note_metrics_report_onset_and_polyphonic_frames():
    reference = [(0.1, 0.5, 40), (0.1, 0.5, 47)]
    predicted = [(0.1, 0.8, 40), (0.1, 0.5, 47)]
    assert note_scores(predicted, reference)["f1"] == 0.5
    assert note_scores(predicted, reference, offset_ratio=None)["f1"] == 1.0
    target = np.array([[True, False], [True, True]])
    estimate = np.array([[True, False], [False, True]])
    score = frame_scores_by_polyphony(estimate, target)
    assert score["frames"] == 1
    assert score["recall"] == 0.5


def test_slakh_evaluation_scores_oracle_and_separated_audio(tmp_path, monkeypatch):
    root = tmp_path / "slakh"
    track = root / "test" / "Track00001"
    (track / "stems").mkdir(parents=True)
    (track / "MIDI").mkdir()
    sr = 16000
    time = np.arange(sr, dtype=np.float32) / sr
    guitar = 0.1 * np.sin(2 * np.pi * 220 * time)
    piano = 0.1 * np.sin(2 * np.pi * 440 * time)
    sf.write(track / "stems" / "S01.wav", guitar, sr)
    sf.write(track / "stems" / "S02.wav", piano, sr)
    sf.write(track / "mix.wav", guitar + piano, sr)
    write_midi([(0.1, 0.4, 40)], track / "MIDI" / "S01.mid")
    metadata = {"audio_dir": "stems", "midi_dir": "MIDI", "stems": {
        "S01": {"inst_class": "Guitar", "is_drum": False},
        "S02": {"inst_class": "Piano", "is_drum": False}}}
    (track / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": Config().to_dict(), "feature_version": FEATURE_VERSION}, checkpoint)

    def fake_transcribe(audio: Path, model: Path, midi: Path):
        write_midi([(0.1, 0.4, 40)], midi)
        return 1

    def fake_separate(audio: Path, output: Path):
        guitar_path = output / eval_slakh.MODEL / audio.stem / "guitar.wav"
        guitar_path.parent.mkdir(parents=True)
        sf.write(guitar_path, guitar[:sf.info(audio).frames], sr)
        return {"guitar": guitar_path}

    monkeypatch.setattr(eval_slakh, "transcribe_stem", fake_transcribe)
    monkeypatch.setattr(eval_slakh, "separate", fake_separate)
    output = tmp_path / "results"
    result = eval_slakh.evaluate(root, checkpoint, output, "test", 0, True, seconds=0.5)
    assert result["tracks_evaluated"] == 1
    assert result["oracle"]["note"]["f1"] == 1.0
    assert result["separated"]["note"]["f1"] == 1.0
    assert isinstance(result["optional_separation_si_sdr_db_median"], float)
    assert json.loads((output / "summary.json").read_text())["tracks_evaluated"] == 1
