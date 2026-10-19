import json

import numpy as np

from polytranscribe.config import Config
from polytranscribe.data import frame_targets, note_events, player_split
from polytranscribe.metrics import decode_notes, frame_scores, note_scores


def test_six_string_jams_and_polyphonic_targets(tmp_path):
    annotations = [{"namespace": "note_midi", "data": []} for _ in range(6)]
    annotations[0]["data"] = [{"time": 0.1, "duration": 0.3, "value": 40.02}]
    annotations[1]["data"] = [{"time": 0.1, "duration": 0.3, "value": 47.01}]
    path = tmp_path / "00_example.jams"
    path.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")
    cfg = Config()
    events = note_events(path)
    notes, onsets = frame_targets(events, 100, cfg)
    assert len(events) == 2
    assert (notes.sum(axis=0) == 2).any()
    assert onsets.sum() >= 2
    assert player_split(path.stem) == "train"
    assert player_split("04_example") == "val"
    assert player_split("05_example") == "test"


def test_decode_preserves_chord_and_note_matching():
    cfg = Config()
    probabilities = np.zeros((cfg.pitches, 100), dtype=np.float32)
    onsets = np.zeros_like(probabilities)
    probabilities[0, 10:35] = 0.9
    probabilities[7, 10:35] = 0.9
    onsets[0, 10] = onsets[7, 10] = 0.9
    events = decode_notes(probabilities, onsets, cfg, threshold=0.5)
    assert len(events) == 2
    assert {event[2] for event in events} == {40, 47}
    assert note_scores(events, events)["f1"] == 1.0
    assert frame_scores(probabilities >= 0.5, probabilities >= 0.5)["f1"] == 1.0


def test_note_metric_rejects_wrong_offset():
    reference = [(0.1, 0.5, 40)]
    prediction = [(0.1, 0.8, 40)]
    assert note_scores(prediction, reference)["f1"] == 0.0
