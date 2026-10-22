"""Frame and note metrics with explicit tolerances."""

import numpy as np
import mir_eval

from .config import Config
from .stems import DRUM_MIDI


def paper_voicing_gate(note_prob: np.ndarray, features: np.ndarray,
                       percentile: float = 10.0) -> np.ndarray:
    """Paper-style global Mel/chroma voicing mask applied to every candidate pitch."""
    if features.shape[0] != 6:
        raise ValueError("Expected six feature maps")
    energy = features[5, 0]
    cutoff = np.percentile(energy, percentile)
    voiced = energy > cutoff
    return note_prob * voiced[None, :]


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def frame_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall), "tp": tp, "fp": fp, "fn": fn}


def frame_scores_by_polyphony(pred: np.ndarray, target: np.ndarray,
                              minimum_notes: int = 2) -> dict[str, float]:
    """Score only frames whose reference contains at least minimum_notes pitches."""
    chosen = target.sum(axis=0) >= minimum_notes
    return {**frame_scores(pred[:, chosen], target[:, chosen]), "frames": int(chosen.sum())}


def decode_notes(note_prob: np.ndarray, onset_prob: np.ndarray, cfg: Config,
                 threshold: float = 0.5, onset_threshold: float = 0.5,
                 min_duration: float = 0.07, fill_gap: float = 0.035) -> list[tuple[float, float, int]]:
    """Decode overlapping pitches independently; onset peaks can split a repeated note."""
    events = []
    max_gap = round(fill_gap / cfg.frame_seconds)
    min_frames = max(1, round(min_duration / cfg.frame_seconds))
    for pitch_index in range(note_prob.shape[0]):
        active = note_prob[pitch_index] >= threshold
        # Close only short interior gaps.
        false = np.flatnonzero(np.diff(np.r_[False, ~active, False].astype(int)))
        for start, end in false.reshape(-1, 2):
            if start > 0 and end < len(active) and end - start <= max_gap:
                active[start:end] = True
        edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int))).reshape(-1, 2)
        for start, end in edges:
            candidates = np.flatnonzero(onset_prob[pitch_index, start + 1:end] >= onset_threshold) + start + 1
            # One local maximum per nearby onset region.
            splits = []
            for index in candidates:
                if index - start >= min_frames and end - index >= min_frames:
                    if not splits or index - splits[-1] > 3:
                        splits.append(int(index))
            boundaries = [int(start), *splits, int(end)]
            for left, right in zip(boundaries, boundaries[1:]):
                if right - left >= min_frames:
                    events.append((left * cfg.frame_seconds, right * cfg.frame_seconds,
                                   pitch_index + cfg.midi_min))
    return sorted(events)


def decode_drums(probabilities: np.ndarray, cfg: Config,
                 threshold: float = 0.5, minimum_gap: float = 0.05) -> list[tuple[float, float, int]]:
    """Local onset peaks per drum class, with overlapping hits across classes."""
    events = []
    gap_frames = max(1, round(minimum_gap / cfg.frame_seconds))
    for drum_class, row in enumerate(probabilities):
        candidates = np.flatnonzero(row >= threshold)
        peaks = [i for i in candidates if row[i] >= row[max(0, i - 1)]
                 and row[i] >= row[min(len(row) - 1, i + 1)]]
        last = -gap_frames
        for index in peaks:
            if index - last < gap_frames:
                if events and row[index] > row[last] and events[-1][2] == DRUM_MIDI[drum_class]:
                    events[-1] = (index * cfg.frame_seconds,
                                  index * cfg.frame_seconds + 0.1,
                                  DRUM_MIDI[drum_class])
                    last = index
                continue
            start = index * cfg.frame_seconds
            events.append((start, start + 0.1, DRUM_MIDI[drum_class]))
            last = index
    return sorted(events)


def note_scores(predicted: list[tuple[float, float, int]],
                reference: list[tuple[float, float, int]],
                onset_tolerance: float = 0.05,
                offset_ratio: float | None = 0.2) -> dict[str, float]:
    """Standard mir_eval one-to-one pitch/onset/optional offset matching."""
    def arrays(events):
        intervals = np.asarray([(start, end) for start, end, _ in events], dtype=float).reshape(-1, 2)
        midi = np.asarray([pitch for _, _, pitch in events], dtype=float)
        hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
        return intervals, hz

    matched = 0
    if predicted and reference:
        ref_intervals, ref_hz = arrays(reference)
        est_intervals, est_hz = arrays(predicted)
        matched = len(mir_eval.transcription.match_notes(
            ref_intervals, ref_hz, est_intervals, est_hz,
            onset_tolerance=onset_tolerance, pitch_tolerance=50.0,
            offset_ratio=offset_ratio, offset_min_tolerance=0.05))
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall),
            "tp": matched, "fp": len(predicted) - matched, "fn": len(reference) - matched}
