"""Transcribe one separated stem with its instrument-specific checkpoint."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import mido
import torch

from .config import Config
from .features import FEATURE_VERSION, drum_features, load_audio, pitch_features
from .metrics import decode_drums, decode_notes, paper_voicing_gate
from .model import DrumTranscriber, PolyphonicTranscriber
from .stems import TARGET_STEMS


def write_midi(events: list[tuple[float, float, int]], path: Path,
               stem: str = "guitar") -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    programs = {"vocals": 53, "bass": 33, "guitar": 24, "piano": 0}
    channel = 9 if stem == "drums" else 0
    if stem in programs:
        track.append(mido.Message("program_change", program=programs[stem],
                                  channel=channel, time=0))
    ordered = []
    for start, end, pitch in events:
        ordered.append((round(start * 960), 1, pitch))
        ordered.append((round(end * 960), 0, pitch))
    previous = 0
    for tick, kind, pitch in sorted(ordered):
        track.append(mido.Message("note_on" if kind else "note_off", note=pitch,
                                  velocity=80 if kind else 0, channel=channel, time=tick - previous))
        previous = tick
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def midi_events(path: Path) -> list[tuple[float, float, int]]:
    """Read note events in seconds, including tempo changes."""
    active = defaultdict(deque)
    events = []
    current = 0.0
    for message in mido.MidiFile(path):
        current += message.time
        key = (getattr(message, "channel", 0), getattr(message, "note", -1))
        if message.type == "note_on" and message.velocity > 0:
            active[key].append(current)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            if active[key]:
                start = active[key].popleft()
                if current > start:
                    events.append((start, current, int(message.note)))
    return sorted(events)


def write_multitrack_midi(events_by_stem: dict[str, list[tuple[float, float, int]]],
                          path: Path) -> None:
    """Combine the five estimated stems without collapsing their instrument identities."""
    programs = {"vocals": 53, "bass": 33, "guitar": 24, "piano": 0}
    channels = {"vocals": 0, "bass": 1, "guitar": 2, "piano": 3, "drums": 9}
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo)
    for stem in TARGET_STEMS:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=stem, time=0))
        channel = channels[stem]
        if stem != "drums":
            track.append(mido.Message("program_change", channel=channel,
                                      program=programs[stem], time=0))
        ordered = []
        for start, end, pitch in events_by_stem[stem]:
            ordered.append((round(start * 960), 1, pitch))
            ordered.append((round(end * 960), 0, pitch))
        previous = 0
        for tick, kind, pitch in sorted(ordered):
            track.append(mido.Message("note_on" if kind else "note_off",
                                      note=pitch, velocity=80 if kind else 0,
                                      channel=channel, time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def transcribe_stem(audio_path: Path, checkpoint: Path, midi_path: Path) -> int:
    """Apply a pitched or percussion model to its corresponding stem."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint uses an outdated feature schema; retrain the model")
    cfg = Config(**saved["config"])
    task = saved.get("task", "pitched")
    model = (DrumTranscriber(classes=cfg.pitches) if task == "drums"
             else PolyphonicTranscriber(saved["variant"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        audio = load_audio(audio_path, cfg)
        features = drum_features(audio, cfg, device) if task == "drums" else pitch_features(audio, cfg, device)
        x = torch.from_numpy(features.astype("float32"))[None].to(device)
        note, onset = model(x)
        note_prob = note[0].cpu().numpy()
        if saved.get("gate_mode") == "paper_fixed":
            if task == "drums":
                raise ValueError("The harmonic voicing gate is not defined for drums")
            note_prob = paper_voicing_gate(note_prob, x[0].cpu().numpy())
        events = (decode_drums(note_prob, cfg, threshold=saved["threshold"])
                  if task == "drums" else
                  decode_notes(note_prob, torch.sigmoid(onset[0]).cpu().numpy(),
                               cfg, threshold=saved["threshold"]))
    write_midi(events, midi_path, stem=saved.get("stem", "guitar"))
    return len(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    args = parser.parse_args()
    count = transcribe_stem(args.audio, args.checkpoint, args.midi)
    print(f"Wrote {count} notes to {args.midi}")


if __name__ == "__main__":
    main()
