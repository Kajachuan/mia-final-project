"""Transcribe one guitar stem with a trained checkpoint to MIDI."""

import argparse
from pathlib import Path

import mido
import torch

from .config import Config
from .features import FEATURE_VERSION, load_audio, pitch_features
from .metrics import decode_notes, paper_voicing_gate
from .model import PolyphonicTranscriber


def write_midi(events: list[tuple[float, float, int]], path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("program_change", program=24, channel=0, time=0))
    ordered = []
    for start, end, pitch in events:
        ordered.append((round(start * 960), 1, pitch))
        ordered.append((round(end * 960), 0, pitch))
    previous = 0
    for tick, kind, pitch in sorted(ordered):
        track.append(mido.Message("note_on" if kind else "note_off", note=pitch,
                                  velocity=80 if kind else 0, channel=0, time=tick - previous))
        previous = tick
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def transcribe_stem(audio_path: Path, checkpoint: Path, midi_path: Path) -> int:
    """Apply the polyphonic note model to a single audio stem."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint uses an outdated feature schema; retrain the model")
    cfg = Config(**saved["config"])
    model = PolyphonicTranscriber(saved["variant"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        audio = load_audio(audio_path, cfg)
        x = torch.from_numpy(pitch_features(audio, cfg, device).astype("float32"))[None].to(device)
        note, onset = model(x)
        note_prob = note[0].cpu().numpy()
        if saved.get("gate_mode") == "paper_fixed":
            note_prob = paper_voicing_gate(note_prob, x[0].cpu().numpy())
        events = decode_notes(note_prob, torch.sigmoid(onset[0]).cpu().numpy(),
                              cfg, threshold=saved["threshold"])
    write_midi(events, midi_path)
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
