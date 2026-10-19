from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    sample_rate: int = 22050
    n_fft: int = 8192
    n_mels: int = 256
    hop_length: int = 256
    midi_min: int = 40
    midi_max: int = 82  # inclusive; official GuitarSet annotations round to 40..81
    window_frames: int = 512
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 1e-3
    seed: int = 42

    @property
    def pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    def to_dict(self) -> dict:
        return asdict(self)
