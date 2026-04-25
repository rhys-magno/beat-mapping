from __future__ import annotations

import numpy as np
import librosa


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(path, mono=True, sr=sr)
    return y, actual_sr


def _interpolate_frame(frame: int, envelope: np.ndarray, hop_length: int, sr: int) -> float:
    fi = int(frame)
    if 0 < fi < len(envelope) - 1:
        a = envelope[fi - 1]
        b = envelope[fi]
        g = envelope[fi + 1]
        denom = a - 2.0 * b + g
        offset = 0.5 * (a - g) / denom if denom != 0.0 else 0.0
        offset = max(-0.5, min(0.5, offset))
    else:
        offset = 0.0
    return (fi + offset) * hop_length / sr


def detect_beats(y: np.ndarray, sr: int, hop_length: int = 512) -> list[float]:
    np.random.seed(42)

    # 128-base hop gives 5.8ms frames at 22050Hz — sufficient for <5ms target
    # without requiring symmetric parabola peaks (sharp transients have zero pre-frame energy).
    hop_length = 128 * sr // 22050

    # Beat tracking — global tempo reference
    tempo_arr, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=120.0,
        tightness=100.0,
        units="frames",
    )

    # Use librosa's onset_strength as the canonical envelope — same envelope
    # used for both onset_detect peak-picking and parabolic interpolation,
    # ensuring the interpolation corrects the integer-frame quantisation error.
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Onset detection on onset_strength envelope
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        pre_max=4,
        post_max=4,
        pre_avg=3,
        post_avg=2,
        delta=0.025,
        wait=4,
        units="frames",
    )

    # Sub-frame interpolation on both sources using the same envelope
    onset_times = [_interpolate_frame(f, envelope, hop_length, sr) for f in onset_frames]
    beat_times = [_interpolate_frame(f, envelope, hop_length, sr) for f in beat_frames]

    # Merge: onset preferred, beat_track fills gaps
    # Place onset events first so they win dedup at same time position
    all_events: list[tuple[float, str]] = (
        [(t, "onset") for t in onset_times] +
        [(t, "beat") for t in beat_times]
    )
    all_events.sort(key=lambda x: x[0])

    DEDUP_WINDOW_SEC = 0.020
    survivors: list[float] = []
    last_kept = -999.0
    for t, _ in all_events:
        if t - last_kept >= DEDUP_WINDOW_SEC:
            survivors.append(t)
            last_kept = t

    return [round(t * 1000.0, 2) for t in survivors]
