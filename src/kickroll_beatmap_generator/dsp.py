from __future__ import annotations

import numpy as np
import librosa


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(path, mono=True, sr=sr)
    return y, actual_sr


def _low_band_flux(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float = 40.0,
    fmax: float = 200.0,
) -> np.ndarray:
    """Half-wave rectified spectral flux restricted to kick-drum frequency band (40-200Hz).

    Useful for analysis and future ML feature extraction. Not used in the main
    pipeline because in hardstyle, kick and bass synths share this frequency range,
    causing false positives in dense breakdown sections.
    """
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    # L1 positive-only spectral difference — half-wave rectified per audio-dsp rules
    flux = np.maximum(0.0, np.diff(S[band_mask], axis=1)).sum(axis=0)
    # Prepend zero to preserve frame alignment after np.diff reduces length by 1
    flux = np.concatenate([[0.0], flux])
    return flux / flux.max() if flux.max() > 0.0 else flux


def _kick_band_strength(y: np.ndarray, sr: int, hop_length: int, cutoff_hz: float = 250.0) -> np.ndarray:
    """onset_strength on LP-filtered audio (kick band only).

    Available for future use. Not used in the main pipeline because in hardstyle,
    bass synths (40-150Hz) are indistinguishable from kicks by frequency alone —
    LP filtering over-detects in dense sections.
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff_hz / (sr / 2.0), "low")
    y_kick = filtfilt(b, a, y).astype(np.float32)
    return librosa.onset.onset_strength(y=y_kick, sr=sr, hop_length=hop_length)


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
