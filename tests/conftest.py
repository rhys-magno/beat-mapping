import numpy as np
import soundfile as sf
import pytest


def make_click_track(
    bpm: float,
    duration_sec: float = 5.0,
    sr: int = 22050,
) -> tuple[np.ndarray, list[float]]:
    beat_interval = 60.0 / bpm
    ground_truth: list[float] = []
    # Start from beat_interval so every click has preceding silence for spectral flux
    t = beat_interval
    while t < duration_sec:
        ground_truth.append(t)
        t += beat_interval

    audio = np.zeros(int(duration_sec * sr), dtype=np.float32)
    # 30ms Hanning-windowed click — spans ~5 frames at hop=128, giving
    # non-zero env on both sides of the peak so parabolic interpolation works.
    # A 5ms delta click has env[frame-1]=0, pushing the parabola vertex forward.
    click_len = int(0.030 * sr)
    freq = 1000.0
    t_arr = np.arange(click_len) / sr
    click = 0.8 * np.sin(2 * np.pi * freq * t_arr) * np.hanning(click_len)

    for gt in ground_truth:
        # Center the click on gt so the Hanning peak aligns with the ground truth time.
        # Parabolic interpolation finds the energy peak (center), not the click start.
        start = max(0, int(gt * sr) - click_len // 2)
        end = min(start + click_len, len(audio))
        audio[start:end] += click[: end - start]

    return audio, ground_truth


@pytest.fixture
def click_track_factory(tmp_path):
    def _factory(bpm: float, duration_sec: float = 5.0):
        audio, gt = make_click_track(bpm=bpm, duration_sec=duration_sec)
        path = tmp_path / f"click_{int(bpm)}bpm.wav"
        sf.write(str(path), audio, 22050)
        return str(path), [round(t * 1000.0, 2) for t in gt]
    return _factory
