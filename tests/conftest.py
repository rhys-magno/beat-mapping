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
    t = 0.0
    while t < duration_sec:
        ground_truth.append(t)
        t += beat_interval

    audio = np.zeros(int(duration_sec * sr), dtype=np.float32)
    click_len = int(0.005 * sr)  # 5ms sine burst
    freq = 1000.0
    t_arr = np.arange(click_len) / sr
    click = 0.8 * np.sin(2 * np.pi * freq * t_arr) * np.hanning(click_len)

    for gt in ground_truth:
        start = int(gt * sr)
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
