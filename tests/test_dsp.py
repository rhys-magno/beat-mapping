import pytest
from kickroll_beatmap_generator.dsp import load_audio, detect_beats


@pytest.mark.parametrize("bpm", [60.0, 120.0, 180.0])
def test_timing_accuracy(bpm, click_track_factory):
    path, ground_truth_ms = click_track_factory(bpm=bpm)
    y, sr = load_audio(path)
    detected = detect_beats(y, sr)

    for gt_ms in ground_truth_ms:
        closest = min(abs(d - gt_ms) for d in detected)
        assert closest < 5.0, (
            f"BPM {bpm}: beat at {gt_ms}ms not within 5ms (closest={closest:.2f}ms)"
        )


def test_no_duplicates_within_20ms(click_track_factory):
    path, _ = click_track_factory(bpm=120.0)
    y, sr = load_audio(path)
    detected = detect_beats(y, sr)
    for i in range(len(detected) - 1):
        assert detected[i + 1] - detected[i] >= 20.0


def test_output_is_sorted_floats(click_track_factory):
    path, _ = click_track_factory(bpm=120.0)
    y, sr = load_audio(path)
    detected = detect_beats(y, sr)
    assert detected == sorted(detected)
    assert all(isinstance(t, float) for t in detected)
    assert all(round(t, 2) == t for t in detected)


def test_deterministic(click_track_factory):
    path, _ = click_track_factory(bpm=120.0)
    y, sr = load_audio(path)
    run1 = detect_beats(y, sr)
    run2 = detect_beats(y, sr)
    assert run1 == run2
