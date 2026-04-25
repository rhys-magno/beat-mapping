"""Unit tests for _low_band_flux and _kick_band_strength utility functions.

These functions are not used in the main detect_beats pipeline (hardstyle kick and
bass synths share the 40-200Hz range, making frequency isolation unreliable), but
they are available for future ML feature extraction or per-section tuning.
"""
from __future__ import annotations

import numpy as np
import pytest

from kickroll_beatmap_generator.dsp import _low_band_flux, _kick_band_strength

SR = 22050
HOP = 128


def _make_pure_sine(freq: float, duration: float = 2.0, amp: float = 0.8) -> np.ndarray:
    t = np.arange(int(duration * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_kick_burst(time_sec: float, freq: float = 80.0, duration: float = 2.0) -> np.ndarray:
    y = np.zeros(int(duration * SR), dtype=np.float32)
    click_len = int(0.030 * SR)
    t_arr = np.arange(click_len) / SR
    click = 0.8 * np.sin(2 * np.pi * freq * t_arr) * np.hanning(click_len)
    center = int(time_sec * SR)
    start = max(0, center - click_len // 2)
    y[start : start + click_len] = click[: len(y) - start]
    return y


class TestLowBandFlux:
    def test_suppresses_high_freq_tone(self):
        """Continuous 1kHz tone must produce near-zero flux in the 40-200Hz band.

        Checks mid-section only — onset/offset edge transients from the abrupt
        start/end of the sine are excluded.
        """
        y = _make_pure_sine(1000.0)
        env = _low_band_flux(y, SR, HOP)
        mid = env[10:-10]
        assert mid.max() < 0.15, (
            f"1kHz tone leaked into low band mid-section: max={mid.max():.4f}"
        )

    def test_detects_low_freq_kick(self):
        """An 80Hz kick burst produces a clear peak near the correct frame."""
        y = _make_kick_burst(time_sec=0.5)
        env = _low_band_flux(y, SR, HOP)
        expected_frame = int(0.5 * SR) // HOP
        peak_frame = int(np.argmax(env))
        assert abs(peak_frame - expected_frame) <= 8, (
            f"Low-freq kick at frame {expected_frame}, peak at {peak_frame}"
        )
        assert env[peak_frame] > 0.5

    def test_normalized_to_unit_range(self):
        """Output is normalized to [0, 1]."""
        y = _make_kick_burst(0.5)
        env = _low_band_flux(y, SR, HOP)
        assert env.min() >= 0.0
        assert env.max() <= 1.0 + 1e-9

    def test_prepended_zero_for_frame_alignment(self):
        """First element is 0.0 — prepended to preserve frame alignment after np.diff."""
        y = _make_kick_burst(0.5)
        env = _low_band_flux(y, SR, HOP)
        assert env[0] == 0.0

    def test_silent_audio_returns_zero_envelope(self):
        """Silent input returns all-zero envelope without division error."""
        y = np.zeros(int(2.0 * SR), dtype=np.float32)
        env = _low_band_flux(y, SR, HOP)
        assert env.max() == 0.0

    def test_shape_matches_stft_frame_count(self):
        """Envelope length matches STFT frame count (within ±2 for padding)."""
        duration = 3.0
        y = np.zeros(int(duration * SR), dtype=np.float32)
        env = _low_band_flux(y, SR, HOP)
        # librosa center=True: n_frames = 1 + n_samples // hop_length
        expected = 1 + len(y) // HOP
        assert abs(len(env) - expected) <= 2, (
            f"Envelope length {len(env)} vs expected ~{expected}"
        )


class TestKickBandStrength:
    def test_returns_1d_array(self):
        """Output is a 1-D float array."""
        y = _make_kick_burst(0.5)
        env = _kick_band_strength(y, SR, HOP)
        assert env.ndim == 1
        assert env.dtype in (np.float32, np.float64)

    def test_detects_low_freq_burst(self):
        """LP-filtered onset_strength produces a peak near a low-freq kick."""
        y = _make_kick_burst(time_sec=0.5, freq=80.0)
        env = _kick_band_strength(y, SR, HOP)
        expected_frame = int(0.5 * SR) // HOP
        window = env[max(0, expected_frame - 10) : expected_frame + 10]
        assert window.max() > 0.0, "No kick-band energy detected near burst location"

    def test_suppresses_high_freq_sustained_tone(self):
        """A sustained 2kHz tone produces near-zero kick-band onset_strength."""
        y = _make_pure_sine(2000.0, duration=3.0)
        env = _kick_band_strength(y, SR, HOP)
        # Skip edge frames; steady-state should be essentially zero
        mid = env[10:-10]
        # The LP filter at 250Hz attenuates 2kHz by >40dB — onset_strength should be tiny
        assert mid.max() < 0.05 * env.max() + 1e-6, (
            f"2kHz tone leaked through kick-band filter: mid max={mid.max():.4f}, "
            f"global max={env.max():.4f}"
        )
