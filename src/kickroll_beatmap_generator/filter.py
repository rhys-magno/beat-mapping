from __future__ import annotations


def validate_sections(
    sections: list[tuple[float, float]],
    duration_sec: float,
) -> None:
    for i, (start, end) in enumerate(sections):
        if start < 0:
            raise ValueError(f"Section {i}: start_sec ({start}) must be >= 0")
        if end > duration_sec + 0.001:
            raise ValueError(f"Section {i}: end_sec ({end}) exceeds file duration ({duration_sec:.3f}s)")
        if start >= end:
            raise ValueError(f"Section {i}: start_sec ({start}) must be < end_sec ({end})")


def filter_timestamps(
    timestamps_ms: list[float],
    sections: list[tuple[float, float]],
) -> list[float]:
    result = []
    for t in timestamps_ms:
        t_sec = t / 1000.0
        for start, end in sections:
            if start <= t_sec <= end:
                result.append(t)
                break
    return sorted(result)
