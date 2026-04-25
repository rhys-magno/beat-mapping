from __future__ import annotations

import json
import sys
import time

import click

from .dsp import load_audio, detect_beats
from .filter import validate_sections, filter_timestamps


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--section", "-s", multiple=True, nargs=2, type=float,
              required=True, metavar="START END",
              help="Scoring section in seconds (inclusive). Repeatable.")
@click.option("--output", "-o", default=None,
              help="Output JSON path. Defaults to stdout.")
@click.option("--sr", default=22050, show_default=True,
              help="Sample rate.")
def main(audio_file: str, section: tuple, output: str | None, sr: int) -> None:
    sections = list(section)

    # Static validation (before any audio I/O)
    for i, (start, end) in enumerate(sections):
        if start >= end:
            click.echo(f"ERROR [VALIDATING]: section {i}: start_sec ({start}) must be < end_sec ({end})", err=True)
            sys.exit(1)
        if start < 0:
            click.echo(f"ERROR [VALIDATING]: section {i}: start_sec ({start}) must be >= 0", err=True)
            sys.exit(1)

    try:
        t0 = time.perf_counter()
        click.echo(f"[kickroll] LOADING enter: {audio_file}", err=True)
        y, actual_sr = load_audio(audio_file, sr=sr)
        duration_sec = len(y) / actual_sr
        click.echo(f"[kickroll] LOADING complete: {duration_sec:.3f}s", err=True)

        try:
            validate_sections(sections, duration_sec)
        except ValueError as e:
            click.echo(f"ERROR [LOADING]: {e}", err=True)
            sys.exit(1)

        click.echo(f"[kickroll] ANALYZING enter", err=True)
        raw_beats = detect_beats(y, actual_sr)
        click.echo(f"[kickroll] ANALYZING complete: {len(raw_beats)} raw beats", err=True)

        filtered = filter_timestamps(raw_beats, sections)
        click.echo(f"[kickroll] FILTERING complete: {len(filtered)} beats in sections", err=True)

        result = {
            "config": {
                "audio_path": audio_file,
                "sr": actual_sr,
                "hop_length": 512 * actual_sr // 22050,
                "sections": sections,
                "seed": 42,
            },
            "source": audio_file,
            "duration_sec": round(duration_sec, 3),
            "sections": sections,
            "beats": filtered,
        }

        out_json = json.dumps(result, indent=2)
        elapsed = time.perf_counter() - t0
        click.echo(f"[kickroll] OUTPUT complete: {elapsed:.2f}s", err=True)

        if output:
            with open(output, "w") as f:
                f.write(out_json)
        else:
            click.echo(out_json)

    except Exception as e:
        click.echo(f"ERROR [PIPELINE]: {e}", err=True)
        sys.exit(1)
