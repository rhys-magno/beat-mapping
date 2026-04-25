from __future__ import annotations

import json
import sys
import time

import click

from .dsp import load_audio, detect_beats
from .filter import validate_sections, filter_timestamps


@click.command(
    help=(
        "Detect kick-drum beats in AUDIO_FILE and output timestamps as JSON.\n\n"
        "Only beats within at least one --section window are included. "
        "Timestamps are in milliseconds.\n\n"
        "Example:\n\n"
        "  kickroll-beatmap track.wav -s 0 60 -s 120 180 -o beats.json\n\n"
        "Output:\n\n"
        '  {"beats": [123.45, 456.78, ...]}'
    )
)
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--section", "-s", multiple=True, nargs=2, type=float,
              required=True, metavar="START END",
              help="Scoring window in seconds (inclusive). Repeat for multiple: -s 0 30 -s 90 120.")
@click.option("--output", "-o", default=None, metavar="PATH",
              help="Write JSON to PATH instead of stdout. Log messages always go to stderr.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Include full config in output JSON (source, duration_sec, sections, config).")
@click.option("--sr", default=22050, show_default=True, metavar="HZ",
              help="Sample rate in Hz. File is resampled to this rate on load.")
def main(audio_file: str, section: tuple, output: str | None, verbose: bool, sr: int) -> None:
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
        click.echo(f"[kickroll] load  : reading {audio_file}", err=True)
        y, actual_sr = load_audio(audio_file, sr=sr)
        duration_sec = len(y) / actual_sr
        click.echo(f"[kickroll] load  : done — {duration_sec:.3f}s at {actual_sr}Hz", err=True)

        try:
            validate_sections(sections, duration_sec)
        except ValueError as e:
            click.echo(f"ERROR [LOADING]: {e}", err=True)
            sys.exit(1)

        click.echo(f"[kickroll] detect: running onset + beat-track", err=True)
        raw_beats = detect_beats(y, actual_sr)
        click.echo(f"[kickroll] detect: done — {len(raw_beats)} raw beats", err=True)

        section_str = ", ".join(f"{s}–{e}s" for s, e in sections)
        click.echo(f"[kickroll] filter: {len(sections)} section(s): {section_str}", err=True)
        filtered = filter_timestamps(raw_beats, sections)
        click.echo(f"[kickroll] filter: done — {len(filtered)} beats retained", err=True)

        if verbose:
            result = {
                "source": audio_file,
                "duration_sec": round(duration_sec, 3),
                "config": {
                    "audio_path": audio_file,
                    "sr": actual_sr,
                    "hop_length": 128 * actual_sr // 22050,
                    "sections": sections,
                    "seed": 42,
                },
                "sections": sections,
                "beats": filtered,
            }
        else:
            result = {"beats": filtered}

        out_json = json.dumps(result, indent=2)
        elapsed = time.perf_counter() - t0
        click.echo(f"[kickroll] output: {'writing ' + output if output else 'stdout'} — {elapsed:.2f}s total", err=True)

        if output:
            with open(output, "w") as f:
                f.write(out_json)
        else:
            click.echo(out_json)

    except Exception as e:
        click.echo(f"ERROR [PIPELINE]: {e}", err=True)
        sys.exit(1)
