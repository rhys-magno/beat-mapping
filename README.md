# kickroll-beatmap-generator

Detects beat timestamps in an audio file and returns only those that fall within manually defined scoring sections. Output is a JSON list of float millisecond timestamps — the raw input for the KickRoll iOS rhythm game's hit detection engine. Does not generate v1.1.0 beatmap schema, zones, or iOS-specific fields.

## Setup

```bash
git clone https://github.com/rhys-magno/beat-mapping.git
cd beat-mapping
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Single scoring section (seconds)
kickroll-beatmap song.mp3 --section 0 30 --output beats.json

# Multiple sections
kickroll-beatmap song.wav \
  --section 15 45 \
  --section 90 120 \
  --output beats.json

# Stdout (pipe to jq)
kickroll-beatmap song.wav --section 0 60 | jq '.beats | length'
```

Example output:
```json
{
  "source": "song.wav",
  "duration_sec": 120.000,
  "sections": [[15.0, 45.0], [90.0, 120.0]],
  "beats": [15020.45, 15520.12, 16019.88]
}
```

## Architecture

- `dsp.py` — audio loading, `beat_track` + `onset_detect`, parabolic sub-frame interpolation, 20ms dedup merge
- `filter.py` — section range validation, timestamp membership filtering
- `cli.py` — click entry point, phase logging to stderr, JSON serialisation
