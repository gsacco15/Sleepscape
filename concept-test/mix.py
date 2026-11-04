#!/usr/bin/env python3
"""Drift concept-test mixer.

Renders one sleepcast from script.json + generated clips into a single MP3,
using the same rules the real Web Audio mixer will follow:

  * narration placed segment by segment, with each segment's pause_after_sec
  * ambient beds crossfade-looped, faded in when a segment "adds" them
  * ambience quieter under the wind-down, fuller in the journey
  * ambience ducked ~30% under narration, swelling back in the pauses
  * one-shot SFX at segment start + offset, panned, with a small random delay
  * master volume drifts down ~20% across the narration
  * after the last line the ambience keeps going, then fades out

Usage:
  python3 mix.py                      # full mix -> out/drift_concept_test.mp3
  python3 mix.py --ambient-only       # soundscape only (no narration)
  python3 mix.py --audio-dir DIR      # where narration/ and sfx/ live

Expected files (first match wins):
  <audio-dir>/narration/<segment id>.mp3          e.g. narration/A1.mp3
  <audio-dir>/sfx/<bed or sfx id>.mp3             e.g. sfx/rain_roof.mp3
  <audio-dir>/sfx/<bed or sfx id>_v1.mp3          (variation files also accepted)
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100
HERE = Path(__file__).resolve().parent

# Mix levels (linear gain). Tuned for a quiet, sleep-friendly balance.
NARRATION_GAIN = 1.0
BED_GAIN = {"rain_roof": 0.55, "fire_stove": 0.35, "wind_pines": 0.25}
PHASE_AMBIENCE = {"arrival": 0.85, "wind_down": 0.6, "journey": 1.0}
DUCK_DEPTH = 0.30          # ambience drops 30% while the voice speaks
DUCK_ATTACK_S = 0.4
DUCK_RELEASE_S = 1.8
BED_FADE_IN_S = 6.0
INTRO_FADE_IN_S = 8.0
MASTER_DRIFT = 0.20        # master ends 20% lower than it starts
LOOP_XFADE_S = 1.5
SFX_MAX_JITTER_S = 0.6


def decode(path: Path) -> np.ndarray:
    """Decode any audio file to float32 stereo at SR via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def find(audio_dir: Path, sub: str, key: str):
    for name in (f"{key}.mp3", f"{key}_v1.mp3", f"{key}.wav"):
        p = audio_dir / sub / name
        if p.exists():
            return p
    return None


def loop_to_length(clip: np.ndarray, n: int, xfade_s: float = LOOP_XFADE_S) -> np.ndarray:
    """Tile a clip to n samples with equal-power crossfades at each seam."""
    xf = min(int(xfade_s * SR), len(clip) // 4)
    if xf <= 0:
        reps = int(np.ceil(n / len(clip)))
        return np.tile(clip, (reps, 1))[:n]
    t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)[:, None]
    fade_out, fade_in = np.cos(t), np.sin(t)
    step = len(clip) - xf
    out = np.zeros((n + len(clip), 2), dtype=np.float32)
    pos = 0
    while pos < n:
        seg = clip.copy()
        if pos > 0:
            seg[:xf] *= fade_in
        seg[-xf:] *= fade_out
        out[pos:pos + len(seg)] += seg
        pos += step
    return out[:n]


def pan(clip: np.ndarray, p: float) -> np.ndarray:
    """Equal-power pan, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    mono = clip.mean(axis=1)
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1) * np.sqrt(2)


def smooth_envelope(active: np.ndarray, attack_s: float, release_s: float) -> np.ndarray:
    """One-pole attack/release smoothing of a 0/1 activity mask (computed at 100 Hz)."""
    hop = SR // 100
    coarse = active[::hop].astype(np.float32)
    a_up = 1 - np.exp(-1 / (attack_s * 100))
    a_dn = 1 - np.exp(-1 / (release_s * 100))
    env = np.zeros_like(coarse)
    y = 0.0
    for i, x in enumerate(coarse):
        y += (a_up if x > y else a_dn) * (x - y)
        env[i] = y
    return np.repeat(env, hop)[:len(active)]


def build_timeline(script, audio_dir, ambient_only):
    """Return per-segment (start_sample, clip or None) and the narration end time."""
    placed, t = [], 0.0
    missing = []
    for seg in script["segments"]:
        clip = None
        if not ambient_only:
            p = find(audio_dir, "narration", seg["id"])
            if p is None:
                missing.append(seg["id"])
            else:
                clip = decode(p)
        # Without narration audio, stand in for its length with a reading-speed estimate
        dur = len(clip) / SR if clip is not None else max(1.5, len(seg["text"].split()) / 2.0)
        placed.append((seg, int(t * SR), clip, dur))
        t += dur + seg.get("pause_after_sec", 0)
    return placed, t, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(HERE / "script.json"))
    ap.add_argument("--audio-dir", default=str(HERE / "audio"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ambient-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    script = json.loads(Path(args.script).read_text())
    audio_dir = Path(args.audio_dir)

    placed, narration_end_s, missing = build_timeline(script, audio_dir, args.ambient_only)
    if missing:
        sys.exit(f"Missing narration files for: {', '.join(missing)} "
                 f"(expected in {audio_dir / 'narration'}). Use --ambient-only to skip narration.")

    tail = script.get("fade", {}).get("ambient_tail_sec", 75)
    fade_out = script.get("fade", {}).get("fade_out_sec", 60)
    total_s = narration_end_s + tail
    n = int(total_s * SR)

    voice = np.zeros((n, 2), dtype=np.float32)
    sfx_bus = np.zeros((n, 2), dtype=np.float32)
    active = np.zeros(n, dtype=bool)

    # Phase-dependent ambience level and bed start times
    phase_gain = np.ones(n, dtype=np.float32)
    bed_start = {}
    for seg, start, clip, dur in placed:
        phase_gain[start:] = PHASE_AMBIENCE.get(seg["phase"], 1.0)
        for bed in seg.get("ambient_change", {}).get("add", []):
            bed_start.setdefault(bed, start)
        if clip is not None:
            end = min(n, start + len(clip))
            voice[start:end] += clip[:end - start] * NARRATION_GAIN
            active[start:end] = True
        for cue in seg.get("sfx", []):
            p = find(audio_dir, "sfx", cue["id"])
            if p is None:
                print(f"  (no file for sfx '{cue['id']}', skipping)")
                continue
            s = decode(p)
            s = pan(s, cue.get("pan", 0.0)) * cue.get("volume", 0.4)
            # gentle 50 ms fades so one-shots never click
            f = min(len(s) // 2, int(0.05 * SR))
            s[:f] *= np.linspace(0, 1, f)[:, None]
            s[-f:] *= np.linspace(1, 0, f)[:, None]
            at = start + int((cue.get("offset_sec", 0) + random.uniform(0, SFX_MAX_JITTER_S)) * SR)
            end = min(n, at + len(s))
            if at < n:
                sfx_bus[at:end] += s[:end - at]

    # Smooth the phase changes so ambience glides rather than steps (~4 s)
    phase_gain = smooth_envelope(phase_gain, 4.0, 4.0) if len(phase_gain) else phase_gain

    # Ambient beds
    amb = np.zeros((n, 2), dtype=np.float32)
    for bed in script["ambient_bed"]:
        bid = bed["id"]
        p = find(audio_dir, "sfx", bid)
        if p is None:
            print(f"  (no file for bed '{bid}', skipping)")
            continue
        start = bed_start.get(bid, 0)
        loop = loop_to_length(decode(p), n - start)
        fade = int((INTRO_FADE_IN_S if start == 0 else BED_FADE_IN_S) * SR)
        loop[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        amb[start:] += loop * BED_GAIN.get(bid, 0.3)

    # Ducking: ambience dips while the voice speaks, swells back in the pauses
    duck = 1 - DUCK_DEPTH * smooth_envelope(active, DUCK_ATTACK_S, DUCK_RELEASE_S)
    amb *= (duck * phase_gain)[:, None]

    mix = voice + amb + sfx_bus

    # Master drift: 1.0 -> 0.8 across the narration, then hold; final fade-out
    ne = int(narration_end_s * SR)
    master = np.ones(n, dtype=np.float32)
    master[:ne] = np.linspace(1.0, 1.0 - MASTER_DRIFT, ne, dtype=np.float32)
    master[ne:] = 1.0 - MASTER_DRIFT
    fo = int(fade_out * SR)
    master[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32) ** 2
    mix *= master[:, None]

    # Peak-normalise to -3 dBFS before encoding, then a gentle loudness target
    peak = float(np.max(np.abs(mix))) or 1.0
    mix *= (10 ** (-3 / 20)) / peak

    out = Path(args.out) if args.out else HERE / "out" / (
        "drift_ambient_only.mp3" if args.ambient_only else "drift_concept_test.mp3")
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-af", "loudnorm=I=-21:TP=-2:LRA=11", "-ar", str(SR),
           "-codec:a", "libmp3lame", "-b:a", "192k", str(out)]
    subprocess.run(enc, input=mix.astype(np.float32).tobytes(), check=True)

    print(f"wrote {out}  ({total_s/60:.1f} min; narration ends at {narration_end_s/60:.1f} min)")
    print("timeline:")
    for seg, start, clip, dur in placed:
        print(f"  {start/SR:6.1f}s  {seg['id']:<3} {seg['phase']:<9} {dur:4.1f}s + {seg.get('pause_after_sec',0)}s pause")


if __name__ == "__main__":
    main()
