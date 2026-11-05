#!/usr/bin/env python3
"""Drift concept-test mixer.

Renders one sleepcast from script.json + generated clips into a single MP3,
using the same rules the real Web Audio mixer will follow:

  * narration placed segment by segment, with each segment's pause_after_sec
  * ambient beds crossfade-looped, faded in when a segment "adds" them
  * ambience quieter under the wind-down, fuller in the journey
  * ambience ducked ~30% under narration, swelling back in the pauses
  * one-shot SFX at segment start + offset, panned, with a small random delay
  * cues and ambient changes can be anchored to the line: "start" (default),
    "line_end" / "after_line", or "fraction" (at x line length); "hit_sec" is the
    moment inside the clip (e.g. a door's thud) that should land on that time
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
BED_GAIN = {"rain_roof": 0.30, "fire_stove": 0.16, "wind_pines": 0.12}
PHASE_AMBIENCE = {"arrival": 0.85, "wind_down": 0.6, "journey": 1.0}
DUCK_DEPTH = 0.30          # ambience drops 30% while the voice speaks
DUCK_ATTACK_S = 0.4
DUCK_RELEASE_S = 1.8
BED_FADE_IN_S = 6.0
BED_FADE_OUT_S = 12.0
INTRO_FADE_IN_S = 8.0
MASTER_DRIFT = 0.20        # master ends 20% lower than it starts
LOOP_XFADE_S = 1.5
SFX_MAX_JITTER_S = 0.6


def decode(path: Path, af: str = None) -> np.ndarray:
    """Decode any audio file to float32 stereo at SR via ffmpeg (optionally through an ffmpeg filter)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + (["-af", af] if af else []) + ["-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def level(clip: np.ndarray, target_db: float, mode: str = "rms") -> np.ndarray:
    """Bring a clip to a consistent level so the gain tables mean the same thing for every file.
    rms  -> RMS of the non-silent part hits target_db (dBFS); used for narration and beds
    peak -> peak hits target_db; used for one-shots, whose transients vary a lot"""
    mono = np.abs(clip).max(axis=1)
    if mode == "peak":
        ref = float(mono.max())
    else:
        loud = clip[mono > 10 ** (-50 / 20)]
        ref = float(np.sqrt(np.mean(loud ** 2))) if len(loud) else 0.0
    if ref <= 0:
        return clip
    return clip * (10 ** (target_db / 20) / ref)


def tame(clip: np.ndarray, crest_db: float = 12.0, release_s: float = 0.08) -> np.ndarray:
    """Soft peak limiter for ambient beds: keep crackles and drops, but stop any single
    transient from jumping more than crest_db above the bed's own RMS (a sleeper shouldn't
    be startled by one loud pop that repeats every loop)."""
    rms = float(np.sqrt(np.mean(clip ** 2))) or 1e-9
    thr = rms * 10 ** (crest_db / 20)
    blk = 64
    nb = int(np.ceil(len(clip) / blk))
    env = np.abs(clip).max(axis=1)
    env = np.pad(env, (0, nb * blk - len(env))).reshape(nb, blk).max(axis=1)
    target = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    # 3-block lookahead so gain is already down when the transient arrives
    look = np.minimum.reduce([np.roll(target, -k) for k in range(4)])
    rel = 1 - np.exp(-blk / (release_s * SR))
    g = np.empty(nb, dtype=np.float32)
    y = 1.0
    for i, t in enumerate(look):
        y = t if t < y else y + rel * (t - y)
        g[i] = y
    gain = np.interp(np.arange(len(clip)), np.arange(nb) * blk + blk / 2, g).astype(np.float32)
    return clip * gain[:, None]


def trim(clip: np.ndarray, floor_db: float = -45.0, margin_s: float = 0.08) -> np.ndarray:
    """Cut leading/trailing silence from a TTS clip so the script's pause_after_sec is the real gap."""
    hop = SR // 100
    n = len(clip) // hop
    if n == 0:
        return clip
    e = np.sqrt(np.mean(clip[:n * hop].reshape(n, hop, 2) ** 2, axis=(1, 2)))
    on = np.where(20 * np.log10(e + 1e-9) > floor_db)[0]
    if len(on) == 0:
        return clip
    m = int(margin_s * SR)
    return clip[max(0, on[0] * hop - m): min(len(clip), (on[-1] + 1) * hop + m)]


NARRATION_RMS_DB = -20.0
BED_RMS_DB = -22.0
SFX_PEAK_DB = -6.0


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


def anchor_time(spec: dict, dur: float) -> float:
    """Seconds from the start of a line for a cue or ambient change."""
    a = spec.get("anchor", "start")
    if a in ("line_end", "after_line"):
        base = dur
    elif a == "fraction":
        base = spec.get("at", 0.0) * dur
    else:
        base = 0.0
    return base + spec.get("offset_sec", 0.0)


def build_timeline(script, audio_dir, ambient_only):
    """Return per-segment (start_sample, clip or None) and the narration end time."""
    placed, t = [], float(script.get("lead_in_sec", 0.0))   # ambience-only lead-in before the first line
    missing = []
    for seg in script["segments"]:
        clip = None
        if not ambient_only:
            p = find(audio_dir, "narration", seg["id"])
            if p is None:
                missing.append(seg["id"])
            else:
                clip = level(trim(decode(p)), NARRATION_RMS_DB)
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
    ap.add_argument("--stems", action="store_true", help="also write voice/ambience/sfx stems (.npy, final gain, before the limiter) next to --out")
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
    bed_start, bed_stop = {}, {}
    lib = {x["id"]: x for x in script.get("sfx_library", [])}
    phase_amb = {**PHASE_AMBIENCE, **script.get("mix", {}).get("phase_ambience", {})}   # per-story override
    cue_log = []
    for i, (seg, start, clip, dur) in enumerate(placed):
        phase_gain[0 if i == 0 else start:] = phase_amb.get(seg["phase"], 1.0)
        change = seg.get("ambient_change", {})
        # changes can land mid-line (e.g. a door closing); the first line's beds start at 0 under the lead-in
        at = 0 if i == 0 and "anchor" not in change and "offset_sec" not in change else start + int(anchor_time(change, dur) * SR)
        for bed in change.get("add", []):
            bed_start.setdefault(bed, at)
        for bed in change.get("remove", []):
            bed_stop.setdefault(bed, at)
        if clip is not None:
            end = min(n, start + len(clip))
            voice[start:end] += clip[:end - start] * NARRATION_GAIN
            active[start:end] = True
        for cue in seg.get("sfx", []):
            spec = {**lib.get(cue["id"], {}), **cue}          # library defaults (file, filter), cue overrides
            p = find(audio_dir, "sfx", spec.get("file", cue["id"]))
            if p is None:
                print(f"  (no file for sfx '{cue['id']}', skipping)")
                continue
            s = level(decode(p, spec.get("filter")), SFX_PEAK_DB, mode="peak")
            if "length_sec" in spec:                          # keep only the first part (e.g. two steps)
                s = s[:int(spec["length_sec"] * SR)].copy()
                fo = min(len(s) // 2, int(0.4 * SR))
                s[-fo:] *= np.linspace(1, 0, fo)[:, None]
            s = pan(s, spec.get("pan", 0.0)) * spec.get("volume", 0.4)
            # gentle 50 ms fades so one-shots never click
            f = min(len(s) // 2, int(0.05 * SR))
            s[:f] *= np.linspace(0, 1, f)[:, None]
            s[-f:] *= np.linspace(1, 0, f)[:, None]
            t = anchor_time(spec, dur) - spec.get("hit_sec", 0.0) + random.uniform(0, spec.get("jitter_sec", SFX_MAX_JITTER_S))
            at = start + int(t * SR)
            if at < 0:
                s, at = s[-at:], 0
            end = min(n, at + len(s))
            if at < n:
                sfx_bus[at:end] += s[:end - at]
                cue_log.append((at / SR, seg["id"], cue["id"], spec.get("file", cue["id"])))

    # Smooth the phase changes so ambience glides rather than steps (~4 s)
    phase_gain = smooth_envelope(phase_gain, 4.0, 4.0) if len(phase_gain) else phase_gain

    # Ambient beds
    amb = np.zeros((n, 2), dtype=np.float32)
    for bed in script["ambient_bed"]:
        bid = bed["id"]
        p = find(audio_dir, "sfx", bed.get("file", bid))   # two beds may share one recording
        if p is None:
            print(f"  (no file for bed '{bid}', skipping)")
            continue
        start = bed_start.get(bid, 0)
        loop = loop_to_length(level(tame(decode(p, bed.get("filter"))), BED_RMS_DB), n - start)
        fade = int(bed.get("fade_in_sec", INTRO_FADE_IN_S if start == 0 else BED_FADE_IN_S) * SR)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        if bed.get("fade_in_curve") == "slow":   # starts barely audible, so a bed grows in rather than switching on
            ramp = ramp ** 2
        loop[:fade] *= ramp[:, None]
        if bid in bed_stop:  # "remove": slow fade-out, then silence
            s0 = max(0, bed_stop[bid] - start)
            fo = min(int(bed.get("fade_out_sec", BED_FADE_OUT_S) * SR), len(loop) - s0)
            loop[s0:s0 + fo] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None] ** 2
            loop[s0 + fo:] = 0
        amb[start:] += loop * bed.get("gain", BED_GAIN.get(bid, 0.3))

    # Ducking: ambience dips while the voice speaks, swells back in the pauses
    duck = 1 - DUCK_DEPTH * smooth_envelope(active, DUCK_ATTACK_S, DUCK_RELEASE_S)
    amb *= (duck * phase_gain)[:, None]

    # voice_gain_db (script "mix" block) moves only the voice; the level reference below
    # is taken from the un-adjusted mix, so the background keeps its absolute level.
    voice_gain = 10 ** (script.get("mix", {}).get("voice_gain_db", 0.0) / 20)
    mix_ref = voice + amb + sfx_bus
    mix = voice * voice_gain + amb + sfx_bus

    # Master drift: 1.0 -> 0.8 across the narration, then hold; final fade-out
    ne = int(narration_end_s * SR)
    master = np.ones(n, dtype=np.float32)
    master[:ne] = np.linspace(1.0, 1.0 - MASTER_DRIFT, ne, dtype=np.float32)
    master[ne:] = 1.0 - MASTER_DRIFT
    fo = int(fade_out * SR)
    master[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32) ** 2
    mix *= master[:, None]
    mix_ref *= master[:, None]

    # One static gain for the whole track (no riding: quiet pauses must stay quiet).
    # With narration, set the voice to ~-21 dBFS RMS; ambient-only, set the bed to ~-30.
    if active.any():
        ref = float(np.sqrt(np.mean(mix_ref[active] ** 2)))
        target = -21.0 + script.get("mix", {}).get("level_db", 0.0)
    else:
        ref = float(np.sqrt(np.mean(mix_ref[:ne] ** 2)))
        target = -30.0
    g = 10 ** (target / 20) / max(ref, 1e-9)
    mix *= g

    out = Path(args.out) if args.out else HERE / "out" / (
        "drift_ambient_only.mp3" if args.ambient_only else "drift_concept_test.mp3")
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-af", "alimiter=limit=0.708:attack=2:release=60:level=disabled",
           "-ar", str(SR),
           "-codec:a", "libmp3lame", "-b:a", "192k", str(out)]
    subprocess.run(enc, input=mix.astype(np.float32).tobytes(), check=True)
    if args.stems:   # for level checks: each part with exactly the gains it has in the mix
        for name, part in (("voice", voice * voice_gain), ("ambience", amb), ("sfx", sfx_bus)):
            np.save(out.with_name(out.stem + f"_{name}.npy"), (part * (master * g)[:, None]).mean(1).astype(np.float32))

    print(f"wrote {out}  ({total_s/60:.1f} min; narration ends at {narration_end_s/60:.1f} min)")
    print("timeline:")
    for seg, start, clip, dur in placed:
        print(f"  {start/SR:6.1f}s  {seg['id']:<3} {seg['phase']:<9} {dur:4.1f}s + {seg.get('pause_after_sec',0)}s pause")
    print("cues:")
    for t, sid, cid, f in sorted(cue_log):
        print(f"  {t:6.1f}s  {sid:<3} {cid}" + (f" ({f})" if f != cid else ""))


if __name__ == "__main__":
    main()
