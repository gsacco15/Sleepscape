# Drift: one-story concept test

One short sleepcast ("Rainy Mountain Cabin"), built to hear the concept before the full app build.
It's shorter than the build spec on purpose:

| Phase | Build spec | This test |
|---|---|---|
| Arrival | ~1 min | ~33 s |
| Wind-down | 5–8 min | ~1:40 (breathing) |
| Journey | rest of session | ~2:20 (9 lines, pauses grow 4 s → 10 s) |
| Fade | ambience until sleep timer, 2 min fade | ~75 s ambience, 60 s fade |
| **Total** | 20 / 30 / 45 min | **≈ 5.5–6 min** |

## Files

- `script.json`: the full session in the app's JSON shape (segments, pauses, SFX cues, ambient changes, voice settings, sound prompts)
- `mix.py`: offline version of the planned Web Audio mixer (loops, ducking, SFX, master drift, fade)
- `manifest.json`: ElevenLabs node and generation IDs for every clip
- `audio/narration/<ID>.mp3` and `audio/sfx/<id>.mp3`: where the clips go (not yet downloaded, see Status)
- `out/drift_concept_test.mp3`: the rendered test track (after running the mixer)

## Voice

- **Theo Silk: British Deep Sleep & Calm** (`UmQN7jS1Ee8B1czsUtQh`), model `eleven_multilingual_v2`
- stability 0.8 · similarity 0.8 · style 0 · speed 0.85 (spec: high stability, low style)
- Alternates worth A/B testing: Serena (`X6keTmExe0nzB4g1Kfno`, warm US female), Chris "Calm Dad" (`XmUeU0FRyne67Dy7UaT4`, deep US baritone), Rainbird (`bgU7lBMo69PNEOWHFqxM`, slow British female)

## Script

**Arrival**
- A1: Welcome. Tonight, there's nowhere you need to be... and nothing you need to do. We're somewhere high in the mountains, in a small wooden cabin, while a slow rain falls outside. *(4 s)* (rain bed fades in)
- A2: The door is closed. The fire is lit. Everything here is already taken care of... so you can simply settle in, and listen. *(5 s)*

**Wind-down: breathing** (cues + silence; the mixer times the breaths, not the voice)
- W1: Before we go any further, let the body get heavy. Let your shoulders drop, just a little. Let your jaw soften. And let whatever is beneath you take all of your weight. *(6 s)*
- W2: Now, a slow breath in, through the nose... *(4 s inhale)*
- W3: and a long breath out, through the mouth. *(6 s)*
- W4: Again. Breathing in... *(4 s)*
- W5: and letting it all go. *(7 s)*
- W6: One more. In... *(4 s)*
- W7: and out. Slowly. All the way. *(8 s)*
- W8: Now let the breath find its own rhythm. There's nothing to control. Each breath out, a little softer... a little slower than the last. *(10 s)*
- W9: Notice the sound of the rain... and let it carry you inside. *(6 s)*

**Journey** (floating camera, no plot, every narrated sound has a cue)
- J1: Inside, the cabin is warm and dim. A fire breathes quietly in the iron stove, its glow moving slowly across the wooden walls. Now and then, a log settles... with a soft pop. *(4 s)* (fire bed in, log-pop SFX)
- J2: On the roof, the rain keeps its steady rhythm. Not heavy. Not light. Just constant... a soft drumming on tin, that seems to have been going on forever. *(5 s)*
- J3: Near the window, there's a deep armchair, with a wool blanket folded over its arm. The wool is thick, and a little rough... and it smells faintly of cedar, and woodsmoke. *(6 s)*
- J4: On the stove, a kettle sits just off the heat... murmuring to itself... never quite ready to boil. *(6 s)* (kettle SFX)
- J5: Outside the glass, the pines are dark and still. Rain gathers at the edge of the roof... and drips... slowly... into the puddles below. *(7 s)* (wind-in-pines bed in, drips SFX)
- J6: Far away, over the ridge, thunder rolls. Low. Unhurried. Already moving on. *(8 s)* (thunder SFX)
- J7: The fire glows. The rain falls. The cabin holds its warmth. *(9 s)*
- J8: Nothing needs to happen here. The rain... the warmth... the dark, quiet pines... *(10 s)*
- J9: Soft rain... warm light... slow... (narration trails off; ambience carries on, then fades)

## Sound prompts (`eleven_text_to_sound_v2`)

| id | type | prompt | settings |
|---|---|---|---|
| rain_roof | bed | Gentle steady rain on a tin roof, heard from inside a wooden cabin, soft even patter | loop, 20 s, influence 0.4 |
| fire_stove | bed | Wood fire crackling softly inside an iron stove, close, gentle pops | loop, 20 s, influence 0.4 |
| wind_pines | bed | Soft wind through pine trees at night, low and distant | loop, 20 s, influence 0.4 |
| log_pop | one-shot | Single log settling in a wood stove, soft crackle and pop | 3 s, influence 0.5 |
| kettle | one-shot | Kettle quietly simmering on a wood stove, soft bubbling hiss, no whistle | 8 s, influence 0.5 |
| drips | one-shot | Slow water drips falling from a roof edge into a puddle | 8 s, influence 0.5 |
| thunder | one-shot | Distant soft rolling thunder, low deep rumble, far away | 8 s, influence 0.5 |

Real costs from this run: a 20 s loop is ~67 credits, an 8 s one-shot ~27, a 3 s one-shot 10. Narration estimate is ~3,600 credits for 20 segments × 2 takes (~1,800 characters).

Chosen variations (picked by measurement: warmer tone, fewer spikes, steady level, clean loop seam), copied to `audio/sfx/`:
rain_roof v4 (sparsest drops, ~11 hits/s vs 22 for v3; low-passed at 4.5 kHz) · fire_stove v2 · wind_pines v1 · log_pop v2 · kettle v4 · drips v1 · thunder v2

## Mixer rules (`mix.py`, same as the planned Web Audio mixer)

- Voice clips have their leading/trailing silence trimmed, so each pause is exactly `pause_after_sec`
- Every clip is levelled first (narration and beds by RMS, one-shots by peak), so the gains below mean the same for any file
- Beds get a soft peak limiter (no transient more than 12 dB above the bed), so one loud crackle can't repeat every loop
- Beds tile with 1.5 s equal-power crossfades; rain fades in over 8 s, later beds over 6 s
- Ambience level by phase: arrival 0.85 · wind-down 0.6 (quiet under the breathing) · journey 1.0, gliding between phases over ~4 s
- Ducking: ambience drops 30% under narration (0.4 s attack, 1.8 s release), so it swells back in the pauses
- SFX land at segment start + offset, equal-power panned, with 0–0.6 s random delay
- Master drifts down 20% across the narration; final fade is a 60 s squared curve
- Bed levels: rain 0.30 · fire 0.16 · wind 0.12. Measured result: rain alone sits ~18 dB under the voice, the full journey ambience ~13 dB under
- Output: one static gain (voice at ~−21 dBFS RMS, no dynamic loudness riding, so pauses stay quiet), limited to −3 dBFS peak, MP3 192 kbps

```bash
python3 mix.py                 # full mix -> out/drift_concept_test.mp3
python3 mix.py --ambient-only  # soundscape only -> out/drift_ambient_only.mp3
```

Needs Python 3 with numpy, plus ffmpeg.

## Status

- [x] Script, sound prompts, voice settings
- [x] All 7 sounds generated on ElevenLabs (4 variations each); you can listen on the flow canvas
- [x] 20 narration nodes set up on the canvas with the settings above
- [x] Narration generated (2 takes per line; the slower clean take kept in `audio/narration/`)
- [x] Sound clips downloaded and best variation of each picked
- [x] Soundscape-only preview rendered (`python3 mix.py --ambient-only`)
- [x] Full mix with narration: `out/drift_concept_test.mp3` (6.8 min; voice ends at 5.6 min)
- [x] Round 1 feedback: rain quieter and sparser
