# edmr-decoder

Decoder for eDMR control channels.

It reads a recording or a live audio input, finds the bursts, decodes the control messages and shows which tower and time slot each one belongs to.

## Contents

- [What it decodes](#what-it-decodes)
- [Requirements](#requirements)
- [Install](#install)
- [Use](#use)
- [Output](#output)
- [How it works](#how-it-works)
- [Limits](#limits)
- [Testing](#testing)

## What it decodes

Signal: 4-FSK at 4800 symbols/s, bursts in three time slots per 140 ms frame.

| Message | Content |
|---|---|
| System info | network code, area, site, frame number |
| Clock | network time and local offset |
| Names | site, district and region names |
| Neighbours | channel offsets and slot sets of nearby towers |
| Settings | network settings broadcast (shown, fields not interpreted) |
| Poll | unit number and counter |
| Unit message | unit number, level, value |
| Idle | filler bursts |

Voice is not decoded.

## Requirements

- Python 3.11 or newer
- numpy
- macOS, Linux or Windows
- For live use: a browser, and the receiver's audio on a sound input

## Install

```
pip install .
```

or run it from the folder without installing: `pip install numpy`, then `python -m edmr`.

## Use

| Command | What it does |
|---|---|
| `python -m edmr` | opens the web page: open a recording or pick an audio input |
| `python -m edmr recording.wav` | decodes a recording: towers, then every message |
| `python -m edmr --summary recording.wav` | towers and counts only |
| `python -m edmr --idle recording.wav` | also lists idle bursts |
| `python -m edmr --raw recording.wav` | adds the 13 data words of each burst |
| `python -m edmr --json recording.wav` | everything as JSON |
| `python -m edmr --selftest` | decodes generated test signals |

Input is a WAV file of FM-demodulated audio or an IQ recording, at any sample rate and bit depth. The symbol clock and polarity are found automatically; `--sps` overrides the clock. On Windows use `py` instead of `python`.

## Output

The text output has a header with the burst counts, a table of towers (network, area/site, slot, clock, names) and one line per message with time, slot, tower, type and content. A `~` after the type marks a weak burst that was recognised as a repeat of a message already decoded.

The web page shows three slot lanes over time, the message list, a tower panel, a details pane and, for live input, a waterfall.

## How it works

1. A burst finder locates each burst by its preamble and sync.
2. Bursts are decoded with soft decisions over the column code and accepted only with a correct CRC and margins that keep false decodes out.
3. An equaliser is fitted on the bursts that decode and the rest are tried again through it.
4. Towers send on a fixed frame beat, so bursts the finder missed are decoded at their predicted positions.
5. Weak bursts on that beat are compared with messages already seen in the recording.

Live input is decoded in overlapping windows that reach 8 s back, once per second.

## Limits

- Control channel only. No call set-up, text or position traffic has been seen, so none is decoded.
- The settings message and some small fields are shown without interpretation.
- Live decoding recovers slightly fewer bursts than decoding the same audio as a file.

## Testing

`python -m edmr --selftest` generates a tower signal at several clock rates, with inverted polarity, noise, as IQ and as a live stream, and checks the result. The same test runs on Linux and Windows through GitHub Actions.
