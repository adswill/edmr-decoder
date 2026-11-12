# edmr-decoder

Decodes eDMR control channels (4-FSK, 4800 symbols/s, three time slots) from WAV recordings or a live audio input.

Input can be FM-demodulated audio or an IQ recording, at any sample rate. The symbol clock and polarity are found automatically.

## Use

Python 3.11 or newer and numpy.

```
pip install numpy
python -m edmr                   # web page: open a recording or pick an audio input
python -m edmr recording.wav     # text output
python -m edmr --help
```

Run the commands from this folder, or `pip install .` once and use `edmr` anywhere. On Windows use `py` instead of `python`.

## Output

System info (network, area, site), clock, names, neighbour lists, settings, polls and unit messages, with the time slot and tower of each burst. Voice is not decoded.

`python -m edmr --selftest` checks the installation with a generated signal.
