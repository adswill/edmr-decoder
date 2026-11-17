# Copyright (c) 2026 Kristijonas Adomaitis (adswill). All rights reserved.
# edmr-decoder Source Licence 1.1: no copying, redistribution or porting. See LICENSE.
"""Test signal generator: the discriminator signal of a tower sending control messages."""
import numpy as np

from .code import CODEWORDS, LEVELS, with_crc

# preamble, sync pattern and header codeword of a burst, as symbol levels
_HEAD = [-3, 3] * 8 + [3, 3, -3, 3, -3, -3, 3, 3] + [3, 3, -1, 1, 3, 3, -1, 1, 3, 3, -1, 3]


def _bits(value, n):
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def system_info(net, area, site, frame):
    b = [0] * 152
    b[24:32] = _bits(net, 8)
    b[32:38] = _bits(area, 6)
    b[38:48] = _bits(site, 10)
    b[48:60] = _bits(frame << 5, 12)
    b[132:144] = _bits(0x018, 12)
    return b


def clock(seconds, utc_minutes):
    b = [0] * 152
    b[0:12] = _bits(0x9FB, 12)
    b[24:36] = _bits(seconds >> 25, 12)
    b[36:61] = _bits(seconds & ((1 << 25) - 1), 25)
    b[61:72] = _bits(utc_minutes, 11)
    return b


def names(name):
    b = [0] * 152
    b[0:8] = _bits(0xE0, 8)
    entry = _bits(0, 3) + _bits(0, 4) + _bits(len(name), 4)
    for ch in name:
        entry += _bits(ord(ch), 7)
    b[8:8 + len(entry)] = entry
    return b


def burst_symbols(bits152):
    """The 196 symbol levels of one burst."""
    words = with_crc(bits152)
    idx = [int(w, 4) for w in words]
    grid = [LEVELS[(36 + 13 * r + c) % 4][CODEWORDS[idx[c]][r]] / 10 for r in range(12) for c in range(13)]
    return np.array(_HEAD + grid + [3, -3, 3, -3], float)


def _rrc(t, alpha=0.2):
    out = np.empty(len(t))
    for i, v in enumerate(t):
        if abs(v) < 1e-9:
            out[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * v) - 1) < 1e-6:
            out[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            out[i] = (np.sin(np.pi * v * (1 - alpha)) + 4 * alpha * v * np.cos(np.pi * v * (1 + alpha))) / (np.pi * v * (1 - 16 * alpha * alpha * v * v))
    return out


def tower(frames=40, sps=2.5, noise=0.15, seed=1, net=0xAA, area=1, site=1, name="TESTSITE"):
    """Discriminator signal (outer level 2900) of a tower using one slot per frame. Returns (signal, messages)."""
    rng = np.random.default_rng(seed)
    n = int((frames + 1) * 672 * sps)
    x = np.zeros(n)
    sent = []
    span = np.arange(-8 * sps, 8 * sps + 1)
    for f in range(frames):
        kind = f % 4
        bits = (system_info(net, area, site, f % 25), clock(400000000 + f, 60), names(name), [0] * 152)[kind]
        sent.append(("id", "clock", "names", "idle")[kind])
        start = 300 + f * 672 * sps
        for k, a in enumerate(burst_symbols(bits)):
            c = start + k * sps
            i0 = int(np.floor(c))
            idx = i0 + span.astype(int)
            x[idx] += a * _rrc((idx - c) / sps)
    x *= 2900 / 3 / _rrc(np.zeros(1))[0]
    return (x + rng.normal(0, noise * 2900 / 3, n)).astype(np.float32), sent
