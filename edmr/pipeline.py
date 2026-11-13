"""From a recording to decoded bursts: clock and polarity are worked out from the signal."""
import numpy as np

from .audio import prepare, read_wav
from .decoder import RATE, accepted, decode, first_pass, strict
from .finder import find_bursts

# symbol clock of the two receiver types the decoder was developed on (samples per symbol at a nominal 12 kHz)
NOMINAL_SPS = {"audio": 2.50339, "iq": 2.50012}


def estimate_sps(starts, sps0):
    """Refine the symbol clock from burst start times: bursts repeat on a 224-symbol slot raster."""
    ts = np.asarray(starts, float)
    if len(ts) < 20:
        return None

    def score(frame):
        h = np.histogram((ts / (frame / 3)) % 1, bins=48)[0].astype(float)
        return (h ** 2).sum() / len(ts) ** 2

    f0 = 672 * sps0
    coarse = np.arange(f0 * (1 - 0.004), f0 * (1 + 0.004), 0.02)
    i = int(np.argmax([score(f) for f in coarse]))
    fine = np.arange(coarse[i] - 0.03, coarse[i] + 0.03, 0.002)
    sf = np.array([score(f) for f in fine])
    return float(fine[int(sf.argmax())] / 672) if sf.max() > 0.05 else None   # uniform phases score 0.021


def _densest(bursts, seconds=180):
    """Start times of the bursts in the busiest stretch of the recording."""
    ts = np.array([b.ts for b in bursts])
    if len(ts) < 20:
        return ts
    span = seconds * RATE
    counts = np.searchsorted(ts, ts + span) - np.arange(len(ts))
    i = int(counts.argmax())
    return ts[i:i + counts[i]]


def decode_signal(x, kind="audio", sps=None, invert=None, progress=None, **options):
    """Decode a discriminator signal at 12 kHz. Returns (results, info).

    progress: optional callable, called with the share of the work done (0..1).
    """
    step = progress or (lambda f: None)
    x = np.asarray(x, np.float32)
    fixed = sps is not None
    sps = sps or NOMINAL_SPS[kind]
    bursts = find_bursts(x, sps)
    direct = sum(strict(d) for d in first_pass(bursts))
    inverted = False
    step(0.2)
    if invert or (invert is None and direct < max(3, 0.1 * len(bursts))):
        other = find_bursts(-x, sps)
        n = sum(strict(d) for d in first_pass(other))
        if invert or n > direct:
            x, bursts, direct, inverted = -x, other, n, True
    step(0.3)
    if not fixed:
        est = estimate_sps(_densest(bursts), sps)
        if est is not None and abs(est - sps) / sps > 2e-5:
            again = find_bursts(x, est)
            n = sum(strict(d) for d in first_pass(again))
            if n >= direct:
                sps, bursts = est, again
    step(0.45)
    res, stats = decode(x, bursts, sps, kind, progress=lambda f: step(0.45 + 0.55 * f), **options)
    step(1.0)
    info = dict(stats, kind=kind, sps=round(sps, 5), inverted=inverted, seconds=round(len(x) / RATE, 1))
    return res, info


def decode_file(source, progress=None, **options):
    """Decode a WAV recording (path or bytes; FM audio or IQ, any sample rate).

    progress: optional callable, called with the share of the work done (0..1).
    """
    step = progress or (lambda f: None)
    step(0.0)
    samples, rate = read_wav(source)
    x, kind = prepare(samples, rate)
    step(0.1)
    res, info = decode_signal(x, kind, progress=lambda f: step(0.1 + 0.9 * f), **options)
    info.update(rate=rate, channels=samples.shape[1])
    return [d for d in res if accepted(d)], info
