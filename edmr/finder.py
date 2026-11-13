"""Burst finder: locates the preamble and sync of each burst in the discriminator signal."""
from dataclasses import dataclass

import numpy as np

from .audio import interp_weights

_K = np.arange(-7, 9)
_TEMPLATE = np.array([3.0 if k & 1 else -3.0 for k in range(16)] + [3.0, 3.0, -3.0])
_NT = len(_TEMPLATE)
_DT = []
_d = -0.6
while _d <= 0.6:
    _DT.append(_d)
    _d += 0.1
_DT = np.array(_DT)


@dataclass
class Burst:
    ts: float            # start of the burst, in samples
    q: float             # correlation with the preamble, 0..1
    dc: float            # offset of the four-level signal
    level: float         # outer level of the four-level signal
    soft: np.ndarray     # 300 symbols, 10 per level unit (outer levels at +-30)
    hard: np.ndarray     # the same symbols sliced to the nearest level (+-10, +-30)


def _rrc(n, sps, alpha):
    t = (np.arange(n) - (n - 1) / 2.0) / sps
    h = np.empty(n)
    for i, v in enumerate(t):
        if abs(v) < 1e-9:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * v) - 1) < 1e-6:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * v * (1 - alpha)) + 4 * alpha * v * np.cos(np.pi * v * (1 + alpha))) / (np.pi * v * (1 - 16 * alpha * alpha * v * v))
    return (h / h.sum()).astype(np.float32)


class _Signal:
    """Filtered signal with band-limited interpolation at fractional sample positions."""

    def __init__(self, x, sps):
        x = np.clip(np.asarray(x, np.float32), -4000, 4000)
        y = np.convolve(x, _rrc(25, sps, 0.2), "same")
        y[:13] = 0
        y[len(y) - 12:] = 0
        self.n = len(y)
        self.y = y
        self.pad = np.concatenate([np.zeros(8, np.float32), y, np.zeros(9, np.float32)]).astype(np.float64)

    def at(self, t):
        t = np.asarray(t, np.float64)
        i = np.floor(t).astype(np.int64)
        w = interp_weights(t - i)
        return (self.pad[np.clip(i[..., None] + _K + 8, 0, len(self.pad) - 1)] * w).sum(-1)


def _slice(u):
    return np.where(u > 2, 3.0, np.where(u > 0, 1.0, np.where(u > -2, -1.0, -3.0)))


def find_bursts(x, sps, threshold=0.8):
    """Return the bursts in x (discriminator output at 12 kHz, outer level near 2900 or in Hz)."""
    sig = _Signal(x, sps)
    n, y = sig.n, sig.y
    if n < 320 * sps:
        return []
    sym = np.arange(_NT) * sps
    # coarse gate at whole samples; the fine search runs only where this comes close to the threshold
    lim = n - int(_NT * sps) - 2
    offs = np.floor(sym + 0.5).astype(np.int64)
    c = np.zeros(lim - 20, np.float32)
    e = np.zeros(lim - 20, np.float32)
    for k in range(_NT):
        v = y[20 + offs[k]:lim + offs[k]]
        c += v * np.float32(_TEMPLATE[k])
        e += v * v
    cz = np.zeros(n + 2, np.float32)
    cz[20:lim] = c / np.sqrt(e * (_NT * 9) + 1e-9)
    gate = cz >= threshold * 0.76
    passing = gate[:-1] | gate[1:]
    pass_idx = np.flatnonzero(passing)

    def corr(t):
        v = sig.at(np.asarray(t)[..., None] + sym)
        return (v * _TEMPLATE).sum(-1) / np.sqrt((v * v).sum(-1) * (_NT * 9) + 1e-9)

    out, last, t, end = [], -1e9, 20.0, n - 300 * sps
    span = np.arange(0, 2 * sps, 0.25)
    while t < end:
        it = int(t)
        if not passing[it]:
            j = np.searchsorted(pass_idx, it + 1)
            if j >= len(pass_idx):
                break
            t = float(pass_idx[j])
            continue
        ts = t + 0.25 * np.arange(48)
        nc = corr(ts)
        hit = None
        for k in range(48):
            if ts[k] >= end or not passing[int(ts[k])]:
                t = ts[k]
                break
            if nc[k] > threshold:
                hit = k
                break
        else:
            t = ts[-1] + 0.25
            continue
        if hit is None:
            continue
        t = ts[hit]
        u = t + span
        q = corr(u)
        b = int(np.argmax(q))
        bt, bc = (u[b], q[b]) if q[b] > nc[hit] else (t, nc[hit])
        if bt - last > 50 * sps:
            last = bt
            burst = _fit(sig, bt, bc, sps)
            if burst is not None:
                out.append(burst)
        t = bt + 10 * sps + 0.25
    return out


def _fit(sig, bt, bc, sps):
    """Fine timing, offset and level of one burst by fitting four levels to its first 260 symbols."""
    k = np.arange(260) * sps
    v = sig.at(bt + _DT[:, None] + k[None, :])                       # one row per timing offset
    dc = v[:, :14].mean(1)
    sc = np.abs(v[:, 16:19] - dc[:, None]).mean(1) / 3
    tot = np.zeros(len(_DT))
    active = np.ones(len(_DT), bool)
    sb = v.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(6):
            lv = _slice((v - dc[:, None]) / sc[:, None])
            t_new = ((v - dc[:, None] - sc[:, None] * lv) ** 2).sum(1)
            tot = np.where(active, t_new, tot)
            sab, saa, sl = (lv * v).sum(1), (lv * lv).sum(1), lv.sum(1)
            det = 260 * saa - sl * sl
            ok = active & (np.abs(det) >= 1e-9)
            ndc = (sb * saa - sl * sab) / det
            nsc = (260 * sab - sl * sb) / det
            ok &= ~(nsc < 50)
            dc = np.where(ok, ndc, dc)
            sc = np.where(ok, nsc, sc)
            active = ok
    tot = np.where(np.isnan(tot), np.inf, tot)
    i = int(np.argmin(tot))
    if not np.isfinite(tot[i]) or not sc[i] > 0:
        return None
    ts = bt + _DT[i]
    u = (sig.at(ts + np.arange(300) * sps) - dc[i]) / sc[i]
    return Burst(ts=float("%.2f" % ts), q=float("%.2f" % bc), dc=float("%.0f" % dc[i]), level=float("%.0f" % (sc[i] * 3)),
                 soft=np.rint(u * 10), hard=_slice(u) * 10)
