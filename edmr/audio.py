"""Reading recordings: any WAV layout, any sample rate, audio or IQ."""
import struct

import numpy as np

RATE = 12000  # internal sample rate, 2.5 samples per symbol


def read_wav(source):
    """Return (samples, rate) for a WAV file given as a path or as bytes.

    samples is float64 with shape (frames, channels), full scale = 1.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        with open(source, "rb") as f:
            raw = f.read()
    if raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        tag, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt = body
        elif tag == b"data":
            # recorders that are stopped abruptly leave a wrong size here; take what is in the file
            data = raw[pos + 8:] if size in (0, 0xFFFFFFFF) or pos + 8 + size > len(raw) else body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("WAV file has no audio data")
    kind, channels, rate = struct.unpack("<HHI", fmt[:8])
    bits = struct.unpack("<H", fmt[14:16])[0]
    if kind == 0xFFFE and len(fmt) >= 26:  # extensible: the real format is in the sub-format field
        kind = struct.unpack("<H", fmt[24:26])[0]
    width = bits // 8
    data = data[:len(data) // (width * channels) * (width * channels)]
    if kind == 3:
        x = np.frombuffer(data, "<f4" if bits == 32 else "<f8").astype(np.float64)
    elif kind == 1 and bits == 8:
        x = (np.frombuffer(data, np.uint8).astype(np.float64) - 128) / 128
    elif kind == 1 and bits == 16:
        x = np.frombuffer(data, "<i2").astype(np.float64) / 32768
    elif kind == 1 and bits == 24:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (v - ((v & 0x800000) << 1)).astype(np.float64) / 8388608
    elif kind == 1 and bits == 32:
        x = np.frombuffer(data, "<i4").astype(np.float64) / 2147483648
    else:
        raise ValueError("unsupported WAV sample format (type %d, %d bits)" % (kind, bits))
    return x.reshape(-1, channels), rate


def is_iq(x):
    """Two channels that carry a complex baseband signal instead of the same audio twice."""
    if x.shape[1] < 2:
        return False
    a, b = x[:200000, 0], x[:200000, 1]
    pa, pb = np.mean(a * a), np.mean(b * b)
    if pa == 0 or pb == 0 or not 0.25 < pa / pb < 4:
        return False
    return abs(np.mean(a * b)) / np.sqrt(pa * pb) < 0.6


def fm_demodulate(x, rate):
    """Frequency-demodulate an IQ recording (channel 0 = I, channel 1 = Q). Output in Hz."""
    z = x[:, 0] + 1j * x[:, 1]
    n = 129
    k = np.arange(n) - (n - 1) / 2
    h = 2 * 4000 / rate * np.sinc(2 * 4000 / rate * k) * np.blackman(n)
    z = np.convolve(z, h / h.sum(), "full")[:len(z)]
    return np.concatenate([[0.0], np.angle(z[1:] * np.conj(z[:-1])) * rate / (2 * np.pi)])


def _lowpass(x, cutoff, taps=None):
    """Windowed-sinc low-pass. cutoff as a fraction of the sample rate."""
    n = taps or (int(4 / cutoff) | 1)
    k = np.arange(n) - (n - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.hamming(n)
    return np.convolve(x, h / h.sum(), "same")


def resample(x, rate, out_rate=RATE):
    """Change the sample rate (band-limited, any ratio)."""
    if rate == out_rate:
        return np.asarray(x, np.float64)
    if rate > out_rate:
        x = _lowpass(x, 0.45 * out_rate / rate)
    n_out = int(len(x) * out_rate / rate)
    k = np.arange(-7, 9)
    out = np.empty(n_out)
    for a in range(0, n_out, 1 << 18):
        t = np.arange(a, min(a + (1 << 18), n_out)) * (rate / out_rate)
        i = np.floor(t).astype(np.int64)
        u = (t - i)[:, None] - k[None, :]
        w = np.sinc(u) * (0.5 + 0.5 * np.cos(np.pi * u / 8))
        out[a:a + len(t)] = (x[np.clip(i[:, None] + k[None, :], 0, len(x) - 1)] * w).sum(1)
    return out


def set_level(d):
    """Scale FM audio so that the outer symbol level lands near 2900 (the burst finder clips at 4000)."""
    d = np.asarray(d, np.float32).copy()
    if not len(d) or d.std() == 0:
        return d
    e = np.sqrt((d[:len(d) // 12 * 12].reshape(-1, 12) ** 2).mean(1))
    lo, hi = np.percentile(e, 20), np.percentile(e, 90)
    loud = e > (lo + hi) / 2

    def high_share():
        # audio power above 3.6 kHz relative to the speech band, in the loud stretches (dB)
        m = np.repeat(loud, 12)[:len(d) // 240 * 240].reshape(-1, 240).mean(1) > 0.8
        seg = d[:len(d) // 240 * 240].reshape(-1, 240)[m][:3000]
        if len(seg) < 20:
            return 0.0
        p = (np.abs(np.fft.rfft(seg * np.hanning(240), axis=1)) ** 2).mean(0)
        f = np.fft.rfftfreq(240, 1 / RATE)
        return 10 * np.log10(p[(f > 3600) & (f < 4800)].mean() / p[(f > 300) & (f < 2400)].mean())

    if lo > 0 and hi / lo > 1.6 and loud.mean() < 0.45 and high_share() < -16:
        # quiet gaps between bursts (a carrier stays up): the loud part is the signal
        d *= 2160 / np.sqrt((e[loud] ** 2).mean())
    else:
        # hiss between bursts: this level lets the finder's clip remove noise clicks
        d *= 3900 / d.std()
    return d


def prepare(samples, rate):
    """Turn a recording into the discriminator signal at 12 kHz. Returns (signal, "audio" or "iq")."""
    if is_iq(samples):
        return resample(fm_demodulate(samples, rate), rate).astype(np.float32), "iq"
    ch = int(np.argmax((samples * samples).mean(0)))
    return set_level(resample(samples[:, ch], rate)), "audio"


class StreamResampler:
    """Resample a live stream to the internal rate block by block, keeping state between blocks."""

    def __init__(self, rate):
        self.rate = rate
        n = int(24 * rate / RATE) | 1
        k = np.arange(n) - (n - 1) / 2
        cut = min(5300, 0.45 * rate) / rate
        h = 2 * cut * np.sinc(2 * cut * k) * np.hamming(n)
        self.h = h / h.sum()
        self.tail = np.zeros(n - 1)
        self.pos = 0.0

    def push(self, x):
        x = np.asarray(x, np.float64)
        if self.rate == RATE:
            return x
        buf = np.concatenate([self.tail, x])
        y = np.convolve(buf, self.h, "valid")
        self.tail = buf[-(len(self.h) - 1):]
        step = self.rate / RATE
        idx = np.arange(self.pos, len(y) - 1, step)
        if not len(idx):
            self.pos -= len(y)
            return np.zeros(0)
        self.pos = idx[-1] + step - len(y)
        return np.interp(idx, np.arange(len(y)), y)
