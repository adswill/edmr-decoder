"""Decoding a live audio stream in short overlapping windows."""
import numpy as np

from .audio import RATE, StreamResampler, set_level
from .decoder import accepted
from .messages import Tracker, is_shifted_copy, to_event
from .pipeline import decode_signal

HOP = 1.0                                # a window is decoded every second ...
OVERLAP = 0.5                            # ... and reaches half a second back, so no burst is cut in two
CLOCKS = (2.50000, 2.50339, 2.50089)     # symbol clocks tried until one of them locks


class Session:
    def __init__(self):
        self.resampler = None
        self.buf = np.zeros(0, np.float32)
        self.base = 0                    # absolute index of buf[0]
        self.done = 0
        self.seen = set()
        self.tracker = Tracker(CLOCKS[0])
        self.lock = None                 # (samples per symbol, inverted) once enough bursts decoded
        self.score = {}
        self.status = dict(locked=False, found=0, decoded=0, seconds=0.0)

    def feed(self, samples, rate):
        """Add a block of mono audio. Returns the events decoded since the last call."""
        if self.resampler is None or self.resampler.rate != rate:
            self.resampler = StreamResampler(rate)
        self.buf = np.concatenate([self.buf, self.resampler.push(samples).astype(np.float32)])
        out = []
        while self.base + len(self.buf) - self.done >= HOP * RATE:
            out += self._window()
        return out

    def _window(self):
        end = self.base + len(self.buf)
        a = max(self.base, self.done - int(OVERLAP * RATE))
        seg = set_level(self.buf[a - self.base:])
        options = [self.lock] if self.lock else [(c, inv) for inv in (False, True) for c in CLOCKS]
        best = None
        for sps, inv in options:
            res, info = decode_signal(seg, "audio", sps=sps, invert=inv)
            good = [d for d in res if accepted(d)]
            if best is None or len(good) > len(best[1]):
                best = ((sps, inv), good, info)
        choice, good, info = best
        if not self.lock:
            self.score[choice] = self.score.get(choice, 0) + len(good)
            if max(self.score.values()) >= 25:
                self.lock = max(self.score, key=self.score.get)
        self.tracker.third = 672 * choice[0] / 3
        events = []
        for d in sorted(good, key=lambda d: d["ts"]):
            ta = a + d["ts"]
            key = int(round(ta / 80))
            if key in self.seen or key - 1 in self.seen or key + 1 in self.seen or is_shifted_copy(d):
                continue
            self.seen.add(key)
            ev = to_event(d)
            ev["t"] = round(ta / RATE, 2)
            events.append(self.tracker.add(ev, ta))
        self.done = end
        cut = max(0, self.done - 2 * RATE - self.base)
        if cut > 30 * RATE:
            self.buf = self.buf[cut:]
            self.base += cut
            self.seen = {k for k in self.seen if k > self.base // 80 - 10}
        self.status = dict(locked=bool(self.lock), sps=choice[0], inverted=choice[1], found=info["found"],
                           decoded=len(good), seconds=round(end / RATE, 1))
        return events
