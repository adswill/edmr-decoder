# Copyright (c) 2026 Kristijonas Adomaitis (adswill). All rights reserved.
# edmr-decoder Source Licence 1.1: no copying, redistribution or porting. See LICENSE.
"""Decoding a live audio stream in overlapping windows."""
import collections

import numpy as np

from .audio import RATE, StreamResampler, set_level
from .decoder import accepted
from .messages import Tracker, is_shifted_copy, to_event
from .pipeline import decode_signal

HOP = 1.0                                # a window is decoded every second ...
OVERLAP = 0.5                            # ... and the clock search reaches half a second back
CONTEXT = 8.0                            # once locked, each window reaches this far back: the frame-timing and
                                         # repeat passes need a few dozen decoded bursts around a weak one
HISTORY = 600                            # messages kept from earlier windows for the repeat pass
HOLD = 1.0                               # repeats are reported only this far behind the newest audio, so a
                                         # burst still near the edge can be read in full by the next window
EMIT = 2.0                               # bursts decoded for the first time are reported up to this far back
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
        self.history = collections.deque(maxlen=HISTORY)   # (position, words) of bursts read in full
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

    def _decode(self, a, choice, prior=None):
        res, info = decode_signal(set_level(self.buf[a - self.base:]), "audio", sps=choice[0], invert=choice[1], prior=prior)
        return [d for d in res if accepted(d)], info

    def _window(self):
        end = self.base + len(self.buf)
        a = max(self.base, self.done - int(OVERLAP * RATE))
        if self.lock:
            choice = self.lock
        else:
            # pick the clock on the short stretch, then decode the long one with it
            best = None
            for c in [(c, inv) for inv in (False, True) for c in CLOCKS]:
                good, info = self._decode(a, c)
                if best is None or len(good) > len(best[1]):
                    best = (c, good, info)
            choice = best[0]
            self.score[choice] = self.score.get(choice, 0) + len(best[1])
            if max(self.score.values()) >= 25:
                self.lock = max(self.score, key=self.score.get)
        lo = max(self.base, end - int(CONTEXT * RATE))
        # messages read before this window: the repeat pass compares weak bursts with them too
        prior = collections.Counter(w for t, w in self.history if t < lo)
        good, info = self._decode(lo, choice, prior)
        self.tracker.third = 672 * choice[0] / 3
        first = min(a, end - int(EMIT * RATE))   # older bursts were reported, or given up on, in earlier windows
        events = []
        for d in sorted(good, key=lambda d: d["ts"]):
            ta = lo + d["ts"]
            key = int(round(ta / 80))
            if d["how"] == "match" and ta > end - HOLD * RATE:
                continue
            if ta < first or key in self.seen or key - 1 in self.seen or key + 1 in self.seen or is_shifted_copy(d):
                continue
            self.seen.add(key)
            if d["how"] == "soft":
                self.history.append((ta, tuple(d["w"])))
            ev = to_event(d)
            ev["t"] = round(ta / RATE, 2)
            events.append(self.tracker.add(ev, ta))
        self.done = end
        cut = max(0, self.done - int((CONTEXT + 1) * RATE) - self.base)
        if cut > 30 * RATE:
            self.buf = self.buf[cut:]
            self.base += cut
            self.seen = {k for k in self.seen if k > self.base // 80 - 10}
        self.status = dict(locked=bool(self.lock), sps=choice[0], inverted=choice[1], found=info["found"],
                           decoded=len(good), seconds=round(end / RATE, 1))
        return events
