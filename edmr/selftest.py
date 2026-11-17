# Copyright (c) 2026 Kristijonas Adomaitis (adswill). All rights reserved.
# edmr-decoder Source Licence 1.1: no copying, redistribution or porting. See LICENSE.
"""Self-test on generated signals. Used by the automated tests and by `python -m edmr --selftest`."""
import numpy as np

from . import synth
from .audio import RATE, prepare, resample
from .decoder import accepted
from .live import Session
from .messages import events_of
from .pipeline import decode_signal


def _check(x, sent, label, verbose):
    res, info = decode_signal(x, "audio")
    events, towers = events_of([d for d in res if accepted(d)], info["sps"])
    kinds = [e["kind"] for e in events]
    ok = (len(kinds) >= len(sent) - 1 and all(k in sent for k in set(kinds)) and len(towers) == 1
          and towers[0]["names"] == ["TESTSITE"] and (towers[0]["area"], towers[0]["site"], towers[0]["utc"]) == (1, 1, 60))
    if verbose:
        print("%-22s %s  (%d of %d bursts)" % (label, "ok" if ok else "FAILED", len(kinds), len(sent)))
    return ok


def run(verbose=True):
    failed = 0
    for sps in (2.5, 2.50339, 2.4975):
        x, sent = synth.tower(sps=sps)
        failed += not _check(x, sent, "clock %.5f" % sps, verbose)
    x, sent = synth.tower()
    failed += not _check(-x, sent, "inverted polarity", verbose)
    failed += not _check(synth.tower(noise=0.3)[0], sent, "noisy", verbose)
    # the same signal as an FM carrier in an IQ recording
    z = np.exp(2j * np.pi * np.cumsum(x.astype(np.float64) * (1944 / 2900)) / RATE)
    signal, kind = prepare(np.c_[z.real, z.imag], RATE)
    res, info = decode_signal(signal, kind)
    ok = kind == "iq" and sum(accepted(d) for d in res) >= len(sent) - 1
    failed += not ok
    if verbose:
        print("%-22s %s  (%d of %d bursts)" % ("IQ input", "ok" if ok else "FAILED", sum(accepted(d) for d in res), len(sent)))
    # live: the same signal at 48 kHz in quarter-second blocks
    x, sent = synth.tower(frames=80)
    audio = resample(x, RATE, 48000) / 8000
    session, events = Session(), []
    for i in range(0, len(audio), 12000):
        events += session.feed(audio[i:i + 12000], 48000)
    towers = session.tracker.tower_list()
    ok = len(events) >= 70 and len(towers) == 1 and towers[0]["names"] == ["TESTSITE"]
    failed += not ok
    if verbose:
        print("%-22s %s  (%d of %d bursts)" % ("live, 48 kHz", "ok" if ok else "FAILED", len(events), len(sent)))
    return 1 if failed else 0
