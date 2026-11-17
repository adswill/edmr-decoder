# Copyright (c) 2026 Kristijonas Adomaitis (adswill). All rights reserved.
# edmr-decoder Source Licence 1.1: no copying, redistribution or porting. See LICENSE.
"""Meaning of decoded bursts, and the bookkeeping of towers and time slots."""
import time

NETWORKS = {"ca": "Turkey", "d6": "France"}
EPOCH = 1386946800                       # the network clock counts seconds from 2013-12-13 15:00 UTC
KNOWN_TYPES = {"9fb", "9ff", "000", "5b3", "5ff", "30b", "301", "30f", "e0e", "e00", "fa0", "f60", "f64"}


def _utc(unix):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(unix))


def _bitstring(words):
    return "".join(format(int(w, 16), "012b") for w in words)


def _signed(v, bits):
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def _slots(tag):
    return "".join(str(k + 1) for k in range(3) if tag >> k & 1)


def parse_names(b):
    """Name entries: 3-bit kind, 4-bit level, (4-bit length on the first entry), 7-bit characters."""
    i, out, first = 8, [], True
    while i + 11 <= 150:
        kind, level = int(b[i:i + 3], 2), int(b[i + 3:i + 7], 2)
        i += 7
        length = 99
        if first:
            length = int(b[i:i + 4], 2)
            i += 4
            first = False
        name = ""
        while i + 7 <= 152 and len(name) < length:
            v = int(b[i:i + 7], 2)
            if not 32 <= v < 127:
                break
            name += chr(v)
            i += 7
        if not name:
            break
        out.append(dict(name=name, level=level, kind=kind))
    return out


def parse_neighbours_short(b):
    """Sliding window of the neighbour table: 4-bit start index, then variable-length elements."""
    idx, i, out = int(b[56:60], 2), 60, []
    while i + 4 <= 152:
        if b[i:i + 4] == "1000":
            break
        if b[i:i + 2] == "01":
            out.append([idx, None])
            idx += 1
            i += 2
            continue
        if b[i] != "1" or i + 13 > 152:
            break
        tag = int(b[i + 1:i + 4], 2)
        if b[i + 4:i + 6] == "00":
            out.append([idx, [tag, _signed(int(b[i + 6:i + 11], 2), 5)]])
            i += 13
        elif b[i + 4:i + 6] == "01" and i + 15 <= 152:
            out.append([idx, [tag, _signed(int(b[i + 6:i + 13], 2), 7)]])
            i += 15
        else:
            break
        idx += 1
    return out


def parse_neighbours_full(b):
    """Whole neighbour table: 14-bit records, 3-bit slot set and 11-bit signed channel offset."""
    body = b[25:152]
    n, out, k = len(body.rstrip("0")), [], 0
    while k * 14 < n:
        r = body[k * 14:k * 14 + 14].ljust(14, "0")
        tag, v = int(r[:3], 2), _signed(int(r[3:], 2), 11)
        out.append([k, [tag, v] if (tag or v) else None])
        k += 1
    return out


def _neighbour_text(elements):
    return " ".join("%+d/s%s" % (e[1], _slots(e[0])) if e else "-" for _, e in elements)


def to_event(d):
    """Describe one decoded burst. d is a result of the decoder."""
    x = d["x"]
    b = _bitstring(x)
    ev = dict(t=round(d["t"], 2), raw=" ".join(x), sync=d["sym"][16:24], errors=d["nerr"], repeat=d["how"] == "match")
    unit = "%05x" % (int(x[1] + x[2], 16) >> 4)
    if d["m"][:38] == "0" * 38:
        ev.update(kind="idle", type="Idle", text="")
    elif x[0] == "9fb":
        secs = (int(x[3], 16) << 13) | (int(x[4], 16) << 1) | (int(x[5], 16) >> 11)
        utc = int(x[5], 16) & 0x7FF
        ev.update(kind="clock", type="Clock", secs=secs, utc=utc, unix=EPOCH + ((int(x[2], 16) & 63) << 25) + secs,
                  text="%s UTC, local offset %+d:%02d" % (_utc(EPOCH + ((int(x[2], 16) & 63) << 25) + secs), utc // 60, utc % 60))
    elif x[0] == "9ff":
        ev.update(kind="clock", type="Clock", text="not set")
    elif x[0] == "000" and x[1] == "000" and x[11] == "018":
        sc = int(x[2][1:] + x[3], 16) & 0xFFFF
        ev.update(kind="id", type="System info", net=x[2][:2], area=sc >> 10, site=sc & 1023, frame=(int(x[4], 16) >> 5) & 31)
        ev["text"] = "%s %02d/%d  frame %d" % (NETWORKS.get(ev["net"], "net " + ev["net"]), ev["area"], ev["site"], ev["frame"])
    elif x[0] in ("e0e", "e00"):
        names = parse_names(b)
        ev.update(kind="names", type="Names", names=names, text=", ".join(q["name"] for q in names))
    elif x[0] == "5b3":
        el = parse_neighbours_short(b)
        ev.update(kind="neighbours", type="Neighbours", elements=el, text=_neighbour_text(el))
    elif x[0] == "5ff":
        el = parse_neighbours_full(b)
        ev.update(kind="neighbours", type="Neighbour table", elements=el, text=_neighbour_text(el))
    elif x[0] in ("30b", "301", "30f"):
        ev.update(kind="settings", type="Settings", text="")
    elif x[0] == "fa0":
        ev.update(kind="poll", type="Poll", unit=unit, counter=(int(x[4], 16) & 7) << 1 | int(x[5], 16) >> 11, text="unit " + unit)
    elif x[0] in ("f60", "f64"):
        level = _signed(int(x[6], 16) >> 4, 8)
        ev.update(kind="unit", type="Unit message", unit=unit, level=level, value=int(x[7], 16),
                  text="unit %s  %d dBm  value %d" % (unit, level, int(x[7], 16)))
    else:
        ev.update(kind="unknown", type="Type " + x[0], text="")
    return ev


def is_shifted_copy(d):
    """A known message read one or two columns late; it shows up as an unknown type and is not a new message."""
    x = d["x"]
    return x[0] not in KNOWN_TYPES and d["m"][:38] != "0" * 38 and (x[1] in KNOWN_TYPES or x[2] in KNOWN_TYPES)


class Tracker:
    """Assigns bursts to the three time slots and collects what each tower says about itself.

    Slots are numbered by timing: each burst is placed relative to the previous one in thirds of a frame.
    Lanes are anchored to towers, so a jump in the audio stream does not rotate the numbering for good.
    """

    def __init__(self, sps):
        self.third = 672 * sps / 3       # samples per slot
        self.ref = None
        self.towers = {}
        self.owner = {}
        self.home = {}
        self.missed = {}

    def add(self, ev, ts):
        """ev: event from to_event, ts: absolute position of the burst in samples. Adds slot and tower to ev."""
        if self.ref is None:
            self.ref = (ts, 0)
        slot = (self.ref[1] + round((ts - self.ref[0]) / self.third)) % 3
        self.ref = (ts, slot)
        hold = 30 * 12000
        if ev["kind"] == "id":
            key = "%s-%d-%d" % (ev["net"], ev["area"], ev["site"])
            home = self.home.get(key)
            if home is not None and home != slot:
                self.missed[key] = self.missed.get(key, 0) + 1
                if self.missed[key] >= 2:
                    slot = home
                    self.ref = (ts, slot)
                    self.missed = {}
            else:
                self.missed[key] = 0
            ow = self.owner.get(slot)
            if ow is None or ow[0] == key or ts - ow[1] > hold or ow[2] >= 2:
                self.owner[slot] = (key, ts, 0)
            else:
                self.owner[slot] = (ow[0], ow[1], ow[2] + 1)
            tw = self.towers.setdefault(key, dict(net=ev["net"], network=NETWORKS.get(ev["net"], ""), area=ev["area"],
                                                  site=ev["site"], names={}, neighbours={}, utc=None, unix=None,
                                                  count=0, slot=slot, slots={}, sync=ev["sync"], last=ev["t"]))
            tw["slots"][slot] = tw["slots"].get(slot, 0) + 1
            tw["slot"] = max(tw["slots"], key=tw["slots"].get)
            if key not in self.home and tw["slots"][slot] >= 3 and slot not in self.home.values():
                self.home[key] = slot
        ev["slot"] = slot + 1
        ow = self.owner.get(slot)
        if ow and ts - ow[1] < hold and ev["kind"] != "idle":
            tw = self.towers[ow[0]]
            tw["count"] += 1
            tw["last"] = ev["t"]
            ev["tower"] = ow[0]
            if ev["kind"] == "names":
                for q in ev["names"]:
                    if q["kind"] != 7:
                        tw["names"][q["name"]] = q["level"]
            if ev["kind"] == "clock" and "utc" in ev:
                tw["utc"], tw["unix"] = ev["utc"], ev["unix"]
            if ev["kind"] == "neighbours":
                for i, e in ev["elements"]:
                    if e:
                        tw["neighbours"][str(i)] = e
        return ev

    def tower_list(self):
        out = []
        for key, tw in self.towers.items():
            t = dict(tw, key=key, slot=tw["slot"] + 1)
            t.pop("slots")
            order = {0: 0, 2: 1, 1: 2}                    # site, district, region
            t["names"] = [n for n, _ in sorted(tw["names"].items(), key=lambda q: order.get(q[1], 3))]
            out.append(t)
        return out


def events_of(results, sps):
    """Events and towers for the accepted results of one recording."""
    tracker, events = Tracker(sps), []
    for d in sorted(results, key=lambda d: d["ts"]):
        if is_shifted_copy(d):
            continue
        events.append(tracker.add(to_event(d), d["ts"]))
    return events, tracker.tower_list()
