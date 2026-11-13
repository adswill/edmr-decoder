"""Burst decoding: soft decision over the column code, CRC, and the equaliser passes for band-limited audio."""
import collections

import numpy as np

from .audio import interp_weights
from .code import CODEWORDS, LEVELS, ONE_HOT_T, crc_ok, message_bits, with_crc, words_hex

RATE = 12000
NTAPS = 17
_J = (np.arange(NTAPS) - NTAPS // 2) * 0.5
_K = np.arange(-7, 9)

# equaliser taps fitted on long recordings; used when a recording has too few bursts to fit its own
STORED_TAPS = {
    "audio": np.array([-0.0427998281, -0.0199440096, 0.00400644117, -0.129308413, -0.0679829201, -0.0686900303,
                       0.00583690006, 0.10568313, 0.513990466, 0.333134162, -0.0622643148, -0.0965126841,
                       0.0552845054, -0.00742711796, 0.0583242374, -0.0708680288, 0.0376580331, 0.0674048264]),
    "iq": np.array([-0.0018111002, -0.00548883995, -0.0061965395, 0.0229642739, 0.0299479885, -0.0916044062,
                    -0.103100298, 0.278042501, 0.556421401, 0.281966137, -0.0987697802, -0.103924049,
                    0.0288997069, 0.0351778418, -0.00810627429, -0.0114998923, 0.000812087807, 0.0510632562]),
}


def _flat_taps(sps):
    """Equaliser taps for a receiver with a flat audio response: the matched filter alone."""
    alpha = 0.2
    t = _J.copy()
    h = np.empty(NTAPS)
    for i, v in enumerate(t):
        if abs(v) < 1e-9:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * v) - 1) < 1e-6:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * v * (1 - alpha)) + 4 * alpha * v * np.cos(np.pi * v * (1 + alpha))) / (np.pi * v * (1 - 16 * alpha * alpha * v * v))
    return np.append(h / h.sum(), 0.0)


def accepted(d):
    """A burst that passed the CRC with margins that keep false decodes out, or was recognised as a repeat."""
    return d["how"] == "match" or strict(d)


def strict(d):
    return d["how"] == "soft" and d["nerr"] <= 24 and d["cost"] < 60


def _column_costs(u, shifts=range(-8, 9)):
    """Distance of every codeword for each of the 13 columns, for every grid start 36 + shift."""
    shs = [sh for sh in shifts if 36 + sh + 156 <= len(u)]
    pos = np.array([36 + sh + np.arange(156) for sh in shs])
    c = (u[pos][:, :, None] - LEVELS[pos % 4]) ** 2
    c = c.reshape(len(shs), 12, 13, 4).transpose(0, 2, 1, 3).reshape(len(shs) * 13, 48).astype(np.float32)
    return shs, (c @ ONE_HOT_T).reshape(len(shs), 13, 4096)


def _digits(i):
    return "".join(str(v) for v in CODEWORDS[i][:6])


def _expected(p0, cw):
    return np.array([[LEVELS[(p0 + 13 * r + c) % 4][CODEWORDS[cw[c]][r]] for c in range(13)] for r in range(12)]).reshape(-1)


def _symbols(u):
    return "".join("#" if v > 20 else "+" if v > 0 else "-" if v > -20 else "=" for v in u)


def _fields(m):
    b = message_bits(m)
    x = words_hex(b)
    return dict(w=[m[6 * i:6 * i + 6] for i in range(13)], bits=b, x=x, m="".join(x))


def first_pass(bursts):
    """Decode the bursts as the finder sampled them. Returns the ones that pass the CRC."""
    out = []
    for b in bursts:
        u = np.clip(b.soft, -45, 45)
        shs, tot_all = _column_costs(u)
        bi = int(tot_all.min(2).sum(1).argmin())
        tot, sh = tot_all[bi].astype(float), shs[bi]
        cw = tot.argmin(1)
        m = "".join(_digits(i) for i in cw)
        if not crc_ok(m):
            continue
        p0 = 36 + sh
        nerr = int((_expected(p0, cw) != b.hard[p0:p0 + 156]).sum())
        d = dict(ts=b.ts, t=b.ts / RATE, q=b.q, sh=sh, how="soft", cost=float(tot[np.arange(13), cw].sum() / 156), nerr=nerr,
                 sym=_symbols(b.hard)[sh:] if sh >= 0 else "?" * (-sh) + _symbols(b.hard))
        d.update(_fields(m))
        out.append(d)
    return out


def decode(x, bursts, sps, kind="audio", grid=True, match=True, match_cost=150.0, match_lead=90.0):
    """Decode all bursts of a recording.

    x: discriminator signal at 12 kHz, bursts: output of the finder, sps: samples per symbol.
    Returns (results sorted by time, counters). Each result is a dictionary; use accepted() to filter.
    """
    x = np.clip(np.asarray(x, np.float64), -4000, 4000)
    rows = [[b.ts, b.q, b.dc, b.level] for b in bursts]
    dec = {d["ts"]: d for d in first_pass(bursts) if strict(d)}
    frame = 672 * sps

    def interp(tt):
        i = np.floor(tt).astype(int)
        w = interp_weights(tt - i)
        return (x[np.clip(i[:, None] + _K[None, :], 0, len(x) - 1)] * w).sum(1)

    def feats(ts, dc, sc, n0=0, n1=200):
        k = np.arange(n0, n1)
        tt = (ts + k[:, None] * sps + _J[None, :] * sps).ravel()
        return ((interp(tt) - dc) / sc).reshape(len(k), NTAPS)

    def equalised(ts, dc, sc, w):
        return np.clip((np.c_[feats(ts, dc, sc, 0, 236), np.ones(236)] @ w) * 10, -45, 45)

    def ideal(d):
        p0 = 36 + d["sh"]
        idx = [int(w, 4) for w in d["w"]]
        return p0, _expected(p0, idx) / 10

    def fit(a, b):
        a, b = np.vstack(a), np.concatenate(b)
        w = np.linalg.lstsq(np.c_[a, np.ones(len(a))], b, rcond=None)[0]
        return w

    def best_words(u):
        shs, tot = _column_costs(u)
        mins = tot.min(2).sum(1)
        bi = int(mins.argmin())
        cw = tot[bi].argmin(1)
        return shs[bi], float(mins[bi]) / 156, "".join(_digits(i) for i in cw), cw

    def refit(u):
        v = u[24:190]
        for _ in range(2):
            lv = np.where(v > 20, 30, np.where(v > 0, 10, np.where(v > -20, -10, -30)))
            g, o = np.linalg.lstsq(np.c_[v, np.ones(len(v))], lv, rcond=None)[0]
            if not 0.5 < g < 2:
                break
            u = u * g + o
            v = u[24:190]
        return np.clip(u, -45, 45)

    def one(r, w, dts=(0.0,)):
        ts, q, dc, sc = r[0], r[1], r[2], r[3] / 3
        best = None
        for dt in dts:
            u = refit(equalised(ts + dt, dc, sc, w)) if len(dts) > 1 else equalised(ts, dc, sc, w)
            sh, cost, m, cw = best_words(u)
            if best is None or cost < best[0]:
                best = (cost, sh, m, cw, u, dt)
        cost, sh, m, cw, u, dt = best
        hard = np.where(u > 20, 30, np.where(u > 0, 10, np.where(u > -20, -10, -30)))
        p0 = 36 + sh
        nerr = int((_expected(p0, cw) != hard[p0:p0 + 156]).sum())
        sym = _symbols(u)
        d = dict(ts=ts, t=ts / RATE, q=q, sh=sh, how="soft" if crc_ok(m) else "fail", cost=cost, nerr=nerr, dt=dt,
                 sym=sym[sh:] if sh >= 0 else "?" * (-sh) + sym)
        d.update(_fields(m))
        return d

    # pass 1: fit the equaliser on bursts that decode as they are, then decode every burst through it
    a, b, n = [], [], 0
    for r in rows:
        d = dec.get(r[0])
        if d is not None and d["nerr"] <= 12:
            p0, lv = ideal(d)
            if p0 < 0:
                continue
            a.append(feats(r[0], r[2], r[3] / 3, p0, p0 + 156))
            b.append(lv)
            n += 1
            if n >= 600:
                break
    if n >= 150:
        w = fit(a, b)
    else:
        # too few bursts to fit: take whichever stored response reads this receiver best
        sample = rows[::max(1, len(rows) // 300)]
        options = [STORED_TAPS[kind]] + [v for k, v in STORED_TAPS.items() if k != kind] + [_flat_taps(sps)]
        w = max(options, key=lambda t: sum(strict(one(r, t)) for r in sample)) if sample else options[0]
    res = [one(r, w) for r in rows]
    n_pass1 = sum(strict(d) for d in res)

    # pass 2: refit on everything pass 1 decoded, then retry the failures with a timing search
    a, b = [], []
    for r, d in zip(rows, res):
        if strict(d) and d["nerr"] <= 16 and 36 + d["sh"] >= 0:
            p0, lv = ideal(d)
            a.append(feats(r[0], r[2], r[3] / 3, p0, p0 + 156))
            b.append(lv)
        if len(a) >= 2500:
            break
    w2 = fit(a, b) if len(a) >= 100 else w
    res = [d if strict(d) else one(r, w2, (-0.5, -0.25, 0.0, 0.25, 0.5)) for r, d in zip(rows, res)]
    for i, (r, d) in enumerate(zip(rows, res)):
        if not strict(d) and r[0] in dec:                    # readable as sampled by the finder: keep that reading
            res[i] = dict(dec[r[0]], dt=0.0)

    # pass 3: towers send on a strict frame beat, so each decoded burst predicts where its neighbours are;
    # positions the finder missed are decoded directly, without a preamble
    n_grid = 0
    fr = frame
    end = len(x) - 320 * sps
    if grid and sum(strict(d) for d in res) >= 5:
        good = sorted((d["ts"], r[2], r[3]) for r, d in zip(rows, res) if strict(d))
        gts = np.array([g[0] for g in good])
        gaps = np.diff(gts)
        nn = np.round(gaps / frame)
        ok = (nn >= 1) & (nn <= 6) & (np.abs(gaps - nn * frame) < 30)
        if ok.sum() >= 5:
            fr = float(np.median(gaps[ok] / nn[ok]))
        known = np.array(sorted(r[0] for r in rows))
        tried, queue = set(), list(good)
        while queue:
            ts0, dc0, lv0 = queue.pop()
            for k in (1, -1, 2, -2):
                tp = ts0 + k * fr
                key = int(round(tp / 40))
                if tp < 50 or tp > end or key in tried or key - 1 in tried or key + 1 in tried:
                    continue
                tried.add(key)
                j = np.searchsorted(known, tp)
                if (j < len(known) and abs(known[j] - tp) < 60) or (j > 0 and abs(known[j - 1] - tp) < 60):
                    continue
                v = equalised(tp, dc0, lv0 / 3, w2)[40:190]
                if np.sqrt((np.minimum(np.abs(np.abs(v) - 10), np.abs(np.abs(v) - 30)) ** 2).mean()) > 6.6:
                    continue                                 # nothing but noise there
                r0 = [float("%.2f" % tp), 0.0, dc0, lv0]
                d = one(r0, w2, (-1.0, -0.5, 0.0, 0.5, 1.0))
                if strict(d):
                    d["ts"] = tp + d["dt"]
                    d["t"] = d["ts"] / RATE
                    d["grid"] = True
                    res.append(d)
                    rows.append(r0)
                    n_grid += 1
                    queue.append((d["ts"], dc0, lv0))
                    known = np.insert(known, np.searchsorted(known, tp), tp)
        order = np.argsort([d["ts"] for d in res])
        res, rows = [res[i] for i in order], [rows[i] for i in order]

    # pass 4: weak bursts on the towers' timing are compared with messages this recording already produced
    # (and with the clock message for the second they fall in). A match needs a low distance and a clear
    # lead over every message of another kind.
    n_match = 0
    good = [d for d in res if strict(d)]
    if match and len(good) >= 20:
        def ideal4(ws):
            idx = [int(wd, 4) for wd in ws]
            return np.array([_expected(ph, idx) for ph in range(4)], np.float32)

        def kind_of(d):
            xw = d["x"]
            if xw[0] == "9fb":
                return "time"
            if xw[0] == "000" and xw[1] == "000" and xw[11] == "018" and d["m"][:38] != "0" * 38:
                return "sys" + xw[2] + xw[3]
            return "m" + d["m"][:36]

        def clock_value(d):
            xw = d["x"]
            return (int(xw[3], 16) << 13) | (int(xw[4], 16) << 1) | (int(xw[5], 16) >> 11)

        count = collections.Counter(tuple(d["w"]) for d in good)
        first = {tuple(d["w"]): d for d in good}
        cands = [(k, kind_of(first[k])) for k, c in count.items() if c >= 2 and kind_of(first[k]) != "time"]
        clocks = [d for d in good if d["x"][0] == "9fb"]
        toff = float(np.median([clock_value(d) - d["t"] for d in clocks])) if clocks else None
        template = clocks[0]["bits"][:152] if clocks else None
        tcache, lcache = {}, {}
        l4 = np.array([ideal4(k) for k, _ in cands]) if cands else np.zeros((0, 4, 156), np.float32)
        kl = [c for _, c in cands]

        def time_cands(t):
            out = []
            if template is None:
                return out
            for tv in (int(np.floor(t + toff)) + q for q in (-1, 0, 1, 2)):
                if tv not in tcache:
                    bits = list(template)
                    bits[36:61] = [(tv >> (24 - i)) & 1 for i in range(25)]
                    tcache[tv] = with_crc(bits)
                if tcache[tv]:
                    out.append(tuple(tcache[tv]))
            return out

        def try_match(ts, dc, lv):
            best = None
            tc = time_cands(ts / RATE)
            for k in tc:
                if k not in lcache:
                    lcache[k] = ideal4(k)
            ll = np.concatenate([l4, np.array([lcache[k] for k in tc], np.float32).reshape(-1, 4, 156)]) if tc else l4
            kinds = kl + ["time"] * len(tc)
            keys = [k for k, _ in cands] + tc
            if not len(ll):
                return None
            for dt in (-0.5, 0.0, 0.5):
                u = refit(equalised(ts + dt, dc, lv / 3, w2))
                for sh in (-2, -1, 0, 1, 2):
                    p0 = 36 + sh
                    cost = ((ll[:, p0 % 4, :] - u[p0:p0 + 156].astype(np.float32)) ** 2).mean(1)
                    i = int(cost.argmin())
                    other = [cost[j] for j in range(len(cost)) if kinds[j] != kinds[i]]
                    lead = (min(other) - cost[i]) if other else 999.0
                    if best is None or cost[i] < best[0]:
                        best = (float(cost[i]), float(lead), keys[i], sh, u, dt)
            return best

        def matched(mm, ts):
            ws = list(mm[2])
            sym, sh = _symbols(mm[4]), mm[3]
            d = dict(ts=ts, t=ts / RATE, q=0.0, sh=sh, how="match", cost=mm[0], lead=mm[1], nerr=99, dt=mm[5],
                     sym=sym[sh:] if sh >= 0 else "?" * (-sh) + sym)
            d.update(_fields("".join(ws)))
            return d

        gt = np.array(sorted(d["ts"] for d in good))
        for r, d in zip(rows, res):
            if strict(d) or d.get("grid"):
                continue
            j = np.searchsorted(gt, d["ts"])
            q = min((q for q in (j - 1, j) if 0 <= q < len(gt)), key=lambda q: abs(gt[q] - d["ts"]))
            g = abs(gt[q] - d["ts"])
            if not (g < 8 * frame and abs(g / frame - round(g / frame)) < 0.04):
                continue
            mm = try_match(d["ts"], r[2], r[3])
            if mm and mm[0] < match_cost and mm[1] > match_lead:
                new = matched(mm, d["ts"])
                new["q"], new["nerr"], new["dt"] = d["q"], d["nerr"], d["dt"]
                d.clear()
                d.update(new)
                n_match += 1
        # pass 5: the same recognition at positions the finder never flagged, walking along the frame beat
        acc = sorted((d["ts"], r[2], r[3]) for r, d in zip(rows, res) if accepted(d))
        known = np.array(sorted(r[0] for r in rows))
        tried, queue, new_rows, new_res = set(), list(acc), [], []
        while queue:
            ts0, dc0, lv0 = queue.pop()
            for k in (1, -1, 2, -2):
                tp = ts0 + k * fr
                key = int(round(tp / 40))
                if tp < 50 or tp > end or key in tried or key - 1 in tried or key + 1 in tried:
                    continue
                tried.add(key)
                j = np.searchsorted(known, tp)
                if (j < len(known) and abs(known[j] - tp) < 60) or (j > 0 and abs(known[j - 1] - tp) < 60):
                    continue
                mm = try_match(tp, dc0, lv0)
                if mm and mm[0] < match_cost and mm[1] > match_lead:
                    tn = tp + mm[5]
                    d = matched(mm, tn)
                    d["grid"] = True
                    new_res.append(d)
                    new_rows.append([float("%.2f" % tn), 0.0, dc0, lv0])
                    n_match += 1
                    queue.append((tn, dc0, lv0))
                    known = np.insert(known, np.searchsorted(known, tp), tp)
        if new_res:
            res += new_res
            rows += new_rows
            order = np.argsort([d["ts"] for d in res])
            res, rows = [res[i] for i in order], [rows[i] for i in order]

    # how many undecoded detections sit on the frame beat of a decoded burst (real bursts) and how many do not
    gt = np.array(sorted(d["ts"] for d in res if accepted(d)))
    on_beat = 0
    if len(gt):
        for d in res:
            if d.get("grid") or accepted(d):
                continue
            j = np.searchsorted(gt, d["ts"])
            q = min((q for q in (j - 1, j) if 0 <= q < len(gt)), key=lambda q: abs(gt[q] - d["ts"]))
            g = abs(gt[q] - d["ts"])
            if g < 8 * frame and abs(g / frame - round(g / frame)) < 0.04:
                on_beat += 1
    n_ok = sum(accepted(d) for d in res)
    detections = sum(1 for d in res if not d.get("grid"))
    stats = dict(decoded=n_ok, found=len(bursts), direct=len(dec), equalised=n_pass1, by_timing=n_grid, repeats=n_match,
                 unreadable=on_beat, noise=detections - (n_ok - n_grid - sum(1 for d in res if d.get("grid") and d["how"] == "match")) - on_beat)
    return res, stats
