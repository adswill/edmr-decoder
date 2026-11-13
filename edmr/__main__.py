"""Command line: decode recordings, or open the web page when no file is given."""
import argparse
import json
import sys
import time

from . import __version__, decode_file
from .messages import events_of


def _clock(unix):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(unix)) if unix else "-"


def _progress(path):
    """Percentage on stderr while a file decodes, when stderr is a terminal."""
    if not sys.stderr.isatty():
        return None
    last = [-1]

    def show(f):
        pct = int(f * 100)
        if pct != last[0]:
            last[0] = pct
            sys.stderr.write("\r%s: %d %%" % (path, pct) if pct < 100 else "\r\033[K")
            sys.stderr.flush()
    return show


def print_report(path, events, towers, info, show_idle=False, raw=False, summary=False):
    kind = "IQ" if info["kind"] == "iq" else "FM audio"
    print("%s  (%s, %d Hz, %.0f s)" % (path, kind, info["rate"], info["seconds"]))
    print("%d bursts decoded: %d found by their start marker, %d more by frame timing, %d weak repeats; %d unreadable"
          % (info["decoded"], info["decoded"] - info["by_timing"] - info["repeats"], info["by_timing"], info["repeats"], info["unreadable"]))
    print("symbol clock %.5f samples%s" % (info["sps"], ", inverted polarity" if info["inverted"] else ""))
    if towers:
        print("\nTowers")
        print("  %-9s %-9s %-4s %-5s %-20s %-7s %s" % ("network", "area/site", "slot", "msgs", "clock (UTC)", "offset", "names"))
        for t in towers:
            off = "-" if t["utc"] is None else "%+d:%02d" % (t["utc"] // 60, t["utc"] % 60)
            print("  %-9s %02d/%-6d %-4d %-5d %-20s %-7s %s" % (t["network"] or t["net"], t["area"], t["site"], t["slot"], t["count"],
                                                              _clock(t["unix"]), off, ", ".join(t["names"])))
    if summary:
        return
    print("\nMessages")
    print("%9s  %-4s %-7s %-16s %s" % ("time s", "slot", "tower", "type", "content"))
    for e in events:
        if e["kind"] == "idle" and not show_idle:
            continue
        tower = "%02d/%d" % tuple(int(v) for v in e["tower"].split("-")[1:]) if "tower" in e else ""
        line = "%9.2f  %-4d %-7s %-16s %s" % (e["t"], e["slot"], tower, e["type"] + ("~" if e["repeat"] else ""), e["text"])
        print(line.rstrip() + ("\n" + " " * 11 + e["raw"] if raw else ""))


def main(argv=None):
    p = argparse.ArgumentParser(prog="edmr", description="eDMR control channel decoder")
    p.add_argument("files", nargs="*", help="WAV recordings (FM audio or IQ)")
    p.add_argument("--summary", action="store_true", help="towers and counts only")
    p.add_argument("--idle", action="store_true", help="also list idle bursts")
    p.add_argument("--raw", action="store_true", help="show the 13 data words of each burst")
    p.add_argument("--json", action="store_true", help="print everything as JSON")
    p.add_argument("--sps", type=float, help="samples per symbol at 12 kHz, if the automatic value is wrong")
    p.add_argument("--ui", action="store_true", help="open the web page (default when no file is given)")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--no-browser", action="store_true", help="with --ui: do not open a browser window")
    p.add_argument("--selftest", action="store_true", help="decode a generated test signal and exit")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    if a.selftest:
        from .selftest import run
        return run()
    if a.ui or not a.files:
        from .web import serve
        return serve(a.port, not a.no_browser)
    status = 0
    for path in a.files:
        try:
            results, info = decode_file(path, sps=a.sps, progress=_progress(path))
        except (OSError, ValueError) as e:
            print("%s: %s" % (path, e), file=sys.stderr)
            status = 1
            continue
        events, towers = events_of(results, info["sps"])
        if a.json:
            print(json.dumps({"file": path, "info": info, "towers": towers, "messages": events}, indent=1))
        else:
            print_report(path, events, towers, info, a.idle, a.raw, a.summary)
            print()
    return status


if __name__ == "__main__":
    sys.exit(main())
