"""Local web page: decode recordings dropped on the page, or a live audio input."""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .live import Session
from .messages import events_of
from .pipeline import decode_file

PAGE = Path(__file__).with_name("page.html")
_lock = threading.Lock()
_live = {"session": None}
_file = {"done": None}                   # share of the current file decode, for the page's progress line


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, kind="application/json", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/progress":
            self._send({"done": _file["done"]})
        else:
            self._send(b"", "text/plain", 404)

    def do_POST(self):
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if url.path == "/file":
                try:
                    results, info = decode_file(body, progress=lambda f: _file.update(done=round(f, 3)))
                finally:
                    _file["done"] = None
                events, towers = events_of(results, info["sps"])
                self._send({"info": info, "towers": towers, "events": events})
            elif url.path == "/audio":
                rate = int(parse_qs(url.query).get("rate", ["48000"])[0])
                samples = np.frombuffer(body[:len(body) // 2 * 2], "<i2").astype(np.float64) / 32768
                with _lock:
                    s = _live["session"]
                    if s is None:
                        s = _live["session"] = Session()
                    events = s.feed(samples, rate)
                    self._send({"events": events, "towers": s.tracker.tower_list(), "status": s.status})
            elif url.path == "/reset":
                with _lock:
                    _live["session"] = None
                self._send({})
            else:
                self._send(b"", "text/plain", 404)
        except (ValueError, OSError) as e:
            self._send({"error": str(e)}, status=400)


def serve(port=8790, open_browser=True):
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if server is None:
        print("no free port near %d" % port)
        return 1
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    print("eDMR decoder running at %s  (Ctrl+C to stop)" % url)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
