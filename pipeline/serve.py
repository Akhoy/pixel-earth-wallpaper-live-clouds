"""Serve output/latest, rewriting root.json's baseUrl to match the request host.

run_pipeline.py has to bake an absolute baseUrl into root.json (the app
follows it to fetch the DDS tiles), but this machine's LAN address keeps
changing as it moves between ethernet/wifi/hotspot -- and a stale IP in
there means the device resolves the tiles to nothing and silently renders
the bundled fallback texture instead.

So rather than regenerate the whole pipeline whenever the address changes,
rewrite baseUrl on the fly to whatever Host the client actually connected
to. Requests from the phone, from localhost, or from any new network all
get a baseUrl that points back at the same place they just reached.

Usage: venv/bin/python3 pipeline/serve.py [--port 8765] [--dir output/latest]
"""
import argparse
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent


class RootJsonRewritingHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        name = self.path.split("?")[0].rsplit("/", 1)[-1]
        if name in ("root.json", "root_day.json"):
            return self._serve_rewritten_root(name)
        return super().do_GET()

    def _serve_rewritten_root(self, name):
        root_file = Path(self.directory) / name
        try:
            data = json.loads(root_file.read_text())
        except (OSError, ValueError) as exc:
            self.send_error(500, f"cannot read root.json: {exc}")
            return

        base = data.get("baseUrl", "")
        host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_port}"
        # keep only the path portion of whatever was baked in, and re-point
        # it at the host this request actually arrived on
        path = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+", "", base)
        data["baseUrl"] = f"http://{host}{path}"

        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        print(f"[serve] {name} -> {data['baseUrl']}  (client {self.client_address[0]})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", type=str, default=str(ROOT / "output" / "latest"))
    args = ap.parse_args()

    handler = partial(RootJsonRewritingHandler, directory=args.dir)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"[serve] serving {args.dir} on 0.0.0.0:{args.port}", flush=True)
    server.serve_forever()
