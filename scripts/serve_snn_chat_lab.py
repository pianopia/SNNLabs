#!/usr/bin/env python3
"""Serve snn-chat-lab and convert uploaded DST-SNN .pt checkpoints."""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CHAT_DIR = ROOT / "snn-chat-lab"
sys.path.insert(0, str(ROOT))

from src.dst_snn.chat_export import checkpoint_to_chat_payload  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CHAT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type, x-filename")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/convert-pt":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0:
                raise ValueError("empty body")
            body = self.rfile.read(length)
            filename = unquote(self.headers.get("x-filename", "uploaded.pt"))
            suffix = ".pt" if filename.endswith(".pt") else ".pt"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(body)
                tmp.flush()
                payload = checkpoint_to_chat_payload(Path(tmp.name))
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            encoded = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"snn-chat-lab: http://127.0.0.1:{port}")
    print("Upload .pt from the UI to convert it locally.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
