#!/usr/bin/env python3
"""Smoke-test the stdlib OpenAI-compatible HTTP fallback client."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.llm_json_utils import OpenAICompatibleHTTPClient, call_llm_json, parse_json_object


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, fmt, *args):  # pragma: no cover - keep test output stable
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        type(self).requests.append({
            "path": self.path,
            "payload": payload,
            "authorization": self.headers.get("Authorization"),
        })
        content = '{"status":"ok","value":"0x00000020"}'
        response = {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> int:
    Handler.requests = []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleHTTPClient(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/compatible-mode/v1",
            timeout=5.0,
        )
        response = call_llm_json(
            client=client,
            model="qwen-plus",
            messages=[{"role": "user", "content": "return json"}],
            max_tokens=64,
            temperature=0.0,
            repair_prompt="return json",
            parse_response=parse_json_object,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    content = response.choices[0].message.content
    parsed = parse_json_object(content)
    checks = {
        "parsed_ok": parsed.get("status") == "ok" and parsed.get("value") == "0x00000020",
        "request_count_ok": len(Handler.requests) == 1,
        "path_ok": Handler.requests[0]["path"].endswith("/chat/completions"),
        "auth_ok": Handler.requests[0]["authorization"] == "Bearer test-key",
        "model_ok": Handler.requests[0]["payload"].get("model") == "qwen-plus",
        "json_mode_ok": Handler.requests[0]["payload"].get("response_format") == {"type": "json_object"},
    }
    print(json.dumps({"checks": checks, "request": Handler.requests[0]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
