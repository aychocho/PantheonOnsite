#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["aiohttp"]
# ///
"""Meta Quest controller pose collector.

Serves a WebXR page over HTTPS (WebXR requires a secure context) and receives
controller poses back over a WebSocket on the same port. Each sample is:
  - logged to a per-session CSV under vr/logs/
  - rebroadcast as JSON over UDP for downstream consumers (end-effector control)

Run:
    uv run quest_server.py
Then on the Quest browser open https://<this-machine-LAN-IP>:8443 , accept the
self-signed cert warning (Advanced -> Proceed), and tap "Enter VR".
"""

import argparse
import asyncio
import csv
import json
import socket
import ssl
import subprocess
import time
from pathlib import Path

from aiohttp import web, WSMsgType

HERE = Path(__file__).parent

CSV_FIELDS = ["t_recv", "t_quest", "hand", "px", "py", "pz",
              "qx", "qy", "qz", "qw", "trigger", "grip",
              "thumb_x", "thumb_y", "buttons"]


def ensure_cert(cert: Path, key: Path):
    if cert.exists() and key.exists():
        return
    print("Generating self-signed cert...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert),
        "-days", "365", "-subj", "/CN=quest-teleop",
    ], check=True, capture_output=True)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Collector:
    def __init__(self, udp_addr, log_dir: Path):
        self.udp_addr = udp_addr
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp_addr else None
        log_dir.mkdir(exist_ok=True)
        path = log_dir / time.strftime("quest_%Y%m%d_%H%M%S.csv")
        self.csv_file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.count = 0
        self.last_print = 0.0
        print(f"Logging to {path}" + (f", rebroadcasting UDP to {udp_addr[0]}:{udp_addr[1]}" if udp_addr else ""))

    def handle(self, msg: dict):
        t_recv = time.time()
        for c in msg.get("controllers", []):
            row = {
                "t_recv": f"{t_recv:.6f}", "t_quest": msg.get("t", ""),
                "hand": c["hand"],
                "px": c["pos"][0], "py": c["pos"][1], "pz": c["pos"][2],
                "qx": c["quat"][0], "qy": c["quat"][1], "qz": c["quat"][2], "qw": c["quat"][3],
                "trigger": c.get("trigger", 0), "grip": c.get("grip", 0),
                "thumb_x": c.get("thumb", [0, 0])[0], "thumb_y": c.get("thumb", [0, 0])[1],
                "buttons": c.get("buttons", 0),
            }
            self.writer.writerow(row)
            self.count += 1
        if self.udp:
            msg["t_recv"] = t_recv
            self.udp.sendto(json.dumps(msg).encode(), self.udp_addr)
        if t_recv - self.last_print > 1.0:
            self.last_print = t_recv
            hands = {c["hand"]: c for c in msg.get("controllers", [])}
            parts = [f"{h}: pos=({c['pos'][0]:+.3f},{c['pos'][1]:+.3f},{c['pos'][2]:+.3f}) trig={c.get('trigger', 0):.2f}"
                     for h, c in sorted(hands.items())]
            print(f"[{self.count:6d} samples] " + "  |  ".join(parts), flush=True)

    def close(self):
        self.csv_file.close()
        if self.udp:
            self.udp.close()


async def ws_handler(request):
    collector = request.app["collector"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print("Quest connected")
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                collector.handle(json.loads(msg.data))
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Bad message: {e}")
        elif msg.type == WSMsgType.ERROR:
            break
    collector.csv_file.flush()
    print("Quest disconnected")
    return ws


async def index(request):
    return web.FileResponse(HERE / "index.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--udp", default="127.0.0.1:5557",
                    help="host:port for UDP JSON rebroadcast, or 'off' (default: 127.0.0.1:5557)")
    args = ap.parse_args()

    udp_addr = None
    if args.udp != "off":
        host, port = args.udp.rsplit(":", 1)
        udp_addr = (host, int(port))

    cert, key = HERE / "cert.pem", HERE / "key.pem"
    ensure_cert(cert, key)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(cert, key)

    app = web.Application()
    app["collector"] = Collector(udp_addr, HERE / "logs")
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)

    print(f"\nOn the Quest browser, open:  https://{lan_ip()}:{args.port}\n"
          f"(accept the self-signed cert warning, then tap 'Enter VR')\n")
    try:
        web.run_app(app, port=args.port, ssl_context=ssl_ctx, print=None)
    finally:
        app["collector"].close()


if __name__ == "__main__":
    main()
