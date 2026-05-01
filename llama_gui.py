#!/usr/bin/env python3
"""Web GUI launcher for llama.cpp profiles. Stdlib only."""
import http.server
import json
import os
import re
import shlex
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
INDEX_PATH = SCRIPT_DIR / "index.html"

DEFAULT_CONFIG = {
    "port": 7777,
    "host": "127.0.0.1",
    "llama_bin": "llama-server",
    "profiles_dir": "./profiles",
    "open_browser_on_launch": True,
    "auto_open_model_ui": False,
}

# Fields that require a restart to take effect.
RESTART_FIELDS = {"port", "host"}

# ---- config -----------------------------------------------------------------

def load_config():
    """Load config, creating it from defaults if missing. Fills in any missing keys."""
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"warning: config.json is invalid ({e}); using defaults", file=sys.stderr)
        return dict(DEFAULT_CONFIG)
    # Merge with defaults so newly-added fields appear without manual editing.
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
    return merged

def save_config(cfg):
    """Atomic write so a crash mid-write doesn't corrupt the file."""
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, CONFIG_PATH)

def profiles_path(cfg):
    p = Path(cfg["profiles_dir"]).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p

def llama_bin(cfg):
    """Expand ~ in the binary path. Bare command names (no slash) are left
    alone so PATH lookup still works."""
    b = cfg["llama_bin"]
    return str(Path(b).expanduser()) if "~" in b or "/" in b else b

# ---- profile parsing --------------------------------------------------------

META_KEYS = ("name", "description")
META_RE = re.compile(r"^#\s*(\w+)\s*:\s*(.*)$")

def _parse_meta_line(line):
    """If `line` is a meta comment like '# name: foo', return (key, value).
    Otherwise return None. Only keys in META_KEYS are recognized."""
    m = META_RE.match(line)
    if not m:
        return None
    key, value = m.group(1).lower(), m.group(2).strip()
    if key in META_KEYS:
        return key, value
    return None

def parse_profile(path):
    """Return (args, port, meta) from a .conf file.

    - Comments starting with # are stripped from arg parsing.
    - Special comments like '# name: Foo' or '# description: ...' are extracted
      into the meta dict (keys: name, description). They must appear on their
      own line, anywhere in the file. First occurrence wins.
    - Tokens starting with ~ are expanded to the user's home directory.
    """
    args, port, meta = [], None, {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            kv = _parse_meta_line(stripped)
            if kv and kv[0] not in meta:
                meta[kv[0]] = kv[1]
            continue
        tokens = [
            os.path.expanduser(t) if t.startswith("~") else t
            for t in shlex.split(stripped)
        ]
        args += tokens
        if "--port" in tokens:
            try:
                port = int(tokens[tokens.index("--port") + 1])
            except (IndexError, ValueError):
                pass
    return args, port, meta

def read_profile_meta(path):
    """Cheap version of parse_profile that only reads meta fields.
    Used for the listing endpoint so we don't shlex.split every profile
    on every poll. Stops as soon as all known META_KEYS are found."""
    meta = {}
    try:
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped or not stripped.startswith("#"):
                    continue
                kv = _parse_meta_line(stripped)
                if kv and kv[0] not in meta:
                    meta[kv[0]] = kv[1]
                    if len(meta) == len(META_KEYS):
                        break
    except OSError:
        pass
    return meta

# ---- process state ----------------------------------------------------------

class State:
    """Shared mutable state for the running llama-server, if any."""
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.profile_name = None
        self.profile_port = None

    def is_running(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return True
            # Reap finished process.
            self.proc = None
            self.profile_name = None
            self.profile_port = None
            return False

    def start(self, name, args, port, binary):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = subprocess.Popen([binary] + args)
            self.profile_name = name
            self.profile_port = port

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            self.profile_name = None
            self.profile_port = None

state = State()

# ---- HTTP -------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep stdout clean for llama-server logs

    def _send(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, code=200):
        self._send(code, "application/json", json.dumps(payload))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return None

    # -- GET --
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_PATH.read_bytes())
        elif path == "/api/state":
            cfg = load_config()
            running = state.is_running()
            try:
                profile_files = sorted(profiles_path(cfg).glob("*.conf"))
            except FileNotFoundError:
                profile_files = []
            profiles = []
            for p in profile_files:
                meta = read_profile_meta(p)
                profiles.append({
                    "id": p.stem,
                    "name": meta.get("name") or p.stem,
                    "description": meta.get("description", ""),
                })
            self._send_json({
                "profiles": profiles,
                "running": state.profile_name if running else None,
                "port": state.profile_port if running else None,
                "auto_open_model_ui": cfg["auto_open_model_ui"],
            })
        elif path == "/api/config":
            cfg = load_config()
            self._send_json({
                "config": cfg,
                "defaults": DEFAULT_CONFIG,
                "restart_fields": sorted(RESTART_FIELDS),
            })
        else:
            self.send_error(404)

    # -- POST --
    def do_POST(self):
        u = urlparse(self.path)
        params = parse_qs(u.query)
        if u.path == "/api/start":
            name = params.get("profile", [""])[0]
            cfg = load_config()
            conf = profiles_path(cfg) / f"{name}.conf"
            if not conf.is_file():
                return self._send_json({"ok": False, "error": "no such profile"}, 404)
            args, port, _meta = parse_profile(conf)
            try:
                state.start(name, args, port, llama_bin(cfg))
            except FileNotFoundError:
                return self._send_json({"ok": False, "error": f"binary not found: {llama_bin(cfg)}"}, 500)
            self._send_json({"ok": True, "port": port})
        elif u.path == "/api/stop":
            state.stop()
            self._send_json({"ok": True})
        elif u.path == "/api/config":
            body = self._read_json()
            if body is None:
                return self._send_json({"ok": False, "error": "invalid JSON"}, 400)
            cfg = load_config()
            # Only accept known keys; ignore others.
            updated = dict(cfg)
            for k, v in body.items():
                if k in DEFAULT_CONFIG:
                    updated[k] = v
            # Light type validation against defaults.
            for k, v in updated.items():
                if not isinstance(v, type(DEFAULT_CONFIG[k])):
                    return self._send_json(
                        {"ok": False, "error": f"wrong type for {k}"}, 400)
            save_config(updated)
            needs_restart = any(updated[k] != cfg[k] for k in RESTART_FIELDS)
            self._send_json({"ok": True, "needs_restart": needs_restart})
        else:
            self.send_error(404)

# ---- main -------------------------------------------------------------------

class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    cfg = load_config()
    profiles_dir = profiles_path(cfg)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        sys.exit(f"missing {INDEX_PATH}")

    url = f"http://{cfg['host']}:{cfg['port']}"
    print(f"→ launcher GUI at {url}")
    print(f"  profiles: {profiles_dir}")
    print(f"  binary:   {llama_bin(cfg)}")

    if cfg["open_browser_on_launch"]:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        with ReusableTCPServer((cfg["host"], cfg["port"]), Handler) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        state.stop()

if __name__ == "__main__":
    main()
