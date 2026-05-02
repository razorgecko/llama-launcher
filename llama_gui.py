#!/usr/bin/env python3
"""Web GUI launcher for llama.cpp profiles. Stdlib only."""
import collections
import http.server
import json
import logging
import os
import re
import shlex
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_PATH = SCRIPT_DIR / "index.html"

_XDG_CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
_XDG_DATA_HOME   = Path(os.environ.get("XDG_DATA_HOME")   or Path.home() / ".local" / "share")

CONFIG_PATH = _XDG_CONFIG_HOME / "llama-launcher" / "config.json"
STATE_PATH  = _XDG_CONFIG_HOME / "llama-launcher" / "state.json"
LOG_PATH    = _XDG_DATA_HOME   / "llama-launcher" / "launcher.log"

DEFAULT_CONFIG = {
    "port": 7777,
    "host": "127.0.0.1",
    "llama_bin": "llama-server",
    "sd_bin": "sd-server",
    "profiles_dir": str(_XDG_CONFIG_HOME / "llama-launcher" / "profiles"),
    "open_browser_on_launch": True,
    "auto_open_model_ui": False,
}

# Fields that require a restart to take effect.
RESTART_FIELDS = {"port", "host"}

# llama-server listens on this port when --port is not specified.
LLAMA_DEFAULT_PORT = 8080

# Hosts considered safe to bind without authentication. Anything else requires
# the user to set LLAMA_LAUNCHER_ALLOW_REMOTE=1 — see the guard in main() and
# in _post_config. Every endpoint is unauthenticated, and several can spawn
# arbitrary processes, so binding to a routable address without that opt-in
# would be an RCE-by-default footgun.
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

def _is_loopback(host):
    return host in LOOPBACK_HOSTS

def _validate_binary(value):
    """True if `value` resolves to an executable file (PATH lookup or path)."""
    expanded = os.path.expanduser(value)
    if "/" in expanded:
        return os.path.isfile(expanded) and os.access(expanded, os.X_OK)
    return shutil.which(expanded) is not None

# ---- config -----------------------------------------------------------------

_cfg_lock = threading.RLock()
_cfg_cache = None
_cfg_mtime = 0.0

def load_config():
    """Load config, creating it from defaults if missing. Fills in any missing keys."""
    global _cfg_cache, _cfg_mtime
    with _cfg_lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            mtime = None
        if _cfg_cache is not None and mtime == _cfg_mtime:
            return dict(_cfg_cache)
        if mtime is None:
            save_config(DEFAULT_CONFIG)
            return dict(DEFAULT_CONFIG)
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError as e:
            logging.warning("config.json is invalid (%s); using defaults", e)
            return dict(DEFAULT_CONFIG)
        merged = dict(DEFAULT_CONFIG)
        merged.update({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
        _cfg_cache = merged
        _cfg_mtime = mtime
        return dict(merged)

def save_config(cfg):
    """Atomic write so a crash mid-write doesn't corrupt the file."""
    global _cfg_cache, _cfg_mtime
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, CONFIG_PATH)
    with _cfg_lock:
        _cfg_cache = dict(cfg)
        try:
            _cfg_mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            _cfg_mtime = 0.0

# ---- user state (favorites, last_used) -------------------------------------

class StateStore:
    """Thread-safe storage for user state (favorites, last_used).
    All mutations go through update() which atomically read-modify-writes."""
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        """Read the JSON file. Returns defaults on missing/corrupt file."""
        if not self.path.exists():
            return {"last_used": None, "favorites": []}
        try:
            s = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            logging.warning("%s is invalid (%s); resetting", self.path.name, e)
            return {"last_used": None, "favorites": []}
        return {
            "last_used": s.get("last_used") if isinstance(s.get("last_used"), str) else None,
            "favorites": [f for f in s.get("favorites", []) if isinstance(f, str)],
        }

    def save(self, s):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, indent=2) + "\n")
        os.replace(tmp, self.path)

    def update(self, mutator):
        """Atomic read-modify-write. mutator(s) mutates in place."""
        with self.lock:
            s = self.load()
            mutator(s)
            self.save(s)
            return s

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

def sd_bin(cfg):
    b = cfg["sd_bin"]
    return str(Path(b).expanduser()) if "~" in b or "/" in b else b

# ---- profile parsing --------------------------------------------------------

META_KEYS = ("name", "description", "type")
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

def parse_profile(path, meta_only=False):
    """Return (args, port, meta) from a .conf file.

    - Comments starting with # are stripped from arg parsing.
    - Special comments like '# name: Foo' or '# description: ...' are extracted
      into the meta dict (keys: name, description). They must appear on their
      own line, anywhere in the file. First occurrence wins.
    - Tokens starting with ~ are expanded to the user's home directory.
    - With meta_only=True, skips arg parsing and returns early once all
      META_KEYS are found (used by read_profile_meta for cheap listing polls).
    """
    args, port, meta = [], None, {}
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                kv = _parse_meta_line(stripped)
                if kv and kv[0] not in meta:
                    meta[kv[0]] = kv[1]
                    if meta_only and len(meta) == len(META_KEYS):
                        return args, port, meta
                continue
            if meta_only:
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
    """Thin wrapper: parse meta only, suppressing OSError for the listing poll."""
    try:
        _, _, meta = parse_profile(path, meta_only=True)
    except OSError:
        return {}
    return meta

_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\- .]*$")

def _valid_profile_id(name):
    """True if name is safe to use as a profile filename stem (no path traversal)."""
    return bool(name and _PROFILE_ID_RE.match(name) and ".." not in name)

# ---- process runner ---------------------------------------------------------

class Runner:
    """Manages the running llama-server subprocess and captures its output.

    Two locks: `lock` guards the process slot (proc/name/port). `logs_lock`
    guards the log buffer. They are separate to prevent a deadlock where
    start() holds `lock` while waiting for an old process to terminate, but
    that process's reader threads need a lock to flush their final lines.
    """
    LOG_MAX_LINES = 5000

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.profile_name = None
        self.profile_port = None
        self._ready = False
        self.logs_lock = threading.Lock()
        self.logs = collections.deque(maxlen=self.LOG_MAX_LINES)
        self._next_idx = 0

    # -- log helpers --
    def _emit(self, stream, text):
        """Append a single log line. text may contain a trailing newline; stripped."""
        with self.logs_lock:
            self.logs.append({
                "idx": self._next_idx,
                "stream": stream,
                "text": text.rstrip("\n"),
            })
            self._next_idx += 1

    def _reader(self, stream_name, fileobj):
        """Thread target: read lines from fileobj until EOF."""
        try:
            for line in iter(fileobj.readline, ""):
                self._emit(stream_name, line)
        except (OSError, ValueError):
            pass  # pipe closed, file detached, etc.
        finally:
            try:
                fileobj.close()
            except OSError:
                pass

    def get_logs(self, since=0):
        """Return (lines, next_idx) where lines have idx >= since."""
        with self.logs_lock:
            lines = [l for l in self.logs if l["idx"] >= since]
            return lines, self._next_idx

    def clear_logs(self):
        """Empty the log buffer. The index counter keeps advancing so any
        pending pollers naturally fetch nothing rather than re-fetching old
        lines."""
        with self.logs_lock:
            self.logs.clear()

    # -- process control --
    def is_running(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return True
            self.proc = None
            self.profile_name = None
            self.profile_port = None
            self._ready = False
            return False

    def _terminate_locked(self):
        """Caller must hold self.lock."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _health_poller(self, port, proc):
        """Poll /health until 200 or process exits (5-minute timeout)."""
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        with self.lock:
                            if self.proc is proc:
                                self._ready = True
                        return
            except Exception:
                pass
            time.sleep(0.5)

    def start(self, name, args, port, binary, skip_health=False):
        with self.lock:
            self._terminate_locked()
            self._ready = False
            self._emit("system", f"--- starting {name} ({binary}) ---")
            self.proc = subprocess.Popen(
                [binary] + args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                errors="replace",
            )
            self.profile_name = name
            poll_port = port or LLAMA_DEFAULT_PORT
            self.profile_port = poll_port
            for stream_name, fh in (("out", self.proc.stdout), ("err", self.proc.stderr)):
                t = threading.Thread(
                    target=self._reader, args=(stream_name, fh), daemon=True)
                t.start()
            if skip_health:
                self._ready = True
            else:
                t = threading.Thread(
                    target=self._health_poller, args=(poll_port, self.proc), daemon=True)
                t.start()

    def stop(self):
        with self.lock:
            was_running = self.proc is not None and self.proc.poll() is None
            self._terminate_locked()
            self.proc = None
            self.profile_name = None
            self.profile_port = None
            self._ready = False
        if was_running:
            self._emit("system", "--- stopped ---")

# ---- HTTP -------------------------------------------------------------------

def make_handler(runner, store):
    """Build a request handler class bound to the given runner and store."""
    class Handler(http.server.BaseHTTPRequestHandler):
        _GET_ROUTES = {
            "/":              "_get_root",
            "/favicon.svg":   "_get_favicon",
            "/api/state":     "_get_state",
            "/api/config":    "_get_config",
            "/api/logs":      "_get_logs",
            "/api/profile":   "_get_profile",
        }
        _POST_ROUTES = {
            "/api/start":      "_post_start",
            "/api/stop":       "_post_stop",
            "/api/favorite":   "_post_favorite",
            "/api/move":       "_post_move",
            "/api/config":     "_post_config",
            "/api/profile":        "_post_profile",
            "/api/profile/copy":   "_post_profile_copy",
            "/api/profile/delete": "_post_profile_delete",
            "/api/logs/clear":     "_post_logs_clear",
            "/api/quit":       "_post_quit",
        }

        def log_message(self, format, *a): pass

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

        def _dispatch(self, routes):
            parsed = urlparse(self.path)
            self._qs = parse_qs(parsed.query)
            m = routes.get(parsed.path)
            if m:
                getattr(self, m)()
            else:
                self.send_error(404)

        def _check_origin(self):
            """Block cross-origin POSTs to defend against CSRF from a browser.
            Browsers always send Origin on POST; same-origin requests have it
            matching the Host header. Tools without an Origin (curl, scripts)
            are out of scope for the browser threat model and are allowed."""
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin == f"http://{self.headers.get('Host', '')}"

        def do_GET(self):  self._dispatch(self._GET_ROUTES)
        def do_POST(self):
            if not self._check_origin():
                return self.send_error(403, "cross-origin POST blocked")
            self._dispatch(self._POST_ROUTES)

        # -- GET handlers --

        def _get_root(self):
            self._send(200, "text/html; charset=utf-8", INDEX_PATH.read_bytes())

        def _get_favicon(self):
            p = SCRIPT_DIR / "llama-icon.svg"
            self._send(200, "image/svg+xml", p.read_bytes())

        def _get_state(self):
            cfg = load_config()
            running = runner.is_running()
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
            # Sort alphabetically by display name (case-insensitive).
            profiles.sort(key=lambda p: p["name"].lower())
            valid_ids = {p["id"] for p in profiles}
            us = store.load()
            last_used = us["last_used"] if us["last_used"] in valid_ids else None
            favorites = [f for f in us["favorites"] if f in valid_ids]
            if last_used != us["last_used"] or favorites != us["favorites"]:
                store.update(lambda s: (
                    s.update({"last_used": last_used, "favorites": favorites})
                ))
            self._send_json({
                "profiles": profiles,
                "running": runner.profile_name if running else None,
                "port": runner.profile_port if running else None,
                "ready": runner._ready if running else None,
                "auto_open_model_ui": cfg["auto_open_model_ui"],
                "last_used": last_used,
                "favorites": favorites,
            })

        def _get_config(self):
            cfg = load_config()
            self._send_json({
                "config": cfg,
                "defaults": DEFAULT_CONFIG,
                "restart_fields": sorted(RESTART_FIELDS),
            })

        def _get_logs(self):
            try:
                since = int(self._qs.get("since", ["0"])[0])
            except ValueError:
                since = 0
            lines, next_idx = runner.get_logs(since)
            self._send_json({"lines": lines, "next": next_idx})

        def _get_profile(self):
            name = self._qs.get("profile", [""])[0]
            if not _valid_profile_id(name):
                return self._send_json({"ok": False, "error": "invalid profile id"}, 400)
            cfg = load_config()
            conf = profiles_path(cfg) / f"{name}.conf"
            if not conf.is_file():
                return self._send_json({"ok": False, "error": "no such profile"}, 404)
            try:
                content = conf.read_text()
            except OSError as e:
                return self._send_json({"ok": False, "error": str(e)}, 500)
            self._send_json({"ok": True, "content": content})

        # -- POST handlers --

        def _post_start(self):
            name = self._qs.get("profile", [""])[0]
            cfg = load_config()
            conf = profiles_path(cfg) / f"{name}.conf"
            if not conf.is_file():
                return self._send_json({"ok": False, "error": "no such profile"}, 404)
            args, port, meta = parse_profile(conf)
            profile_type = meta.get("type", "llama")
            if profile_type not in ("llama", "sd"):
                return self._send_json(
                    {"ok": False, "error": f"unknown profile type: {profile_type!r}"}, 400)
            binary = sd_bin(cfg) if profile_type == "sd" else llama_bin(cfg)
            # Record last_used since the user explicitly chose this profile.
            store.update(lambda s: s.update({"last_used": name}))
            try:
                runner.start(name, args, port, binary, skip_health=(profile_type == "sd"))
            except FileNotFoundError:
                msg = f"binary not found: {binary}"
                runner._emit("system", f"ERROR: {msg}")
                return self._send_json({"ok": False, "error": msg}, 500)
            self._send_json({"ok": True, "port": port})

        def _post_stop(self):
            runner.stop()
            self._send_json({"ok": True})

        def _post_favorite(self):
            name = self._qs.get("profile", [""])[0]
            if not name:
                return self._send_json({"ok": False, "error": "missing profile"}, 400)
            def toggle(s):
                if name in s["favorites"]:
                    s["favorites"].remove(name)
                else:
                    s["favorites"].append(name)
            store.update(toggle)
            self._send_json({"ok": True})

        def _post_move(self):
            name = self._qs.get("profile", [""])[0]
            direction = self._qs.get("direction", [""])[0]
            if direction not in ("up", "down"):
                return self._send_json({"ok": False, "error": "bad direction"}, 400)
            delta = -1 if direction == "up" else 1
            def move(s):
                favs = s["favorites"]
                # Visible order excludes last_used (which lives in its own slot).
                visible = [f for f in favs if f != s["last_used"]]
                if name not in visible:
                    return
                i = visible.index(name)
                if not 0 <= i + delta < len(visible):
                    return
                a, b = favs.index(name), favs.index(visible[i + delta])
                favs[a], favs[b] = favs[b], favs[a]
            store.update(move)
            self._send_json({"ok": True})

        def _post_config(self):
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
            if not (1 <= updated["port"] <= 65535):
                return self._send_json({"ok": False, "error": "port must be 1-65535"}, 400)
            if (not _is_loopback(updated["host"])
                    and not os.environ.get("LLAMA_LAUNCHER_ALLOW_REMOTE")):
                return self._send_json(
                    {"ok": False, "error":
                     "non-loopback host requires LLAMA_LAUNCHER_ALLOW_REMOTE=1 "
                     "(every endpoint is unauthenticated)"}, 400)
            if not _validate_binary(updated["llama_bin"]):
                return self._send_json(
                    {"ok": False, "error":
                     f"llama_bin {updated['llama_bin']!r} is not an executable"}, 400)
            if (updated["sd_bin"] != cfg["sd_bin"]
                    and not _validate_binary(updated["sd_bin"])):
                return self._send_json(
                    {"ok": False, "error":
                     f"sd_bin {updated['sd_bin']!r} is not an executable"}, 400)
            save_config(updated)
            needs_restart = any(updated[k] != cfg[k] for k in RESTART_FIELDS)
            self._send_json({"ok": True, "needs_restart": needs_restart})

        def _post_profile(self):
            name = self._qs.get("profile", [""])[0]
            if not _valid_profile_id(name):
                return self._send_json({"ok": False, "error": "invalid profile id"}, 400)
            body = self._read_json()
            if body is None or not isinstance(body.get("content"), str):
                return self._send_json({"ok": False, "error": "missing content"}, 400)
            new_name = body.get("new_name")
            if new_name is not None and (not isinstance(new_name, str)
                                         or not _valid_profile_id(new_name)):
                return self._send_json({"ok": False, "error": "invalid new filename"}, 400)

            cfg = load_config()
            dest_name = new_name if new_name and new_name != name else name
            old_conf = profiles_path(cfg) / f"{name}.conf"
            new_conf = profiles_path(cfg) / f"{dest_name}.conf"
            is_rename = new_conf != old_conf

            if (body.get("is_new") or is_rename) and new_conf.exists():
                return self._send_json(
                    {"ok": False, "error": "a profile with that filename already exists"}, 409)

            # Hold runner.lock so the running-profile check and the file write
            # are atomic with respect to /api/start.
            with runner.lock:
                if (runner.proc and runner.proc.poll() is None
                        and runner.profile_name in (name, dest_name)):
                    return self._send_json(
                        {"ok": False, "error": "cannot edit a running profile"}, 409)
                tmp = new_conf.with_suffix(".conf.tmp")
                try:
                    tmp.write_text(body["content"])
                    os.replace(tmp, new_conf)
                    if is_rename:
                        old_conf.unlink(missing_ok=True)
                except OSError as e:
                    return self._send_json({"ok": False, "error": str(e)}, 500)

            if is_rename:
                def rename_refs(s):
                    if s.get("last_used") == name:
                        s["last_used"] = dest_name
                    if name in s["favorites"]:
                        i = s["favorites"].index(name)
                        s["favorites"][i] = dest_name
                store.update(rename_refs)
            self._send_json({"ok": True})

        def _post_profile_copy(self):
            name = self._qs.get("profile", [""])[0]
            if not _valid_profile_id(name):
                return self._send_json({"ok": False, "error": "invalid profile id"}, 400)
            cfg = load_config()
            src = profiles_path(cfg) / f"{name}.conf"
            if not src.exists():
                return self._send_json({"ok": False, "error": "no such profile"}, 404)
            # Find a unique destination name: "name-copy", "name-copy-2", …
            base = f"{name}-copy"
            new_name = base
            counter = 2
            while (profiles_path(cfg) / f"{new_name}.conf").exists():
                new_name = f"{base}-{counter}"
                counter += 1
            dst = profiles_path(cfg) / f"{new_name}.conf"
            try:
                dst.write_text(src.read_text())
            except OSError as e:
                return self._send_json({"ok": False, "error": str(e)}, 500)
            self._send_json({"ok": True, "new_profile": new_name})

        def _post_profile_delete(self):
            name = self._qs.get("profile", [""])[0]
            if not _valid_profile_id(name):
                return self._send_json({"ok": False, "error": "invalid profile id"}, 400)
            cfg = load_config()
            conf = profiles_path(cfg) / f"{name}.conf"
            # Hold runner.lock so the running-profile check and the unlink
            # are atomic with respect to /api/start.
            with runner.lock:
                if runner.proc and runner.proc.poll() is None and runner.profile_name == name:
                    return self._send_json(
                        {"ok": False, "error": "cannot delete a running profile"}, 409)
                if not conf.exists():
                    return self._send_json({"ok": False, "error": "no such profile"}, 404)
                try:
                    conf.unlink()
                except OSError as e:
                    return self._send_json({"ok": False, "error": str(e)}, 500)
            # Remove from favorites / last_used so stale ids don't linger.
            def cleanup(s):
                if name in s["favorites"]:
                    s["favorites"].remove(name)
                if s.get("last_used") == name:
                    s["last_used"] = None
            store.update(cleanup)
            self._send_json({"ok": True})

        def _post_logs_clear(self):
            runner.clear_logs()
            self._send_json({"ok": True})

        def _post_quit(self):
            # Reply first; then shut down on a worker thread.
            # Calling server.shutdown() from a handler thread deadlocks.
            self._send_json({"ok": True})
            srv = self.server
            def shutdown_seq():
                runner.stop()
                time.sleep(0.1)  # let the response flush
                srv.shutdown()
            threading.Thread(target=shutdown_seq, daemon=True).start()

    return Handler

# ---- main -------------------------------------------------------------------

class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def _setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=512 * 1024, backupCount=2)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

def main():
    _setup_logging()
    cfg = load_config()
    if (not _is_loopback(cfg["host"])
            and not os.environ.get("LLAMA_LAUNCHER_ALLOW_REMOTE")):
        sys.exit(
            f"refusing to bind to non-loopback host {cfg['host']!r}: every "
            f"endpoint is unauthenticated and would let anyone reachable on "
            f"the network execute arbitrary commands.\n"
            f"set LLAMA_LAUNCHER_ALLOW_REMOTE=1 to override."
        )
    profiles_dir = profiles_path(cfg)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        sys.exit(f"missing {INDEX_PATH}")

    url = f"http://{cfg['host']}:{cfg['port']}"
    logging.info("→ launcher GUI at %s", url)
    logging.info("  profiles: %s", profiles_dir)
    logging.info("  llama-bin: %s", llama_bin(cfg))
    logging.info("  sd-bin:    %s", sd_bin(cfg))
    logging.info("  log:      %s", LOG_PATH)

    if cfg["open_browser_on_launch"]:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    runner = Runner()
    store = StateStore(STATE_PATH)
    handler_cls = make_handler(runner, store)

    try:
        with ReusableTCPServer((cfg["host"], cfg["port"]), handler_cls) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
        runner.stop()

if __name__ == "__main__":
    main()
