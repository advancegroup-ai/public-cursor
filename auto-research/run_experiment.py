#!/usr/bin/env python3
"""
Robust wrapper for VT API that properly handles Supabase refresh token rotation.
CRITICAL: Saves new refresh tokens to disk so they survive process restarts.
Uses `requests` library for better SSL handling on long-running connections.

Usage:
    python3 run_experiment.py --script train.py --timeout 300
"""
import json
import sys
import time
import os
import re

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

API_URL = "https://vibe-track.ngrok.app"
TOKEN_CACHE = os.path.expanduser("~/.vt_token_cache.json")


def load_cache():
    try:
        with open(TOKEN_CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(access_token, refresh_token=""):
    data = {"access_token": access_token, "refresh_token": refresh_token, "ts": time.time()}
    with open(TOKEN_CACHE, "w") as f:
        json.dump(data, f)
    print(f"[auth] Tokens saved to {TOKEN_CACHE}", file=sys.stderr)
    if refresh_token:
        print(f"[auth] New refresh token: {refresh_token[:10]}...", file=sys.stderr)
    return data


def refresh_token_call(refresh_tok):
    """Exchange refresh token for new access+refresh tokens. Saves both to cache."""
    r = requests.post(
        f"{API_URL}/api/auth/refresh",
        json={"refresh_token": refresh_tok},
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        access = data.get("access_token", "")
        new_refresh = data.get("refresh_token", "")
        if access:
            save_cache(access, new_refresh)
            return access
    print(f"[auth] Refresh failed: {r.status_code} {r.text[:200]}", file=sys.stderr)
    return None


def get_valid_token():
    """Get a valid access token from cache, refresh, or environment."""
    cache = load_cache()

    if cache.get("access_token"):
        try:
            r = requests.post(
                f"{API_URL}/api/tools/python",
                json={"script": "print('auth_ok')", "timeout": 10},
                headers={"Authorization": f"Bearer {cache['access_token']}",
                         "Content-Type": "application/json"},
                timeout=30,
            )
            if r.status_code == 200:
                print("[auth] Cached access token is valid", file=sys.stderr)
                return cache["access_token"]
        except Exception:
            pass
        print("[auth] Cached access token expired", file=sys.stderr)

    for source_name, refresh_tok in [
        ("cache", cache.get("refresh_token", "")),
        ("env", os.environ.get("VT_REFRESH_TOKEN", "")),
    ]:
        if not refresh_tok:
            continue
        print(f"[auth] Trying refresh from {source_name}: {refresh_tok[:10]}...", file=sys.stderr)
        token = refresh_token_call(refresh_tok)
        if token:
            return token

    env_token = os.environ.get("VT_TOKEN", "")
    if env_token:
        print("[auth] Using VT_TOKEN from environment", file=sys.stderr)
        return env_token

    return None


def submit_script(script_path, timeout=300):
    """Submit a Python script to DGX1 and print results."""
    token = get_valid_token()
    if not token:
        print("[error] No valid token available. Set VT_REFRESH_TOKEN or VT_TOKEN.", file=sys.stderr)
        sys.exit(1)

    with open(script_path) as f:
        code = f.read()

    print(f"[submit] Sending {script_path} (timeout={timeout}s)...", file=sys.stderr)
    t0 = time.time()

    try:
        r = requests.post(
            f"{API_URL}/api/tools/python",
            json={"script": code, "timeout": timeout},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=timeout + 120,
        )
    except requests.exceptions.ConnectionError as e:
        print(f"[submit] Connection error: {e}", file=sys.stderr)
        print("[submit] Checking NAS for results (job may have completed)...", file=sys.stderr)
        metrics = read_nas_result(token)
        if metrics:
            print_metrics(metrics)
            return
        sys.exit(1)

    elapsed = time.time() - t0

    if r.status_code == 401:
        print("[submit] Token expired mid-request, trying refresh...", file=sys.stderr)
        cache = load_cache()
        rt = cache.get("refresh_token") or os.environ.get("VT_REFRESH_TOKEN", "")
        new_token = refresh_token_call(rt) if rt else None
        if new_token:
            r = requests.post(
                f"{API_URL}/api/tools/python",
                json={"script": code, "timeout": timeout},
                headers={"Authorization": f"Bearer {new_token}",
                         "Content-Type": "application/json"},
                timeout=timeout + 120,
            )
        else:
            print("[error] Cannot refresh token", file=sys.stderr)
            sys.exit(1)

    if r.status_code != 200:
        print(f"[error] HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
        sys.exit(1)

    result = r.json()
    status = result.get("status", "unknown")
    stdout = result.get("stdout", "")

    if stdout:
        print(stdout)

    if status == "completed":
        metrics = extract_metrics(stdout)
        if not metrics:
            metrics = read_nas_result(token)
            if metrics:
                print_metrics(metrics)
        print(f"[submit] Completed in {elapsed:.1f}s", file=sys.stderr)
    elif status == "timeout":
        print(f"[submit] Timed out after {elapsed:.1f}s", file=sys.stderr)
        metrics = read_nas_result(token)
        if metrics:
            print_metrics(metrics)
    else:
        print(f"[submit] Job status: {status}", file=sys.stderr)
        for log in result.get("logs", []):
            print(f"[DGX1] {log.get('message', '')}", file=sys.stderr)
        sys.exit(1)


def print_metrics(metrics):
    print("\n---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("---")


def extract_metrics(stdout):
    metrics = {}
    in_block = False
    for line in stdout.split("\n"):
        if line.strip() == "---":
            in_block = not in_block
            continue
        if in_block:
            m = re.match(r"\s*(\w+):\s+(.+)", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def read_nas_result(token):
    try:
        r = requests.post(
            f"{API_URL}/api/tools/fs",
            json={"operation": "read",
                  "path": "/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json"},
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            content = r.json().get("content", "")
            return json.loads(content)
    except Exception:
        pass
    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Submit Python script to DGX1 via VT API")
    parser.add_argument("--script", required=True, help="Path to .py file to execute")
    parser.add_argument("--timeout", type=int, default=300, help="Max execution time (seconds)")
    args = parser.parse_args()
    submit_script(args.script, args.timeout)
