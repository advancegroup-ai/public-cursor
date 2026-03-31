#!/usr/bin/env python3
"""
Wrapper around the VT API that properly handles Supabase refresh token rotation.
Saves new refresh tokens so they aren't lost between calls.
"""
import json
import sys
import time
import os
import re

import requests

API_URL = "https://vibe-track.ngrok.app"
TOKEN_CACHE = "/tmp/vt_token_cache.json"


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
    return data


def refresh_token_call(refresh_tok):
    r = requests.post(
        f"{API_URL}/api/auth/refresh",
        json={"refresh_token": refresh_tok},
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        access = data.get("access_token", "")
        new_refresh = data.get("refresh_token", refresh_tok)
        if access:
            save_cache(access, new_refresh)
            print(f"[auth] Got fresh token, new refresh saved", file=sys.stderr)
            return access
    print(f"[auth] Refresh failed: {r.status_code} {r.text[:200]}", file=sys.stderr)
    return None


def get_valid_token():
    cache = load_cache()
    
    if cache.get("access_token"):
        r = requests.post(
            f"{API_URL}/api/tools/python",
            json={"script": "print('auth_ok')", "timeout": 10},
            headers={"Authorization": f"Bearer {cache['access_token']}", "Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code == 200:
            print("[auth] Cached access token is valid", file=sys.stderr)
            return cache["access_token"]
        print(f"[auth] Cached access token expired", file=sys.stderr)

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
        return env_token
    
    return None


def submit_script(script_path, timeout=300):
    token = get_valid_token()
    if not token:
        print("[error] No valid token available", file=sys.stderr)
        sys.exit(1)

    with open(script_path) as f:
        code = f.read()

    print(f"[submit] Sending {script_path} (timeout={timeout}s)...", file=sys.stderr)
    t0 = time.time()

    session = requests.Session()
    r = session.post(
        f"{API_URL}/api/tools/python",
        json={"script": code, "timeout": timeout},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=timeout + 120,
    )
    elapsed = time.time() - t0

    if r.status_code == 401:
        print("[submit] Token expired mid-request, trying refresh...", file=sys.stderr)
        cache = load_cache()
        rt = cache.get("refresh_token") or os.environ.get("VT_REFRESH_TOKEN", "")
        new_token = refresh_token_call(rt) if rt else None
        if new_token:
            r = session.post(
                f"{API_URL}/api/tools/python",
                json={"script": code, "timeout": timeout},
                headers={"Authorization": f"Bearer {new_token}", "Content-Type": "application/json"},
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
                print("\n---")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                print("---")
        print(f"[submit] Completed in {elapsed:.1f}s", file=sys.stderr)
    elif status == "timeout":
        print(f"[submit] Timed out after {elapsed:.1f}s", file=sys.stderr)
        metrics = read_nas_result(token)
        if metrics:
            print("\n---")
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("---")
    else:
        print(f"[submit] Job status: {status}", file=sys.stderr)
        for log in result.get("logs", []):
            print(f"[DGX1] {log.get('message', '')}", file=sys.stderr)
        sys.exit(1)


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
            json={"operation": "read", "path": "/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json"},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    submit_script(args.script, args.timeout)
