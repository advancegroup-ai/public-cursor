#!/usr/bin/env python3
"""
run_remote.py — Bridge for Cloud Agent to execute training code on DGX1.

The Cloud Agent sandbox has no GPU. This script sends Python code to DGX1 
via the VT Backend API and returns stdout.

Usage (from Cloud Agent sandbox):
    python3 run_remote.py --script train.py --timeout 300
    python3 run_remote.py --code "import torch; print(torch.cuda.device_count())"
"""
import argparse
import json
import sys
import time
import re
import os

API_URL = "https://vibe-track.ngrok.app"
API_TOKEN = os.environ.get("VT_TOKEN", "")
REFRESH_TOKEN = os.environ.get("VT_REFRESH_TOKEN", "")
_cached_token = ""


def refresh_access_token() -> str:
    """Use refresh token to get a fresh access token."""
    global _cached_token
    if not REFRESH_TOKEN:
        return ""
    import urllib.request
    try:
        body = json.dumps({"refresh_token": REFRESH_TOKEN}).encode()
        req = urllib.request.Request(
            f"{API_URL}/api/auth/refresh",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        _cached_token = data.get("access_token", "")
        return _cached_token
    except Exception:
        return ""


def get_token():
    global _cached_token
    if _cached_token:
        return _cached_token
    if API_TOKEN:
        return API_TOKEN
    # Try refresh token first (works in Cloud Agent sandbox)
    if REFRESH_TOKEN:
        t = refresh_access_token()
        if t:
            return t
    # Fallback: read from NAS (local testing only)
    try:
        import subprocess
        result = subprocess.run(
            ["node", "-e",
             "console.log(require(require('os').homedir()+'/.vibe-track/auth.json').accessToken)"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def run_on_dgx1(script_code: str, timeout: int = 300) -> dict:
    """Send script to DGX1 via VT API, return result dict."""
    import urllib.request
    import urllib.error

    token = get_token()
    if not token:
        return {"status": "error", "error": "No API token available"}

    body = json.dumps({"script": script_code, "timeout": timeout}).encode()
    req = urllib.request.Request(
        f"{API_URL}/api/tools/python",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout + 60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401 and REFRESH_TOKEN:
            # Token expired — refresh and retry once
            new_token = refresh_access_token()
            if new_token:
                req.remove_header("Authorization")
                req.add_header("Authorization", f"Bearer {new_token}")
                try:
                    with urllib.request.urlopen(req, timeout=timeout + 60) as resp:
                        return json.loads(resp.read().decode())
                except Exception as e2:
                    return {"status": "error", "error": str(e2)}
        return {"status": "error", "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


RESULT_FILE_PATH = "/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json"


def extract_metrics(stdout: str) -> dict:
    """Extract metrics from the standard output format or NAS result file."""
    metrics = {}
    # Try stdout first
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


def read_result_from_nas(token: str) -> dict:
    """Read last_result.json from NAS via FS API (fallback when stdout is empty)."""
    import urllib.request
    try:
        body = json.dumps({"operation": "read", "path": RESULT_FILE_PATH}).encode()
        req = urllib.request.Request(
            f"{API_URL}/api/tools/fs",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        content = data.get("content", "")
        return json.loads(content)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Execute Python on DGX1 via VT API")
    parser.add_argument("--script", help="Path to .py file to execute")
    parser.add_argument("--code", help="Inline Python code to execute")
    parser.add_argument("--timeout", type=int, default=300, help="Max execution time (seconds)")
    args = parser.parse_args()

    if args.script:
        with open(args.script) as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("Error: provide --script or --code", file=sys.stderr)
        sys.exit(1)

    print(f"Submitting to DGX1 (timeout={args.timeout}s)...", file=sys.stderr)
    t0 = time.time()
    result = run_on_dgx1(code, args.timeout)
    elapsed = time.time() - t0

    status = result.get("status", "unknown")
    if status == "completed":
        stdout = result.get("stdout", "")
        if stdout:
            print(stdout)
        metrics = extract_metrics(stdout)
        # Fallback: read result from NAS file if stdout is empty
        if not metrics:
            metrics = read_result_from_nas(get_token())
            if metrics:
                print("\n---")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                print("---")
        if metrics:
            print(f"\n[run_remote] Metrics: {json.dumps(metrics)}", file=sys.stderr)
        print(f"[run_remote] Completed in {elapsed:.1f}s", file=sys.stderr)
    elif status == "failed":
        logs = result.get("logs", [])
        for log in logs:
            print(f"[DGX1] {log.get('message', '')}", file=sys.stderr)
        sys.exit(1)
    elif status == "timeout":
        print(f"[run_remote] Job still running after {elapsed:.1f}s", file=sys.stderr)
        print(f"[run_remote] job_id={result.get('job_id')}", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"[run_remote] Error: {result.get('error', status)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
