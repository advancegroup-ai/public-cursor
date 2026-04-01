"""
Auto-Research Orchestrator — drives continuous experiment loop via Cursor Cloud Agent API.

Adapted from Karpathy's autoresearch: instead of one agent looping forever in an IDE,
we launch sequential Cloud Agent sessions, each picking up where the last left off.

Usage:
    python orchestrator.py test               # Test API connectivity
    python orchestrator.py launch             # Launch a single session
    python orchestrator.py status <agent_id>  # Check agent status
    python orchestrator.py loop               # Run continuous experiment loop
    python orchestrator.py loop --max-iters 5 # Run N iterations
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import re
import subprocess

import requests

REPO_DIR = Path("/tmp/public-cursor")
GIT_SSH_CMD = "ssh -i ~/.ssh/id_ed25519_github -o StrictHostKeyChecking=no"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://api.cursor.com/v0"
API_KEY = os.environ.get(
    "CURSOR_API_KEY",
    "crsr_42dd9959d2bd8c98804d11b17a471873c6a05854446f59a1763177efc6a522b3",
)
REPO = "github.com/advancegroup-ai/public-cursor"
DEFAULT_MODEL = "claude-4.6-opus-high-thinking"

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_FILE = Path(__file__).parent / "loop_state.json"

NOTIFY_EMAIL = "hiu.tuan.ting@advancegroup.com"
VT_API_URL = "https://vibe-track.ngrok.app"

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _auth_header() -> str:
    return "Basic " + base64.b64encode(f"{API_KEY}:".encode()).decode()

def _headers() -> dict:
    return {"Authorization": _auth_header(), "Content-Type": "application/json"}

def api_get(path: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  [retry] API GET {path} timeout, retrying in {wait}s ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

def api_post(path: str, body: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(f"{API_BASE}{path}", headers=_headers(), json=body, timeout=90)
            if not r.ok:
                print(f"  API error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  [retry] API POST {path} timeout, retrying in {wait}s ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

# ---------------------------------------------------------------------------
# Agent operations
# ---------------------------------------------------------------------------

def launch_agent(prompt: str, model: str = DEFAULT_MODEL, ref: str = "main",
                 branch_name: Optional[str] = None) -> dict:
    body = {
        "prompt": {"text": prompt},
        "model": model,
        "source": {"repository": REPO, "ref": ref},
    }
    if branch_name:
        body["target"] = {"branchName": branch_name, "autoCreatePr": False}
    result = api_post("/agents", body)
    print(f"  Agent launched: {result.get('id', '?')}")
    return result

def get_status(agent_id: str) -> dict:
    return api_get(f"/agents/{agent_id}")

def get_conversation(agent_id: str) -> list:
    data = api_get(f"/agents/{agent_id}/conversation")
    return data.get("messages", data.get("conversation", []))

def wait_for_completion(agent_id: str, poll_interval: int = 30, timeout: int = 1800) -> dict:
    start = time.time()
    consecutive_errors = 0
    while time.time() - start < timeout:
        try:
            status = get_status(agent_id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:4d}s] Poll error #{consecutive_errors}: {e}")
            if consecutive_errors >= 5:
                print(f"  Too many poll errors, assuming agent is still running")
                # Don't abort — just wait longer and try again
            time.sleep(poll_interval * 2)
            continue

        state = status.get("status", "UNKNOWN")
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:4d}s] {agent_id[:16]}... {state}")
        if state in ("FINISHED", "FAILED", "CANCELLED", "STOPPED"):
            return status
        time.sleep(poll_interval)
    print(f"  Timeout after {timeout}s")
    try:
        api_post(f"/agents/{agent_id}/stop", {})
    except Exception:
        pass
    try:
        return get_status(agent_id)
    except Exception:
        return {"status": "TIMEOUT"}

# ---------------------------------------------------------------------------
# Loop state management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"iteration": 0, "latest_ref": "main", "latest_branch": None,
            "total_experiments": 0, "history": []}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Git operations: consolidate results across branches
# ---------------------------------------------------------------------------

def _git(args: list, cwd=None) -> str:
    """Run a git command and return stdout."""
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = GIT_SSH_CMD
    result = subprocess.run(
        ["/usr/bin/git"] + args,
        capture_output=True, text=True, timeout=60,
        cwd=cwd or REPO_DIR, env=env,
    )
    return result.stdout.strip()


def _parse_results_tsv(content: str) -> list:
    """Parse results.tsv into list of (bal_acc, full_line) tuples."""
    entries = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("commit") or line.startswith("---"):
            continue
        parts = line.split("\t")
        try:
            bal_acc = float(parts[1])
            entries.append((bal_acc, line))
        except (IndexError, ValueError):
            if "blocked" not in line.lower() and "N/A" not in line:
                entries.append((0.0, line))
    return entries


def consolidate_results():
    """Merge all branch results.tsv entries into main and push."""
    print("  [consolidate] Fetching branches...")
    _git(["fetch", "--all", "--prune"])
    _git(["checkout", "main"])
    _git(["pull", "origin", "main"])

    main_tsv = _git(["show", "HEAD:auto-research/results.tsv"])
    main_entries = _parse_results_tsv(main_tsv)
    main_descs = {e[1].split("\t")[-1].strip() for e in main_entries}

    branches = _git(["branch", "-r"]).split("\n")
    auto_branches = [b.strip().replace("origin/", "")
                     for b in branches if "autoresearch/liveness-v2" in b]

    new_entries = []
    for branch in auto_branches:
        try:
            branch_tsv = _git(["show", f"origin/{branch}:auto-research/results.tsv"])
        except Exception:
            continue
        for bal_acc, line in _parse_results_tsv(branch_tsv):
            desc = line.split("\t")[-1].strip()
            if desc not in main_descs and bal_acc > 0:
                new_entries.append((bal_acc, line))
                main_descs.add(desc)

    if not new_entries:
        print("  [consolidate] No new results to merge")
        return

    new_entries.sort(key=lambda x: -x[0])
    print(f"  [consolidate] Merging {len(new_entries)} new results into main")

    results_path = REPO_DIR / "auto-research" / "results.tsv"
    current = results_path.read_text()
    additions = "\n".join(line for _, line in new_entries)
    results_path.write_text(current.rstrip() + "\n" + additions + "\n")

    _git(["add", "auto-research/results.tsv"])
    msg_file = Path("/tmp/_consolidate_msg.txt")
    msg_file.write_text(f"consolidate: merge {len(new_entries)} results from branches")
    _git(["commit", "-F", str(msg_file)])
    _git(["push", "origin", "main"])
    print(f"  [consolidate] Pushed {len(new_entries)} new results to main")


def merge_best_train():
    """Copy the best-performing train.py from branches to main.

    Only considers [DGX1] results (real GPU experiments on full dataset),
    ignoring early synthetic/CPU experiments that had inflated scores.
    """
    _git(["checkout", "main"])

    branches = _git(["branch", "-r"]).split("\n")
    auto_branches = [b.strip().replace("origin/", "")
                     for b in branches if "autoresearch/liveness-v2" in b]

    best_acc = 0.0
    best_branch = None
    for branch in auto_branches:
        try:
            branch_tsv = _git(["show", f"origin/{branch}:auto-research/results.tsv"])
        except Exception:
            continue
        for bal_acc, line in _parse_results_tsv(branch_tsv):
            if bal_acc > best_acc and "[DGX1]" in line:
                best_acc = bal_acc
                best_branch = branch

    if not best_branch or best_acc <= 0:
        print("  [merge-train] No branch with DGX1 results found")
        return

    try:
        best_train = _git(["show", f"origin/{best_branch}:auto-research/train.py"])
    except Exception:
        print(f"  [merge-train] Could not read train.py from {best_branch}")
        return

    train_path = REPO_DIR / "auto-research" / "train.py"
    current_train = train_path.read_text()
    if best_train == current_train:
        print(f"  [merge-train] train.py already up to date (best DGX1: {best_acc:.4f} from {best_branch})")
        return

    train_path.write_text(best_train)
    _git(["add", "auto-research/train.py"])
    msg_file = Path("/tmp/_merge_train_msg.txt")
    msg_file.write_text(f"merge best train.py from {best_branch} (DGX1 bal_acc={best_acc:.4f})")
    try:
        _git(["commit", "-F", str(msg_file)])
        _git(["push", "origin", "main"])
        print(f"  [merge-train] Merged train.py from {best_branch} (DGX1 bal_acc={best_acc:.4f})")
    except Exception:
        print(f"  [merge-train] Nothing to commit (train.py unchanged)")


def get_top_results(n: int = 10) -> list:
    """Parse results.tsv from disk and return top-N results.

    Prioritizes [DGX1] results but includes others too so the agent
    sees the full picture.
    """
    results_path = REPO_DIR / "auto-research" / "results.tsv"
    if not results_path.exists():
        _git(["checkout", "main"])
        _git(["pull", "origin", "main"])
    if not results_path.exists():
        return []
    content = results_path.read_text()
    dgx1_entries = []
    other_entries = []
    seen_descs = set()
    for bal_acc, line in _parse_results_tsv(content):
        if bal_acc > 0.5:
            parts = line.split("\t")
            desc = parts[-1].strip() if len(parts) >= 3 else line
            # Deduplicate by (rounded accuracy + simplified description)
            dedup_key = f"{bal_acc:.4f}_{desc[:50]}"
            if dedup_key in seen_descs:
                continue
            seen_descs.add(dedup_key)
            if "[DGX1]" in desc:
                dgx1_entries.append((bal_acc, desc))
            else:
                other_entries.append((bal_acc, desc))
    dgx1_entries.sort(key=lambda x: -x[0])
    other_entries.sort(key=lambda x: -x[0])
    combined = dgx1_entries[:n]
    remaining = n - len(combined)
    if remaining > 0:
        combined.extend(other_entries[:remaining])
    return combined


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

LOOP_PROMPT = """You are an autonomous liveness detection researcher. Read `auto-research/program.md` for detailed instructions.

## Iteration {iteration}

You are continuing an ongoing autonomous research project. This is session {iteration}.

{context}

## STEP 0: Preflight Check (MANDATORY)

Before anything else, run this connectivity test:
```bash
cd auto-research
export VT_TOKEN="<token from API Authentication section below>"
export VT_REFRESH_TOKEN="<refresh token from API Authentication section below>"
python3 run_remote.py --code "import torch; print(f'DGX1_OK gpus={{torch.cuda.device_count()}}')" --timeout 30
```
- If it prints `DGX1_OK` → proceed to experiments.
- If it fails → set env vars, retry ONCE. If still failing → commit your train.py ideas and STOP. Do NOT debug infrastructure.

## Your task

1. Read `auto-research/program.md` — it has FULL architecture details
2. Check `auto-research/results.tsv` for previous experiment results
3. Read `auto-research/train.py` to see the current best approach
4. Modify `auto-research/train.py` with a NEW experimental idea
5. Commit: `git add auto-research/train.py && git commit -m "experiment: description"`
6. Execute on DGX1: `cd auto-research && python3 run_remote.py --script train.py --timeout 300`
7. Parse stdout for the `---` block containing `balanced_accuracy`
8. If improved → keep commit. If worse → `git reset --hard HEAD~1`
9. Update results.tsv, commit, push
10. **IMMEDIATELY start the next experiment. Repeat steps 4-9.**

## Architecture

**You do NOT have a GPU.** You write code, then run it remotely on DGX1 (4x V100-32GB).

- `run_remote.py` sends `train.py` to DGX1 via API and returns stdout
- DGX1 has: PyTorch 2.4, CUDA 12.1, torchvision, sklearn, PIL
- Data on NAS: `/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/`
- ~3,600 human-annotated samples with 3 images each: face far shot + face near shot + ID card
- Binary: Positive (live, label=0) vs Negative (attack/deepfake, label=1)

## CRITICAL RULES

- **DO NOT STOP.** Run as many experiments as you can. Target 3-5 per session.
- **DO NOT ask for confirmation.** You are fully autonomous.
- **DO NOT modify** `prepare.py` or `run_remote.py` — those are fixed infrastructure.
- **5 min max per training run.** Set MAX_SECONDS=270 in train.py.
- **train.py must print** the `---` metrics block. See program.md for exact format.

## ANTI-WASTE RULES (follow strictly)

- **If `run_remote.py` fails on first try**: try ONE more time. If it fails again, commit your train.py idea and move to the next experiment. DO NOT spend >5 minutes debugging infrastructure.
- **If token/auth errors**: set the env vars as shown above and retry ONCE. If still failing, commit your code and move on. The NEXT iteration will get fresh tokens.
- **DO NOT re-examine data files** (labels.json, annotations_full.jsonl, batch files). The data format is documented in program.md. Trust it.
- **DO NOT read entire directories** or scan NAS — it's slow and wastes your session time.
- **Prefer modifying train.py and running immediately** over exploratory analysis. Your session is ~30 minutes — every minute debugging is a minute not running experiments.
- **Commit every experiment** (even failed ones) with descriptive messages. This creates a record for the next iteration.
"""


def send_email_report(subject: str, html: str):
    """Send an email report via VT deliver API."""
    try:
        token, _ = refresh_vt_tokens()
        if not token:
            print("  [email] No VT token, skipping email")
            return
        r = requests.post(
            f"{VT_API_URL}/api/tools/deliver",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"channel": "email", "to": [NOTIFY_EMAIL], "subject": subject, "html": html},
            timeout=30,
        )
        if r.ok:
            print(f"  [email] Report sent to {NOTIFY_EMAIL}")
        else:
            print(f"  [email] Failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"  [email] Error: {e}")


def build_iteration_report(iteration: int, entry: dict, state: dict, conversation: list) -> str:
    """Build HTML email for a completed iteration with leaderboard."""
    agent_id = entry.get("agent_id", "?")
    status = entry.get("final_status", "?")
    experiments = entry.get("experiments", 0)
    total = state.get("total_experiments", 0)

    # Extract experiment details from conversation
    exp_rows = ""
    agent_msgs = [m for m in conversation if m.get("type") != "user_message"]
    for m in agent_msgs:
        text = m.get("text", "")
        for line in text.split("\n"):
            ll = line.strip()
            if any(kw in ll.lower() for kw in ["balanced_acc", "got 0.", "achieved 0.", "result:", "bal_acc"]):
                exp_rows += f"<tr><td style='padding:4px 8px;border-bottom:1px solid #eee;font-size:13px'>{ll[:150]}</td></tr>\n"

    # Build leaderboard from consolidated results
    top = get_top_results(10)
    leaderboard_rows = ""
    for i, (acc, desc) in enumerate(top, 1):
        bg = "#d4edda" if i == 1 else ("#f8f9fa" if i % 2 == 0 else "white")
        leaderboard_rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:4px 8px;text-align:center;font-weight:bold'>{i}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;font-weight:bold'>{acc:.4f}</td>"
            f"<td style='padding:4px 8px;font-size:12px'>{desc[:80]}</td></tr>\n"
        )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 20px auto; color: #333; line-height: 1.5; }}
.hdr {{ background: #1a1a2e; color: white; padding: 15px 20px; border-radius: 8px 8px 0 0; }}
.body {{ background: #f8f9fa; padding: 15px 20px; border-radius: 0 0 8px 8px; }}
table {{ width: 100%; border-collapse: collapse; border: 1px solid #ddd; }}
th {{ background: #16213e; color: white; padding: 8px; text-align: left; font-size: 13px; }}
td {{ border-bottom: 1px solid #eee; }}
</style></head><body>
<div class="hdr">
<h2 style="margin:0;color:white">Auto-Research Iteration {iteration}</h2>
<p style="margin:5px 0 0;color:#ccc">Agent: <a href="https://cursor.com/agents/{agent_id}" style="color:#667eea">{agent_id[:20]}...</a></p>
</div>
<div class="body">
<p><strong>Status:</strong> {status} | <strong>Experiments this session:</strong> ~{experiments} | <strong>Total experiments:</strong> {total}</p>
<p><strong>Launched:</strong> {entry.get('launched_at','')} &rarr; <strong>Finished:</strong> {entry.get('finished_at','')}</p>
<p><strong>Files changed:</strong> {entry.get('files_changed',0)} | <strong>Lines added:</strong> {entry.get('lines_added',0)}</p>

<h3>Top 10 Results (All-Time Leaderboard)</h3>
<table>
<tr><th>#</th><th>Bal. Acc</th><th>Approach</th></tr>
{leaderboard_rows or '<tr><td colspan="3">No results yet</td></tr>'}
</table>

<h3>This Session</h3>
<table>
{exp_rows or '<tr><td>No detailed results captured from conversation</td></tr>'}
</table>

<p style="font-size:12px;color:#999;margin-top:20px">
{state['iteration']} iterations completed. {total} total experiments.
<br>View all agents: <a href="https://cursor.com/agents">cursor.com/agents</a>
</p>
</div></body></html>"""


def _read_auth_json() -> dict:
    """Read the local auth.json file."""
    import subprocess
    try:
        result = subprocess.run(
            ["node", "-e", "console.log(JSON.stringify(require(require('os').homedir()+'/.vibe-track/auth.json')))"],
            capture_output=True, text=True, timeout=5,
        )
        return json.loads(result.stdout.strip())
    except Exception:
        return {}


def _write_auth_json(data: dict):
    """Update auth.json with new tokens."""
    import subprocess
    try:
        result = subprocess.run(
            ["node", "-e", "console.log(require('os').homedir())"],
            capture_output=True, text=True, timeout=5,
        )
        home = result.stdout.strip()
        auth_path = Path(home) / ".vibe-track" / "auth.json"
        existing = json.loads(auth_path.read_text()) if auth_path.exists() else {}
        existing.update(data)
        auth_path.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        print(f"  [auth] Failed to write auth.json: {e}")


def refresh_vt_tokens() -> tuple:
    """Refresh VT tokens via API. Returns (access_token, refresh_token).

    Supabase refresh tokens are single-use: after one refresh, the old token
    is dead. We must save the new refresh token for the next iteration.
    """
    auth = _read_auth_json()
    rt = auth.get("refreshToken", "")
    if not rt:
        return auth.get("accessToken", ""), ""

    try:
        r = requests.post(
            f"{VT_API_URL}/api/auth/refresh",
            json={"refresh_token": rt},
            timeout=15, verify=False,
        )
        if r.ok:
            data = r.json()
            new_access = data.get("access_token", "")
            new_refresh = data.get("refresh_token", "")
            if new_access and new_refresh:
                _write_auth_json({"accessToken": new_access, "refreshToken": new_refresh})
                print(f"  [auth] Token refreshed OK (new refresh token saved)")
                return new_access, new_refresh
            elif new_access:
                _write_auth_json({"accessToken": new_access})
                return new_access, rt
        else:
            print(f"  [auth] Refresh failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"  [auth] Refresh error: {e}")

    return auth.get("accessToken", ""), rt


def get_vt_token() -> str:
    """Read VT access token from local auth file."""
    return _read_auth_json().get("accessToken", "")


def get_vt_refresh_token() -> str:
    """Read VT refresh token from local auth file."""
    return _read_auth_json().get("refreshToken", "")


def build_prompt(iteration: int, latest_branch: Optional[str]) -> str:
    context_parts = []

    # Inject top results leaderboard directly into prompt
    top = get_top_results(15)
    if top:
        best_acc = top[0][0]
        table_rows = "\n".join(
            f"| {i} | {acc:.4f} | {desc[:80]} |"
            for i, (acc, desc) in enumerate(top, 1)
        )
        context_parts.append(
            f"## Leaderboard — Top Results So Far\n\n"
            f"| Rank | Bal.Acc | Approach |\n"
            f"|------|---------|----------|\n"
            f"{table_rows}\n\n"
            f"**Your goal: beat {best_acc:.4f}.** Try something DIFFERENT from the approaches above."
        )
    else:
        context_parts.append("No previous results. Start from the baseline in train.py.")

    # Refresh tokens right before launch — gives the agent a fresh 1-hour access token
    vt_token, refresh_token = refresh_vt_tokens()
    token_block = "## API Authentication\n\nBefore running experiments, set these environment variables:\n```bash\n"
    if refresh_token:
        token_block += f'export VT_REFRESH_TOKEN="{refresh_token}"\n'
    if vt_token:
        token_block += f'export VT_TOKEN="{vt_token}"\n'
    token_block += (
        "```\n"
        "**IMPORTANT:** VT_TOKEN is valid for ~1 hour. VT_REFRESH_TOKEN is single-use backup.\n"
        "`run_remote.py` will auto-refresh if VT_TOKEN expires. Always set BOTH variables."
    )
    context_parts.append(token_block)

    context = "\n\n".join(context_parts)
    return LOOP_PROMPT.format(iteration=iteration, context=context)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(max_iters: int = 999, model: str = DEFAULT_MODEL, cooldown: int = 15):
    state = load_state()
    print("=" * 70)
    print(f"  AUTO-RESEARCH CONTINUOUS LOOP")
    print(f"  Starting from iteration {state['iteration'] + 1}")
    print(f"  Model: {model}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Cooldown between sessions: {cooldown}s")
    print("=" * 70)

    for _ in range(max_iters):
        state["iteration"] += 1
        iteration = state["iteration"]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'=' * 70}")
        print(f"  ITERATION {iteration} — {ts}")
        print(f"{'=' * 70}")

        # Consolidate results from all branches into main before launching
        try:
            consolidate_results()
            merge_best_train()
        except Exception as e:
            print(f"  [consolidate] Error (non-fatal): {e}")

        # Always fork from main; results are now consolidated there
        ref = "main"
        branch = f"autoresearch/liveness-v2-iter{iteration}"

        prompt = build_prompt(iteration, state.get("latest_branch"))

        try:
            # Retry agent launch up to 3 times
            result = None
            for launch_attempt in range(3):
                try:
                    result = launch_agent(prompt=prompt, model=model, ref=ref, branch_name=branch)
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    print(f"  [retry] Launch attempt {launch_attempt+1} failed: {e}")
                    time.sleep(30 * (launch_attempt + 1))

            if not result:
                print(f"  ERROR: All launch attempts failed")
                state["history"].append({
                    "iteration": iteration, "error": "All launch attempts timed out",
                    "timestamp": datetime.now().isoformat(),
                })
                save_state(state)
                time.sleep(60)
                continue

            agent_id = result.get("id")
            if not agent_id:
                print(f"  ERROR: No agent ID: {result}")
                time.sleep(60)
                continue

            entry = {
                "iteration": iteration,
                "agent_id": agent_id,
                "branch": branch,
                "ref": ref,
                "launched_at": ts,
            }

            print(f"  View: https://cursor.com/agents/{agent_id}")

            # Wait for completion (30min timeout)
            final = wait_for_completion(agent_id, poll_interval=30, timeout=1800)
            entry["final_status"] = final.get("status")
            entry["lines_added"] = final.get("linesAdded", 0)
            entry["files_changed"] = final.get("filesChanged", 0)
            entry["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get conversation summary
            try:
                msgs = get_conversation(agent_id)
                entry["num_messages"] = len(msgs)
                assistant_msgs = [m for m in msgs if m.get("type") != "user_message"]
                experiment_count = sum(
                    1 for m in assistant_msgs
                    if any(kw in m.get("text", "").lower()
                           for kw in ["experiment", "balanced_accuracy", "1.0", "0."])
                )
                entry["experiments"] = experiment_count

                # Extract summary from last few messages
                summaries = []
                for m in assistant_msgs[-3:]:
                    text = m.get("text", "")
                    if len(text) > 50:
                        summaries.append(text[:200])
                entry["summary"] = " | ".join(summaries)
            except Exception as e:
                entry["summary"] = f"Error getting conversation: {e}"

            # Save log
            log_file = LOG_DIR / f"iter{iteration:03d}_{agent_id[:12]}.json"
            with open(log_file, "w") as f:
                json.dump({"meta": entry, "conversation": msgs if 'msgs' in dir() else []},
                          f, indent=2, default=str)

            # Update state
            if final.get("status") == "FINISHED":
                state["latest_branch"] = branch
                state["latest_ref"] = branch
            state["total_experiments"] = state.get("total_experiments", 0) + entry.get("experiments", 0)
            state["history"].append(entry)
            save_state(state)

            print(f"\n  Session {iteration} done: {final.get('status')}")
            print(f"  +{entry.get('lines_added', 0)} lines, "
                  f"{entry.get('files_changed', 0)} files, "
                  f"~{entry.get('experiments', '?')} experiments")
            print(f"  Total experiments so far: {state['total_experiments']}")

            # Send email report
            try:
                report_html = build_iteration_report(iteration, entry, state, msgs if 'msgs' in dir() else [])
                send_email_report(
                    f"[Auto-Research] Iter {iteration} done — {final.get('status')} — ~{entry.get('experiments','?')} experiments",
                    report_html,
                )
            except Exception as e:
                print(f"  [email] Report error: {e}")

        except KeyboardInterrupt:
            print("\n\n  Interrupted by user. Saving state...")
            save_state(state)
            return
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            state["history"].append({
                "iteration": iteration, "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            save_state(state)
            time.sleep(60)
            continue

        # Cooldown between sessions
        print(f"  Cooling down {cooldown}s before next session...")
        time.sleep(cooldown)

    print(f"\n{'=' * 70}")
    print(f"  LOOP COMPLETE — {state['iteration']} iterations, "
          f"{state['total_experiments']} total experiments")
    print(f"{'=' * 70}")
    save_state(state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-Research Orchestrator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("test", help="Test API connectivity")

    lp = sub.add_parser("launch", help="Launch single session")
    lp.add_argument("--model", default=DEFAULT_MODEL)
    lp.add_argument("--ref", default="main")
    lp.add_argument("--branch", type=str)

    sp = sub.add_parser("status", help="Check agent status")
    sp.add_argument("agent_id")

    cp = sub.add_parser("conversation", help="Get conversation")
    cp.add_argument("agent_id")

    loop_p = sub.add_parser("loop", help="Run continuous loop")
    loop_p.add_argument("--max-iters", type=int, default=999)
    loop_p.add_argument("--model", default=DEFAULT_MODEL)
    loop_p.add_argument("--cooldown", type=int, default=15,
                        help="Seconds between sessions")

    sub.add_parser("state", help="Show loop state")
    sub.add_parser("reset", help="Reset loop state")

    args = parser.parse_args()

    if args.command == "test":
        print("Testing API...")
        info = api_get("/me")
        print(f"  Key: {info.get('apiKeyName')}, Email: {info.get('userEmail')}")
        models = api_get("/models")
        print(f"  Models: {', '.join(models.get('models', []))}")
        print("OK!")

    elif args.command == "launch":
        state = load_state()
        iteration = state["iteration"] + 1
        prompt = build_prompt(iteration, state.get("latest_branch"))
        branch = args.branch or f"autoresearch/liveness-iter{iteration}"
        result = launch_agent(prompt, args.model, args.ref, branch)
        print(json.dumps(result, indent=2))

    elif args.command == "status":
        print(json.dumps(get_status(args.agent_id), indent=2))

    elif args.command == "conversation":
        msgs = get_conversation(args.agent_id)
        for m in msgs:
            role = "USER" if m.get("type") == "user_message" else "AGENT"
            text = m.get("text", "")
            print(f"[{role}] {text[:300]}")
            print()

    elif args.command == "loop":
        run_loop(args.max_iters, args.model, args.cooldown)

    elif args.command == "state":
        state = load_state()
        print(json.dumps(state, indent=2, default=str))

    elif args.command == "reset":
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("State reset.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
