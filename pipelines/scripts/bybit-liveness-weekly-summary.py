"""
Bybit Liveness Weekly Summary - Python processing node
Pipeline ID: bf776eea
Schedule: Every Monday at 9:00 AM UTC

Reads SQL query results from stdin, computes global stats,
flags regions below 90% pass rate, and outputs a formatted
Lark IM message.
"""

import json
import sys

THRESHOLD = 90.0

raw = sys.stdin.read() or "{}"
try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    payload = {}

rows = payload.get("rows") or payload.get("data") or payload.get("result") or []
if isinstance(rows, dict) and "rows" in rows:
    rows = rows["rows"]

total_attempts = 0
total_passed = 0
flagged = []
top_regions = []

for r in rows:
    try:
        region = str(r.get("real_region") or "").strip()
        rate = float(r.get("pass_rate_pct") or 0)
        n = int(r.get("liveness_attempts") or 0)
        passed = int(r.get("liveness_passed") or 0)
        total_attempts += n
        total_passed += passed
        if region and n >= 10:
            top_regions.append(
                {"region": region, "pass_rate_pct": rate, "attempts": n, "passed": passed}
            )
            if rate < THRESHOLD:
                flagged.append({"region": region, "pass_rate_pct": rate, "attempts": n})
    except (TypeError, ValueError):
        continue

global_rate = round(100.0 * total_passed / total_attempts, 2) if total_attempts > 0 else 0

lines = []
lines.append("\U0001f4ca Bybit Liveness Weekly Summary")
lines.append("")
lines.append("\U0001f4c5 Period: Last 7 days")
lines.append("\U0001f30d Total attempts: {:,}".format(total_attempts))
lines.append("\u2705 Total passed: {:,}".format(total_passed))
lines.append("\U0001f4c8 Global pass rate: {:.2f}%".format(global_rate))
lines.append("")

if flagged:
    lines.append(
        "\U0001f6a8 {} region(s) below {:.0f}% pass rate (n>=10):".format(
            len(flagged), THRESHOLD
        )
    )
    for f in sorted(flagged, key=lambda x: x["pass_rate_pct"]):
        lines.append(
            "  \u2022 {} \u2014 {:.2f}% ({:,} attempts)".format(
                f["region"], f["pass_rate_pct"], f["attempts"]
            )
        )
else:
    lines.append(
        "\u2705 All regions (n>=10) above {:.0f}% pass rate.".format(THRESHOLD)
    )

lines.append("")
lines.append("Top 10 regions by volume:")
for r in sorted(top_regions, key=lambda x: -x["attempts"])[:10]:
    emoji = "\u2705" if r["pass_rate_pct"] >= THRESHOLD else "\u26a0\ufe0f"
    lines.append(
        "  {} {} \u2014 {:.2f}% ({:,} attempts)".format(
            emoji, r["region"], r["pass_rate_pct"], r["attempts"]
        )
    )

body = "\n".join(lines)

out = {
    "total_attempts": total_attempts,
    "total_passed": total_passed,
    "global_pass_rate": global_rate,
    "flagged_count": len(flagged),
    "threshold": THRESHOLD,
    "lark_text": body,
}
print(json.dumps(out, ensure_ascii=False))
