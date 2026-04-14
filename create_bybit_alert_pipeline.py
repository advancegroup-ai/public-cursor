#!/usr/bin/env python3
"""Create Bybit liveness daily alert pipeline schedule via VT API."""
import json
import requests

VT_TOKEN = "vt_sk_21089b3935576ce6c994c3aa2f14ad0d160b8eeafc583bd0b7583e520ecba2af"
VT_BASE = "https://vibe-track.ngrok.app"

# Node 1: SQL query — Bybit liveness pass rate by country (yesterday's data)
sql_node = {
    "id": "sql_bybit_liveness_country",
    "type": "sql_query",
    "config": {
        "sql": (
            "SELECT\n"
            "  b.real_region,\n"
            "  COUNT(DISTINCT a.uid) AS total_pv,\n"
            "  SUM(CASE WHEN a.is_idv_passed = 1 THEN 1 ELSE 0 END) AS passed,\n"
            "  ROUND(SUM(CASE WHEN a.is_idv_passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(DISTINCT a.uid), 2) AS pass_rate\n"
            "FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail a\n"
            "JOIN adv_guardian_data_core.dw_ekyc_uid_mapping_v2 b ON a.uid = b.signature_id\n"
            "WHERE a.pt = TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), -1, 'dd'), 'yyyyMMdd')\n"
            "  AND a.customer_id = 9929123352\n"
            "  AND a.is_face_image_uploaded = 1\n"
            "GROUP BY b.real_region\n"
            "ORDER BY total_pv DESC\n"
            "LIMIT 50"
        ),
        "description": "Bybit liveness pass rate by country for previous day (customer_id=9929123352)"
    }
}

# Node 2: Python — check threshold, build alert message
python_script = r"""import json
import sys

THRESHOLD = 80.0

raw = sys.stdin.read() or "{}"
try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    payload = {}

rows = payload.get("rows") or payload.get("data") or payload.get("result") or []
if isinstance(rows, dict) and "rows" in rows:
    rows = rows["rows"]

flagged = []
for r in rows:
    try:
        region = str(r.get("real_region") or r.get("country") or "").strip()
        rate = float(r.get("pass_rate") or 0)
        n = int(r.get("total_pv") or 0)
        if region and rate < THRESHOLD:
            flagged.append({"region": region, "pass_rate": rate, "total_pv": n})
    except (TypeError, ValueError):
        continue

if flagged:
    body = "\U0001f6a8 Bybit Liveness Daily Alert\n\n{} country(ies) below {:.0f}% pass rate:\n".format(
        len(flagged), THRESHOLD
    ) + "\n".join(
        "\u2022 {} \u2014 {:.2f}% (n={})".format(f["region"], f["pass_rate"], f["total_pv"])
        for f in sorted(flagged, key=lambda x: x["pass_rate"])
    )
else:
    body = "\u2705 Bybit Liveness Daily Check: All countries above {:.0f}% pass rate.".format(THRESHOLD)

out = {"flagged_count": len(flagged), "threshold": THRESHOLD, "lark_text": body}
print(json.dumps(out, ensure_ascii=False))
"""

python_node = {
    "id": "python_check_threshold",
    "type": "python",
    "config": {
        "script": python_script,
        "description": "Flag countries with pass_rate < 80% and prepare Lark alert text",
        "stdin_from": "sql_bybit_liveness_country"
    }
}

# Node 3: Deliver — send to Lark IM
deliver_node = {
    "id": "lark_alert",
    "type": "deliver",
    "config": {
        "channel": "lark_im",
        "text_from": "python_check_threshold.lark_text",
        "text": "Bybit Liveness Daily Alert (fallback)"
    }
}

workflow = {
    "name": "bybit-liveness-daily-alert-cursor-test",
    "nodes": [sql_node, python_node, deliver_node]
}

schedule_body = {
    "name": "bybit-liveness-daily-alert-cursor-test",
    "cron_expression": "0 9 * * *",
    "workflow_template": json.dumps(workflow, ensure_ascii=False)
}

pipeline_payload = {
    "method": "POST",
    "path": "/api/schedules",
    "body": schedule_body
}

print("=== Pipeline Payload ===")
print(json.dumps(pipeline_payload, indent=2, ensure_ascii=False))
print("\n=== Sending to VT Pipeline API ===")

resp = requests.post(
    f"{VT_BASE}/api/tools/pipeline",
    headers={
        "Authorization": f"Bearer {VT_TOKEN}",
        "Content-Type": "application/json"
    },
    json=pipeline_payload,
    timeout=60
)

print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")

try:
    data = resp.json()
    if isinstance(data, dict) and "id" in data:
        print(f"\n✅ Schedule created successfully!")
        print(f"   Schedule ID: {data['id']}")
        print(f"   Name: {data.get('name', 'N/A')}")
        print(f"   Cron: {data.get('cron_expression', 'N/A')}")
        print(f"   Active: {data.get('is_active', 'N/A')}")
        print(f"   Next run: {data.get('next_run_at', 'N/A')}")
except Exception as e:
    print(f"Error parsing response: {e}")
