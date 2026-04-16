#!/usr/bin/env python3
"""
自研不支持证件类型分析 — 单点 Global OCR + Solution/IDV
======================================================
统计最近一个月的流量里单点 Global OCR 和 Solution 中自研不支持的证件类型，
计算比例和量。

Usage:
    # Via VT Tool Proxy (default)
    python3 analysis_report.py

    # Via direct VT API
    VT_TOKEN="<token>" python3 analysis_report.py --direct

    # Print SQL queries only (no execution)
    python3 analysis_report.py --sql-only
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta

TOOL_PROXY = os.environ.get("TOOL_PROXY", "http://localhost:3100/api/agent/tool-proxy")
VT_API = "https://vibe-track.ngrok.app"
VT_TOKEN = os.environ.get("VT_TOKEN", "")
DIRECT_MODE = "--direct" in sys.argv
SQL_ONLY = "--sql-only" in sys.argv
CHAT_SESSION_ID = "fe3a0163-cccb-4bc1-8aee-d4d5c9cb6fdd"

TODAY = datetime.now()
START_DATE = (TODAY - timedelta(days=30)).strftime("%Y%m%d")
END_DATE = (TODAY - timedelta(days=1)).strftime("%Y%m%d")


def query(sql, limit=2000, retries=3):
    """Execute SQL via VT API."""
    for attempt in range(retries):
        try:
            if DIRECT_MODE:
                url = f"{VT_API}/api/tools/query"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VT_TOKEN}",
                }
                body = {"sql": sql, "limit": limit}
            else:
                url = TOOL_PROXY
                headers = {"Content-Type": "application/json"}
                body = {"endpoint": "/api/tools/query", "body": {"sql": sql, "limit": limit}}

            data = json.dumps(body).encode()
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())
                return result
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [retry {attempt+1}] {str(e)[:80]}, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                return {"error": str(e)}


def publish(html, oss_key):
    """Publish HTML report to OSS."""
    try:
        if DIRECT_MODE:
            url = f"{VT_API}/api/tools/deliver"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VT_TOKEN}",
            }
        else:
            url = TOOL_PROXY
            headers = {"Content-Type": "application/json"}

        body = {
            "endpoint": "/api/tools/deliver",
            "body": {
                "channel": "oss_html",
                "oss_key": oss_key,
                "html": html,
                "chat_session_id": CHAT_SESSION_ID,
            },
        }
        if DIRECT_MODE:
            body = body["body"]

        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Phase 1: Schema Discovery
# ============================================================================
DISCOVER_QUERIES = [
    ("Find OCR tables", f"""
SELECT table_name
FROM INFORMATION_SCHEMA.TABLES
WHERE table_schema = 'adv_guardian_data_core'
  AND (LOWER(table_name) LIKE '%ocr%' OR LOWER(table_name) LIKE '%global_ocr%')
ORDER BY table_name
LIMIT 50"""),

    ("Sample funnel_detail columns", f"""
SELECT *
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail
WHERE pt = '{END_DATE}'
LIMIT 2"""),

    ("Sample ekyc_txn columns", f"""
SELECT *
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt = '{END_DATE}'
LIMIT 2"""),

    ("Sample DOCUMENT sub_node data", f"""
SELECT signature_id, node_type, sub_node_type,
       SUBSTR(data, 1, 500) AS data_preview
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt = '{END_DATE}'
  AND node_type = 'DOCUMENT'
LIMIT 5"""),
]


# ============================================================================
# Phase 2: Global OCR Analysis (try multiple table patterns)
# ============================================================================
GLOBAL_OCR_QUERIES = [
    ("Global OCR doc types (pattern 1: ods_sg_global_ocr)", f"""
SELECT
    document_type,
    vendor_type,
    COUNT(*) AS total_calls,
    SUM(CASE WHEN vendor_type IN ('SELF', 'INHOUSE', 'self', 'inhouse', 'ADVANCE')
        THEN 1 ELSE 0 END) AS self_calls,
    SUM(CASE WHEN vendor_type NOT IN ('SELF', 'INHOUSE', 'self', 'inhouse', 'ADVANCE')
        OR vendor_type IS NULL THEN 1 ELSE 0 END) AS unsupported_calls
FROM (
    SELECT document_type, vendor_type
    FROM adv_guardian_data_core.ods_sg_advance_business_global_ocr_result
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
    UNION ALL
    SELECT document_type, vendor_type
    FROM adv_guardian_data_core.ods_advance_business_global_ocr_result
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
) t
GROUP BY document_type, vendor_type
ORDER BY total_calls DESC
LIMIT 200"""),

    ("Global OCR doc types (pattern 2: dw_global_ocr)", f"""
SELECT
    document_type,
    ocr_vendor,
    COUNT(*) AS total_calls,
    SUM(CASE WHEN ocr_vendor IN ('SELF', 'INHOUSE', 'ADVANCE')
        THEN 1 ELSE 0 END) AS self_calls,
    SUM(CASE WHEN ocr_vendor NOT IN ('SELF', 'INHOUSE', 'ADVANCE')
        OR ocr_vendor IS NULL THEN 1 ELSE 0 END) AS unsupported_calls
FROM adv_guardian_data_core.dw_global_ocr_transaction_detail
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
GROUP BY document_type, ocr_vendor
ORDER BY total_calls DESC
LIMIT 200"""),

    ("Global OCR doc types (pattern 3: ods_sg_global_ocr_service)", f"""
SELECT
    GET_JSON_OBJECT(result, '$.documentType') AS document_type,
    vendor_type,
    COUNT(*) AS total_calls
FROM adv_guardian_data_core.ods_sg_global_ocr_service_result
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
GROUP BY GET_JSON_OBJECT(result, '$.documentType'), vendor_type
ORDER BY total_calls DESC
LIMIT 200"""),

    ("Global OCR summary (pattern 4: dw_global_ocr_transaction)", f"""
SELECT
    document_type,
    vendor,
    is_self_supported,
    COUNT(*) AS cnt
FROM adv_guardian_data_core.dw_global_ocr_transaction
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
GROUP BY document_type, vendor, is_self_supported
ORDER BY cnt DESC
LIMIT 200"""),
]


# ============================================================================
# Phase 3: Solution/IDV Analysis
# ============================================================================
SOLUTION_QUERIES = [
    ("Solution doc type distribution from funnel", f"""
SELECT
    document_type,
    COUNT(DISTINCT uid) AS pv
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
  AND solution_code != 'DOCUMENT_DATABASE_FACE'
GROUP BY document_type
ORDER BY pv DESC
LIMIT 100"""),

    ("Solution OCR vendor from sub_node DOCUMENT", f"""
SELECT
    GET_JSON_OBJECT(data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(data, '$.vendor') AS vendor,
    GET_JSON_OBJECT(data, '$.isSelfSupported') AS is_self_supported,
    COUNT(DISTINCT signature_id) AS cnt
FROM (
    SELECT signature_id, data
    FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND node_type = 'DOCUMENT'
    UNION ALL
    SELECT signature_id, data
    FROM adv_guardian_data_core.ods_advance_business_ekyc_transaction_sub_sub_node
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND node_type = 'DOCUMENT'
) t
GROUP BY
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.ocrVendor'),
    GET_JSON_OBJECT(data, '$.vendor'),
    GET_JSON_OBJECT(data, '$.isSelfSupported')
ORDER BY cnt DESC
LIMIT 200"""),

    ("Solution unsupported doc types from ekyc_txn", f"""
SELECT
    document_type,
    idv_code,
    COUNT(DISTINCT signature_id) AS pv,
    SUM(CASE WHEN ocr_vendor NOT IN ('SELF', 'INHOUSE', 'ADVANCE')
             OR ocr_vendor IS NULL THEN 1 ELSE 0 END) AS unsupported_count
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
GROUP BY document_type, idv_code
ORDER BY pv DESC
LIMIT 200"""),

    ("Solution unsupported from funnel + ocr_vendor field", f"""
SELECT
    document_type,
    ocr_vendor,
    COUNT(DISTINCT uid) AS pv,
    SUM(CASE WHEN is_idv_passed = 1 THEN 1 ELSE 0 END) AS passed_count
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
  AND solution_code != 'DOCUMENT_DATABASE_FACE'
GROUP BY document_type, ocr_vendor
ORDER BY pv DESC
LIMIT 200"""),
]

# ============================================================================
# Phase 4: Summary Queries
# ============================================================================
SUMMARY_QUERIES = [
    ("Global OCR total + unsupported (aggregated)", f"""
SELECT
    'Global OCR' AS source,
    COUNT(*) AS total_volume,
    SUM(CASE WHEN vendor_type NOT IN ('SELF', 'INHOUSE', 'ADVANCE', 'self', 'inhouse')
             AND vendor_type IS NOT NULL THEN 1 ELSE 0 END) AS unsupported_volume,
    ROUND(
        SUM(CASE WHEN vendor_type NOT IN ('SELF', 'INHOUSE', 'ADVANCE', 'self', 'inhouse')
                 AND vendor_type IS NOT NULL THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(COUNT(*), 0), 2
    ) AS unsupported_pct
FROM (
    SELECT vendor_type
    FROM adv_guardian_data_core.ods_sg_advance_business_global_ocr_result
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
    UNION ALL
    SELECT vendor_type
    FROM adv_guardian_data_core.ods_advance_business_global_ocr_result
    WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
) t"""),

    ("Solution total + unsupported (aggregated)", f"""
SELECT
    'Solution/IDV' AS source,
    COUNT(DISTINCT uid) AS total_volume,
    COUNT(DISTINCT CASE 
        WHEN ocr_vendor NOT IN ('SELF', 'INHOUSE', 'ADVANCE', 'self', 'inhouse')
             OR ocr_vendor IS NULL 
        THEN uid END) AS unsupported_volume,
    ROUND(
        COUNT(DISTINCT CASE 
            WHEN ocr_vendor NOT IN ('SELF', 'INHOUSE', 'ADVANCE', 'self', 'inhouse')
                 OR ocr_vendor IS NULL 
            THEN uid END)
        * 100.0 / NULLIF(COUNT(DISTINCT uid), 0), 2
    ) AS unsupported_pct
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail
WHERE pt BETWEEN '{START_DATE}' AND '{END_DATE}'
  AND solution_code != 'DOCUMENT_DATABASE_FACE'"""),
]


def fmt_table(data, label=""):
    """Format query result as text table."""
    if not data:
        return f"  [{label}] No data\n"
    if isinstance(data, dict) and "error" in data:
        return f"  [{label}] ERROR: {data['error'][:120]}\n"

    rows = data if isinstance(data, list) else data.get("data", data.get("rows", []))
    if not rows:
        return f"  [{label}] Empty result\n"

    if isinstance(rows[0], dict):
        headers = list(rows[0].keys())
    else:
        return f"  [{label}] {rows}\n"

    col_widths = {}
    for h in headers:
        vals = [str(r.get(h, "")) for r in rows]
        col_widths[h] = min(max(len(h), max((len(v) for v in vals), default=0)), 40)

    lines = []
    hdr = " | ".join(h.ljust(col_widths[h])[:col_widths[h]] for h in headers)
    lines.append(f"  {hdr}")
    lines.append(f"  {'-' * len(hdr)}")
    for row in rows[:50]:
        line = " | ".join(str(row.get(h, "")).ljust(col_widths[h])[:col_widths[h]] for h in headers)
        lines.append(f"  {line}")
    if len(rows) > 50:
        lines.append(f"  ... ({len(rows) - 50} more rows)")
    return "\n".join(lines) + "\n"


def build_html(all_results):
    """Build a comprehensive HTML report."""
    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f6fa; }
    h1 { color: #2d3436; border-bottom: 3px solid #e17055; padding-bottom: 10px; }
    h2 { color: #2d3436; margin-top: 40px; }
    h3 { color: #636e72; }
    .card { background: white; border-radius: 12px; padding: 20px; margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    table { border-collapse: collapse; width: 100%; }
    th { background: #2d3436; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
    td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
    tr:hover { background: #ffeaa7; }
    .tag-error { background: #ff7675; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
    .tag-ok { background: #00b894; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
    .metric-card { text-align: center; padding: 20px; }
    .metric-value { font-size: 32px; font-weight: bold; color: #e17055; }
    .metric-label { font-size: 13px; color: #636e72; margin-top: 4px; }
    .note { background: #dfe6e9; padding: 12px 16px; border-radius: 8px; font-size: 13px; color: #636e72; }
    """

    def tbl(result, title):
        rows = []
        if isinstance(result, list):
            rows = result
        elif isinstance(result, dict):
            if "error" in result:
                return f'<div class="card"><h3>{title}</h3><span class="tag-error">ERROR</span> {result["error"][:200]}</div>'
            rows = result.get("data", result.get("rows", []))
        if not rows:
            return f'<div class="card"><h3>{title}</h3><p>No data returned</p></div>'

        headers = list(rows[0].keys()) if isinstance(rows[0], dict) else []
        if not headers:
            return f'<div class="card"><h3>{title}</h3><p>{str(rows[:5])}</p></div>'

        html = f'<div class="card"><h3>{title} <span class="tag-ok">{len(rows)} rows</span></h3><table><thead><tr>'
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for row in rows[:100]:
            html += "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
        if len(rows) > 100:
            html += f'<tr><td colspan="{len(headers)}" style="text-align:center;color:#999">... {len(rows)-100} more rows</td></tr>'
        html += "</tbody></table></div>"
        return html

    sections_html = ""
    for phase_name, phase_results in all_results.items():
        sections_html += f"<h2>{phase_name}</h2>"
        for qname, result in phase_results.items():
            sections_html += tbl(result, qname)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>自研不支持证件类型分析</title>
<style>{style}</style></head>
<body>
<h1>自研不支持证件类型分析</h1>
<p style="color:#999">日期范围: {START_DATE} ~ {END_DATE} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="card">
<h2>分析目标</h2>
<p>统计最近一个月的流量中，<strong>单点 Global OCR</strong> 和 <strong>Solution/IDV</strong> 流程里
自研 OCR 不支持的证件类型的数量和占比。</p>
<div class="note">
<strong>判定逻辑:</strong> 当 OCR vendor 不是 SELF/INHOUSE/ADVANCE（自研）时，认为该证件类型自研不支持。
</div>
</div>

{sections_html}

<div class="card">
<h2>取数逻辑说明</h2>
<ul>
<li><strong>数据源:</strong> ODPS adv_guardian_data_core 项目下的 OCR / funnel / sub_node 表</li>
<li><strong>时间范围:</strong> pt BETWEEN '{START_DATE}' AND '{END_DATE}'</li>
<li><strong>Global OCR:</strong> 独立 OCR 识别调用（非 IDV/Solution 流程）</li>
<li><strong>Solution/IDV:</strong> 完整身份验证流程中的 OCR 环节（排除 DOCUMENT_DATABASE_FACE）</li>
<li><strong>自研判定:</strong> vendor_type / ocr_vendor IN ('SELF', 'INHOUSE', 'ADVANCE') 视为自研支持</li>
</ul>
</div>
</body></html>"""


def print_sql():
    """Print all SQL for manual execution."""
    print(f"-- 自研不支持证件类型分析 SQL")
    print(f"-- 日期范围: {START_DATE} ~ {END_DATE}")
    print(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    all_qs = [
        ("Phase 1: Schema Discovery", DISCOVER_QUERIES),
        ("Phase 2: Global OCR", GLOBAL_OCR_QUERIES),
        ("Phase 3: Solution/IDV", SOLUTION_QUERIES),
        ("Phase 4: Summary", SUMMARY_QUERIES),
    ]
    for phase, qs in all_qs:
        print(f"\n-- {'='*60}")
        print(f"-- {phase}")
        print(f"-- {'='*60}")
        for name, sql in qs:
            print(f"\n-- [{name}]")
            print(sql.strip() + ";")
            print()


def run_analysis():
    """Run full analysis."""
    print(f"=== 自研不支持证件类型分析 ===")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"模式: {'Direct API' if DIRECT_MODE else 'Tool Proxy'}")
    print()

    all_results = {}

    phases = [
        ("1. Schema Discovery", DISCOVER_QUERIES),
        ("2. 单点 Global OCR — 证件类型", GLOBAL_OCR_QUERIES),
        ("3. Solution/IDV — 证件类型", SOLUTION_QUERIES),
        ("4. 汇总", SUMMARY_QUERIES),
    ]

    for phase_name, queries_list in phases:
        print(f"\n{'='*60}")
        print(f"  {phase_name}")
        print(f"{'='*60}")
        phase_results = {}

        for q_name, sql in queries_list:
            print(f"\n>>> {q_name}")
            result = query(sql)

            if "error" in result:
                err = str(result["error"])
                print(f"    ERROR: {err[:120]}")
                if any(skip in err.lower() for skip in ["table not found", "does not exist", "no such table"]):
                    print(f"    (table doesn't exist, skipping)")
                    continue
            else:
                rows = result.get("data", result.get("rows", []))
                print(f"    OK: {len(rows)} rows returned")
                print(fmt_table(result, q_name))

            phase_results[q_name] = result

        all_results[phase_name] = phase_results

    # Build and save report
    html = build_html(all_results)
    report_path = "/workspace/unsupported_doc_types_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nHTML report saved: {report_path}")

    # Publish
    oss_key = f"ekyc/unsupported-doc-types/{END_DATE}/index.html"
    print(f"Publishing to: {oss_key}")
    pub = publish(html, oss_key)
    if "error" not in pub:
        url = f"https://vibe-track.ngrok.app/r/{oss_key}"
        print(f"Published: {url}")
    else:
        print(f"Publish failed: {pub.get('error', '')[:200]}")

    return all_results


if __name__ == "__main__":
    if SQL_ONLY:
        print_sql()
    else:
        run_analysis()
