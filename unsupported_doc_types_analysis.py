#!/usr/bin/env python3
"""
Unsupported Document Types Analysis
====================================
Analyzes the proportion and volume of document types NOT supported by
in-house (自研) OCR in both:
  1. Standalone Global OCR (单点 Global OCR)
  2. Solution/IDV flow

Date range: last 30 days
Output: Volume counts, proportions, and breakdown by document type

Usage:
    Via VT API tool proxy:
        Set TOOL_PROXY env var, then run this script.
    Via direct VT API:
        Set VT_API_URL and VT_TOKEN env vars.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOOL_PROXY = os.environ.get("TOOL_PROXY", "http://localhost:3100/api/agent/tool-proxy")
VT_API_URL = os.environ.get("VT_API_URL", "https://vibe-track.ngrok.app")
VT_TOKEN = os.environ.get("VT_TOKEN", "")
CHAT_SESSION_ID = "536bdecc-e495-40d7-b8e4-309241809550"

end_date = datetime.now()
start_date = end_date - timedelta(days=30)
START_PT = start_date.strftime("%Y%m%d")
END_PT = end_date.strftime("%Y%m%d")

print(f"Analysis period: {START_PT} to {END_PT}")
print(f"=" * 60)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def query_via_proxy(sql, limit=2000):
    """Execute SQL via the tool proxy."""
    import urllib.request
    body = json.dumps({
        "endpoint": "/api/tools/query",
        "body": {"sql": sql, "limit": limit}
    }).encode()
    req = urllib.request.Request(
        TOOL_PROXY,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def query_via_api(sql, limit=2000):
    """Execute SQL via direct VT API."""
    import urllib.request
    body = json.dumps({"sql": sql, "limit": limit}).encode()
    req = urllib.request.Request(
        f"{VT_API_URL}/api/tools/query",
        data=body,
        headers={
            "Authorization": f"Bearer {VT_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def run_query(sql, limit=2000):
    """Try proxy first, then direct API."""
    for attempt in range(3):
        try:
            try:
                return query_via_proxy(sql, limit)
            except Exception:
                if VT_TOKEN:
                    return query_via_api(sql, limit)
                raise
        except Exception as e:
            print(f"  Query attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def run_python(script):
    """Execute Python via the tool proxy."""
    import urllib.request
    body = json.dumps({
        "endpoint": "/api/tools/python",
        "body": {"script": script}
    }).encode()
    req = urllib.request.Request(
        TOOL_PROXY,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# SQL Queries
# ---------------------------------------------------------------------------

# Step 0: Discover OCR-related tables
DISCOVER_OCR_TABLES = """
-- Find Global OCR related tables
-- Try common naming patterns
SELECT 'check' AS status
"""

# Step 1: Standalone Global OCR - document type distribution with unsupported flag
# The Global OCR standalone service likely stores results in a dedicated table.
# Common table candidates:
#   - adv_guardian_data_core.dw_ekyc_global_ocr_result
#   - adv_guardian_data_core.ods_*ocr*
#   - adv_guardian_data_core.dw_advance_business_ekyc_transaction (with ocr fields)

GLOBAL_OCR_TOTAL_VOLUME = f"""
SELECT
    COUNT(*) AS total_requests
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND solution_code = 'GLOBAL_OCR'
"""

GLOBAL_OCR_DOC_TYPE_DISTRIBUTION = f"""
SELECT
    GET_JSON_OBJECT(extra_info, '$.documentType') AS document_type,
    GET_JSON_OBJECT(extra_info, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(extra_info, '$.isInHouseSupported') AS is_inhouse_supported,
    COUNT(DISTINCT signature_id) AS pv_count
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND solution_code = 'GLOBAL_OCR'
GROUP BY
    GET_JSON_OBJECT(extra_info, '$.documentType'),
    GET_JSON_OBJECT(extra_info, '$.ocrVendor'),
    GET_JSON_OBJECT(extra_info, '$.isInHouseSupported')
ORDER BY pv_count DESC
LIMIT 200
"""

# Alternative: check the OCR node in sub_sub_node tables
GLOBAL_OCR_VIA_SUB_NODE_SG = f"""
SELECT
    GET_JSON_OBJECT(data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.ocrProvider') AS ocr_provider,
    GET_JSON_OBJECT(data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(data, '$.countryCode') AS country_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND node_type = 'DOCUMENT'
  AND sub_node_type IN ('GLOBAL_OCR', 'OCR', 'ID_OCR')
GROUP BY
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.ocrProvider'),
    GET_JSON_OBJECT(data, '$.cardType'),
    GET_JSON_OBJECT(data, '$.countryCode')
ORDER BY cnt DESC
LIMIT 200
"""

GLOBAL_OCR_VIA_SUB_NODE_ID = f"""
SELECT
    GET_JSON_OBJECT(data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.ocrProvider') AS ocr_provider,
    GET_JSON_OBJECT(data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(data, '$.countryCode') AS country_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND node_type = 'DOCUMENT'
  AND sub_node_type IN ('GLOBAL_OCR', 'OCR', 'ID_OCR')
GROUP BY
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.ocrProvider'),
    GET_JSON_OBJECT(data, '$.cardType'),
    GET_JSON_OBJECT(data, '$.countryCode')
ORDER BY cnt DESC
LIMIT 200
"""

# Step 2: Explore ekyc_txn table for OCR-related fields
EKYC_TXN_OCR_FIELDS = f"""
SELECT
    idv_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND (
    idv_code LIKE '%OCR%'
    OR idv_code LIKE '%DOCUMENT%'
    OR idv_code LIKE '%UNSUPPORTED%'
  )
GROUP BY idv_code
ORDER BY cnt DESC
LIMIT 100
"""

# Step 3: Funnel detail - check for document-related rejection reasons
FUNNEL_DOC_REJECT = f"""
SELECT
    solution_code,
    COUNT(DISTINCT uid) AS total_pv,
    SUM(CASE WHEN is_idv_passed = 0 THEN 1 ELSE 0 END) AS rejected_pv
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND solution_code IN ('GLOBAL_OCR', 'DOCUMENT_FACE', 'DOCUMENT_DATABASE_FACE')
GROUP BY solution_code
ORDER BY total_pv DESC
LIMIT 50
"""

# Step 4: Explore schema to find OCR-specific tables
EXPLORE_TABLES_1 = """
-- Try to find OCR-related table structure
SELECT column_name, column_type, column_comment
FROM information_schema.columns
WHERE table_schema = 'adv_guardian_data_core'
  AND table_name LIKE '%ocr%'
LIMIT 100
"""

# Step 5: Look for unsupported document types in the sub_sub_node DOCUMENT node
UNSUPPORTED_IN_SOLUTION_SG = f"""
SELECT
    GET_JSON_OBJECT(data, '$.result.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.result.ocrResult.vendor') AS ocr_vendor,
    GET_JSON_OBJECT(data, '$.result.ocrResult.isSupported') AS is_supported,
    GET_JSON_OBJECT(data, '$.result.cardType') AS card_type,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND node_type = 'DOCUMENT'
GROUP BY
    GET_JSON_OBJECT(data, '$.result.documentType'),
    GET_JSON_OBJECT(data, '$.result.ocrResult.vendor'),
    GET_JSON_OBJECT(data, '$.result.ocrResult.isSupported'),
    GET_JSON_OBJECT(data, '$.result.cardType')
ORDER BY cnt DESC
LIMIT 200
"""

# Step 6: Check Global OCR standalone table (if it exists)
GLOBAL_OCR_STANDALONE_CHECK = f"""
SELECT
    COUNT(*) AS total_calls,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM adv_guardian_data_core.dw_advance_business_global_ocr_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
"""

# Step 7: Alternative - check for a global_ocr result table
GLOBAL_OCR_RESULT_TABLE = f"""
SELECT
    document_type,
    card_type,
    country_code,
    ocr_vendor,
    is_inhouse_supported,
    COUNT(*) AS cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM adv_guardian_data_core.dw_advance_business_global_ocr_result
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
GROUP BY document_type, card_type, country_code, ocr_vendor, is_inhouse_supported
ORDER BY cnt DESC
LIMIT 200
"""

# Step 8: Comprehensive search in ekyc_txn for unsupported doc types
EKYC_TXN_UNSUPPORTED = f"""
SELECT
    solution_code,
    idv_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND (
    idv_code LIKE '%UNSUPPORT%'
    OR idv_code LIKE '%NOT_SUPPORT%'
    OR idv_code LIKE '%UNKNOWN_DOC%'
    OR idv_code LIKE '%UNRECOGNIZED%'
    OR idv_code LIKE '%FALLBACK%'
  )
GROUP BY solution_code, idv_code
ORDER BY cnt DESC
LIMIT 200
"""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if result is None:
        print("  [FAILED] Could not execute query")
        return
    if isinstance(result, dict):
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
            return
        if "data" in result:
            rows = result["data"]
            if not rows:
                print("  (no rows returned)")
                return
            if isinstance(rows, list) and len(rows) > 0:
                if isinstance(rows[0], dict):
                    headers = list(rows[0].keys())
                    print("  " + " | ".join(f"{h:>20}" for h in headers))
                    print("  " + "-" * (22 * len(headers)))
                    for row in rows[:50]:
                        print("  " + " | ".join(f"{str(row.get(h, '')):>20}" for h in headers))
                    if len(rows) > 50:
                        print(f"  ... and {len(rows) - 50} more rows")
                else:
                    for row in rows[:50]:
                        print(f"  {row}")
            return
        if "columns" in result and "rows" in result:
            cols = result["columns"]
            rows = result["rows"]
            print("  " + " | ".join(f"{c:>20}" for c in cols))
            print("  " + "-" * (22 * len(cols)))
            for row in rows[:50]:
                print("  " + " | ".join(f"{str(v):>20}" for v in row))
            return
    print(f"  Raw result: {json.dumps(result, indent=2, ensure_ascii=False)[:2000]}")


def main():
    queries = [
        ("1. Global OCR Total Volume (ekyc_txn)", GLOBAL_OCR_TOTAL_VOLUME),
        ("2. Global OCR Doc Type Distribution (ekyc_txn)", GLOBAL_OCR_DOC_TYPE_DISTRIBUTION),
        ("3. Global OCR via Sub Node (SG)", GLOBAL_OCR_VIA_SUB_NODE_SG),
        ("4. Global OCR via Sub Node (ID)", GLOBAL_OCR_VIA_SUB_NODE_ID),
        ("5. eKYC TXN OCR-related idv_codes", EKYC_TXN_OCR_FIELDS),
        ("6. Funnel Detail by Solution Code", FUNNEL_DOC_REJECT),
        ("7. Unsupported Doc Types in Solution (SG sub_sub_node)", UNSUPPORTED_IN_SOLUTION_SG),
        ("8. Global OCR Standalone Table Check", GLOBAL_OCR_STANDALONE_CHECK),
        ("9. Global OCR Result Table", GLOBAL_OCR_RESULT_TABLE),
        ("10. eKYC TXN Unsupported Doc Types", EKYC_TXN_UNSUPPORTED),
    ]

    results = {}
    for label, sql in queries:
        print(f"\nExecuting: {label}...")
        print(f"SQL: {sql.strip()[:200]}...")
        result = run_query(sql)
        results[label] = result
        print_result(label, result)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"Date range: {START_PT} to {END_PT}")
    print(f"Queries executed: {len(queries)}")
    successful = sum(1 for r in results.values() if r and "error" not in r)
    print(f"Successful queries: {successful}/{len(queries)}")

    return results


# ---------------------------------------------------------------------------
# SQL snippets for manual execution via Vibe Track UI
# ---------------------------------------------------------------------------
MANUAL_QUERIES = f"""
=== MANUAL QUERY GUIDE ===
Copy these queries into Vibe Track or ODPS console.
Date range: {START_PT} to {END_PT}

--- Query 1: Find all solution codes with volume ---
SELECT
    solution_code,
    COUNT(DISTINCT uid) AS pv,
    COUNT(DISTINCT biz_user_id) AS uv
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail f
LEFT JOIN adv_guardian_data_core.dw_ekyc_uid_mapping_v2 u
    ON f.uid = u.signature_id AND f.pt = u.pt
WHERE f.pt BETWEEN '{START_PT}' AND '{END_PT}'
GROUP BY solution_code
ORDER BY pv DESC
LIMIT 50;

--- Query 2: Check sub_sub_node for DOCUMENT nodes and their sub types ---
SELECT
    sub_node_type,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND node_type = 'DOCUMENT'
GROUP BY sub_node_type
ORDER BY cnt DESC
LIMIT 50;

--- Query 3: Sample DOCUMENT sub_sub_node data to find OCR fields ---
SELECT
    signature_id,
    sub_node_type,
    SUBSTR(data, 1, 500) AS data_sample
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt = '{END_PT}'
  AND node_type = 'DOCUMENT'
LIMIT 10;

--- Query 4: Check ekyc_txn for OCR-related error codes ---
SELECT
    solution_code,
    idv_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
GROUP BY solution_code, idv_code
ORDER BY cnt DESC
LIMIT 200;

--- Query 5: Look for Global OCR specific tables ---
-- Try these table names (run each separately):
-- SELECT COUNT(*) FROM adv_guardian_data_core.dw_advance_business_global_ocr_transaction WHERE pt = '{END_PT}';
-- SELECT COUNT(*) FROM adv_guardian_data_core.ods_sg_advance_business_global_ocr_result WHERE pt = '{END_PT}';
-- SELECT COUNT(*) FROM adv_guardian_data_core.dw_ekyc_global_ocr_result WHERE pt = '{END_PT}';

--- Query 6: Unsupported doc types in Solution (via forgery/doc sub_node) ---
SELECT
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType') AS doc_type,
    GET_JSON_OBJECT(data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(data, '$.countryCode') AS country,
    GET_JSON_OBJECT(data, '$.ocrVendor') AS vendor,
    GET_JSON_OBJECT(data, '$.isInHouseSupported') AS inhouse,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '{START_PT}' AND '{END_PT}'
  AND node_type = 'DOCUMENT'
GROUP BY
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.cardType'),
    GET_JSON_OBJECT(data, '$.countryCode'),
    GET_JSON_OBJECT(data, '$.ocrVendor'),
    GET_JSON_OBJECT(data, '$.isInHouseSupported')
ORDER BY cnt DESC
LIMIT 200;
"""


if __name__ == "__main__":
    if "--manual" in sys.argv:
        print(MANUAL_QUERIES)
    else:
        main()
