#!/bin/bash
# =============================================================================
# Unsupported Document Types Analysis - Shell Script Version
# Run via: bash run_analysis.sh
# Requires: TOOL_PROXY or (VT_API_URL + VT_TOKEN) environment variables
# =============================================================================

set -euo pipefail

TOOL_PROXY="${TOOL_PROXY:-http://localhost:3100/api/agent/tool-proxy}"
VT_API="${VT_API_URL:-https://vibe-track.ngrok.app}"
TOKEN="${VT_TOKEN:-}"

# Date range: last 30 days
END_PT=$(date +%Y%m%d)
START_PT=$(date -d "30 days ago" +%Y%m%d 2>/dev/null || date -v-30d +%Y%m%d 2>/dev/null || echo "20260316")

echo "==========================================="
echo "  Unsupported Document Types Analysis"
echo "  Period: $START_PT - $END_PT"
echo "==========================================="

# Helper: run query via proxy
query_proxy() {
    local sql="$1"
    local limit="${2:-2000}"
    curl -s "$TOOL_PROXY" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg sql "$sql" --argjson limit "$limit" \
            '{"endpoint": "/api/tools/query", "body": {"sql": $sql, "limit": $limit}}')"
}

# Helper: run query via direct API
query_api() {
    local sql="$1"
    local limit="${2:-2000}"
    curl -s "$VT_API/api/tools/query" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg sql "$sql" --argjson limit "$limit" \
            '{"sql": $sql, "limit": $limit}')"
}

# Use proxy if available, fallback to direct API
query() {
    local result
    result=$(query_proxy "$1" "${2:-2000}" 2>/dev/null) || true
    if [ -z "$result" ] || echo "$result" | grep -q '"error"'; then
        if [ -n "$TOKEN" ]; then
            result=$(query_api "$1" "${2:-2000}")
        fi
    fi
    echo "$result"
}

echo ""
echo ">>> Step 1: Check solution_code distribution for Global OCR"
query "SELECT solution_code, COUNT(DISTINCT signature_id) AS pv FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction WHERE pt BETWEEN '${START_PT}' AND '${END_PT}' AND solution_code LIKE '%OCR%' GROUP BY solution_code ORDER BY pv DESC LIMIT 50" | jq .

echo ""
echo ">>> Step 2: Get DOCUMENT sub_node_type distribution (SG)"
query "SELECT sub_node_type, COUNT(DISTINCT signature_id) AS cnt FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node WHERE pt BETWEEN '${START_PT}' AND '${END_PT}' AND node_type = 'DOCUMENT' GROUP BY sub_node_type ORDER BY cnt DESC LIMIT 50" | jq .

echo ""
echo ">>> Step 3: Sample DOCUMENT node data to discover OCR fields"
query "SELECT signature_id, sub_node_type, SUBSTR(data, 1, 500) AS data_sample FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node WHERE pt = '${END_PT}' AND node_type = 'DOCUMENT' LIMIT 5" | jq .

echo ""
echo ">>> Step 4: Check for unsupported doc type codes in ekyc_txn"
query "SELECT solution_code, idv_code, COUNT(DISTINCT signature_id) AS cnt FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction WHERE pt BETWEEN '${START_PT}' AND '${END_PT}' AND (idv_code LIKE '%UNSUPPORT%' OR idv_code LIKE '%OCR%' OR idv_code LIKE '%UNKNOWN%' OR idv_code LIKE '%FALLBACK%') GROUP BY solution_code, idv_code ORDER BY cnt DESC LIMIT 100" | jq .

echo ""
echo ">>> Step 5: Document type + vendor breakdown in sub_sub_node (SG)"
query "SELECT GET_JSON_OBJECT(data, '\$.documentType') AS doc_type, GET_JSON_OBJECT(data, '\$.ocrVendor') AS vendor, GET_JSON_OBJECT(data, '\$.cardType') AS card_type, GET_JSON_OBJECT(data, '\$.countryCode') AS country, COUNT(DISTINCT signature_id) AS cnt FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node WHERE pt BETWEEN '${START_PT}' AND '${END_PT}' AND node_type = 'DOCUMENT' AND sub_node_type IN ('GLOBAL_OCR', 'OCR', 'ID_OCR') GROUP BY GET_JSON_OBJECT(data, '\$.documentType'), GET_JSON_OBJECT(data, '\$.ocrVendor'), GET_JSON_OBJECT(data, '\$.cardType'), GET_JSON_OBJECT(data, '\$.countryCode') ORDER BY cnt DESC LIMIT 200" | jq .

echo ""
echo ">>> Step 6: Try standalone Global OCR table"
query "SELECT COUNT(*) AS total FROM adv_guardian_data_core.dw_advance_business_global_ocr_transaction WHERE pt BETWEEN '${START_PT}' AND '${END_PT}'" 2>/dev/null | jq . || echo "Table may not exist"

echo ""
echo "==========================================="
echo "  Analysis Complete"
echo "==========================================="
