#!/bin/bash
# Submit train.py to DGX1 via curl (workaround for Python SSL timeout issues)
set -e

SCRIPT_PATH="${1:-train.py}"
TIMEOUT="${2:-300}"
API_URL="https://vibe-track.ngrok.app"
REFRESH_TOKEN="${VT_REFRESH_TOKEN}"

# Get fresh access token
TOKEN=$(curl -s -X POST "$API_URL/api/auth/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"$REFRESH_TOKEN\"}" \
  --max-time 15 | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to get access token"
  exit 1
fi

echo "Got fresh token, submitting $SCRIPT_PATH (timeout=${TIMEOUT}s)..." >&2

SCRIPT_CONTENT=$(python3 -c "
import json, sys
with open('$SCRIPT_PATH') as f:
    code = f.read()
print(json.dumps({'script': code, 'timeout': $TIMEOUT}))
")

RESULT=$(curl -s -X POST "$API_URL/api/tools/python" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$SCRIPT_CONTENT" \
  --max-time $((TIMEOUT + 120)))

STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))")
STDOUT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stdout',''))")

echo "$STDOUT"

if [ "$STATUS" != "completed" ]; then
  echo "Job status: $STATUS" >&2
  echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''), d.get('logs',''))" >&2
  exit 1
fi

echo "[submit.sh] Completed successfully" >&2
