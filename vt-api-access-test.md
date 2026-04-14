# VT API Access Test Results

## 1. SQL Query Test

**Query:** `SELECT 1+1 as result`

**Response:**
```json
{"status":"completed","columns":["result"],"data":[{"result":2}],"total_rows":1,"returned_rows":1,"truncated":false}
```

Result: **PASS** — The API correctly returned `2`.

## 2. Search Test

**Search term:** `liveness`

**Response:**
```json
{"results":[],"skill_matches":[],"scope_info":{"users":17,"kbs":0}}
```

Result: **PASS** — The API responded successfully, though no matching conversations were found.

## Summary

The VT API is fully operational. The SQL query endpoint correctly computed `1+1 = 2`, confirming that query execution works as expected. The search endpoint responded successfully when searching for "liveness", but returned zero results, indicating no past conversations contain that term. The scope info shows 17 users and 0 knowledge bases in the system.
