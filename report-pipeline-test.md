# Report Pipeline Test

**Date:** 2026-04-14  
**Status:** Passed

## Pipeline Steps Verified

1. **SQL Query** — Executed `SELECT` with 3 dummy rows via `/api/tools/query` → returned 3 rows with columns `id`, `name`, `score`
2. **HTML Report** — Built self-contained dark-themed HTML with summary stats, data table, and collapsible SQL section
3. **OSS Publish** — Delivered via `/api/tools/deliver` with `oss_html` channel

## Published Report URL

https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/test/pipeline-test/report-2026-04-14/index.html

## Sample Data

| ID | Name    | Score |
|----|---------|-------|
| 1  | Alice   | 95    |
| 2  | Bob     | 87    |
| 3  | Charlie | 92    |
