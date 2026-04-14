-- Bybit Liveness Pass Rate Report - W15 2026
-- Data source: idv_aai_liveness_details (ODPS)
-- Generated: 2026-04-14

-- Query 1: W15 Pass Rate by Country (Top 10 by volume)
SELECT region,
       COUNT(*) as total_requests,
       SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) as liveness_pass,
       SUM(CASE WHEN liveness_result1 = 'fail' THEN 1 ELSE 0 END) as liveness_fail,
       ROUND(SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pass_rate
FROM idv_aai_liveness_details
WHERE pt >= '20260407' AND pt <= '20260413'
GROUP BY region
ORDER BY total_requests DESC
LIMIT 10;

-- Query 2: Daily Pass Rate Trend for Top 5 Countries
SELECT pt, region, COUNT(*) as total_requests,
       SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) as liveness_pass,
       ROUND(SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pass_rate
FROM idv_aai_liveness_details
WHERE pt >= '20260407' AND pt <= '20260413'
  AND region IN ('PHL', 'MYS', 'THA', 'NGA', 'RUS')
GROUP BY pt, region
ORDER BY pt, region;

-- Query 3: W14 Comparison (Mar 31 - Apr 6)
SELECT region, COUNT(*) as total_requests,
       SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) as liveness_pass,
       ROUND(SUM(CASE WHEN liveness_result1 = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pass_rate
FROM idv_aai_liveness_details
WHERE pt >= '20260331' AND pt <= '20260406'
  AND region IN ('PHL', 'MYS', 'THA', 'NGA', 'RUS')
GROUP BY region
ORDER BY total_requests DESC;

-- Query 4: Liveness Result Distribution (pass/fail/null)
SELECT region, liveness_result1, COUNT(*) as cnt
FROM idv_aai_liveness_details
WHERE pt >= '20260407' AND pt <= '20260413'
  AND region IN ('PHL', 'MYS', 'THA', 'NGA', 'RUS')
GROUP BY region, liveness_result1
ORDER BY region, cnt DESC;
