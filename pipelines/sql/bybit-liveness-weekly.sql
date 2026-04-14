-- Bybit Liveness Weekly Summary SQL
-- Pipeline ID: bf776eea
-- Schedule: Every Monday at 9:00 AM UTC (cron: 0 9 * * 1)
-- Purpose: Aggregate Bybit LIVENESS_SELF pass rate by real_region for the past 7 days
-- Variable: {{schedule_date}} is injected by the scheduler (yyyy-MM-dd format)

WITH base AS (
  SELECT DISTINCT
    a.uid AS signature_id,
    b.real_region,
    a.pt
  FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail a
  JOIN adv_guardian_data_core.dw_ekyc_uid_mapping_v2 b ON a.uid = b.signature_id
  WHERE a.pt >= TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), -7, 'dd'), 'yyyyMMdd')
    AND a.pt < TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), 0, 'dd'), 'yyyyMMdd')
    AND a.customer_id = 9929123352
    AND a.is_face_image_uploaded = 1
),
liv_raw AS (
  SELECT signature_id, model_result, pt
  FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_model_record
  WHERE vendor_type = 'LIVENESS_SELF'
    AND pt >= TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), -7, 'dd'), 'yyyyMMdd')
    AND pt < TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), 0, 'dd'), 'yyyyMMdd')
  UNION ALL
  SELECT signature_id, model_result, pt
  FROM adv_guardian_data_core.ods_advance_business_ekyc_transaction_model_record
  WHERE vendor_type = 'LIVENESS_SELF'
    AND pt >= TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), -7, 'dd'), 'yyyyMMdd')
    AND pt < TO_CHAR(DATEADD(TO_DATE('{{schedule_date}}', 'yyyy-MM-dd'), 0, 'dd'), 'yyyyMMdd')
),
liv_dedup AS (
  SELECT
    signature_id,
    pt,
    model_result,
    ROW_NUMBER() OVER (PARTITION BY signature_id, pt ORDER BY pt DESC) AS rn
  FROM liv_raw
),
joined AS (
  SELECT
    b.real_region,
    b.pt,
    b.signature_id,
    l.model_result
  FROM base b
  JOIN liv_dedup l
    ON b.signature_id = l.signature_id
    AND b.pt = l.pt
    AND l.rn = 1
)
SELECT
  real_region,
  COUNT(DISTINCT signature_id) AS liveness_attempts,
  SUM(
    CASE
      WHEN GET_JSON_OBJECT(model_result, '$.result.details.rejectModules') IS NULL
        OR GET_JSON_OBJECT(model_result, '$.result.details.rejectModules') = '[]'
        OR TRIM(COALESCE(GET_JSON_OBJECT(model_result, '$.result.details.rejectModules'), '')) = ''
      THEN 1
      ELSE 0
    END
  ) AS liveness_passed,
  ROUND(
    100.0 * SUM(
      CASE
        WHEN GET_JSON_OBJECT(model_result, '$.result.details.rejectModules') IS NULL
          OR GET_JSON_OBJECT(model_result, '$.result.details.rejectModules') = '[]'
          OR TRIM(COALESCE(GET_JSON_OBJECT(model_result, '$.result.details.rejectModules'), '')) = ''
        THEN 1
        ELSE 0
      END
    ) / NULLIF(COUNT(DISTINCT signature_id), 0),
    2
  ) AS pass_rate_pct
FROM joined
GROUP BY real_region
ORDER BY liveness_attempts DESC
LIMIT 50
