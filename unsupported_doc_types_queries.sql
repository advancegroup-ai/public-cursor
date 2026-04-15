-- =============================================================================
-- Unsupported Document Types Analysis (自研不支持的证件类型)
-- 分析范围: 最近 30 天 (20260316 - 20260415)
-- 分析目标: 统计单点 Global OCR 和 Solution 中自研 OCR 不支持的证件类型的比例和量
-- =============================================================================

-- ===========================
-- Part 0: 表结构探索
-- ===========================

-- 0.1 查看所有 solution_code 及其流量
SELECT
    solution_code,
    COUNT(DISTINCT uid) AS pv,
    COUNT(DISTINCT biz_user_id) AS uv
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail f
LEFT JOIN adv_guardian_data_core.dw_ekyc_uid_mapping_v2 u
    ON f.uid = u.signature_id AND f.pt = u.pt
WHERE f.pt BETWEEN '20260316' AND '20260415'
GROUP BY solution_code
ORDER BY pv DESC
LIMIT 50;

-- 0.2 查看 DOCUMENT 节点的 sub_node_type 分布
SELECT
    sub_node_type,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '20260316' AND '20260415'
  AND node_type = 'DOCUMENT'
GROUP BY sub_node_type
ORDER BY cnt DESC
LIMIT 50;

-- 0.3 采样查看 DOCUMENT 节点的 data JSON 结构
SELECT
    signature_id,
    sub_node_type,
    SUBSTR(data, 1, 1000) AS data_sample
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt = '20260415'
  AND node_type = 'DOCUMENT'
LIMIT 10;

-- 0.4 查看 ekyc_txn 中所有 idv_code 分布
SELECT
    solution_code,
    idv_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '20260316' AND '20260415'
GROUP BY solution_code, idv_code
ORDER BY cnt DESC
LIMIT 200;


-- ===========================
-- Part 1: 单点 Global OCR 分析
-- ===========================

-- 1.1 Global OCR 总量
SELECT
    COUNT(DISTINCT signature_id) AS total_calls
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '20260316' AND '20260415'
  AND solution_code LIKE '%OCR%';

-- 1.2 Global OCR 按证件类型统计 (通过 ekyc_txn 表)
SELECT
    solution_code,
    GET_JSON_OBJECT(extra_info, '$.documentType') AS document_type,
    GET_JSON_OBJECT(extra_info, '$.cardType') AS card_type,
    GET_JSON_OBJECT(extra_info, '$.countryCode') AS country_code,
    GET_JSON_OBJECT(extra_info, '$.ocrVendor') AS ocr_vendor,
    COUNT(DISTINCT signature_id) AS pv,
    ROUND(COUNT(DISTINCT signature_id) * 100.0 /
        SUM(COUNT(DISTINCT signature_id)) OVER(), 2) AS pct
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '20260316' AND '20260415'
  AND solution_code LIKE '%OCR%'
GROUP BY
    solution_code,
    GET_JSON_OBJECT(extra_info, '$.documentType'),
    GET_JSON_OBJECT(extra_info, '$.cardType'),
    GET_JSON_OBJECT(extra_info, '$.countryCode'),
    GET_JSON_OBJECT(extra_info, '$.ocrVendor')
ORDER BY pv DESC
LIMIT 200;

-- 1.3 Global OCR 不支持的证件类型 (自研不支持 = 使用三方 vendor)
-- 通常自研 OCR vendor 标记为 'INHOUSE' 或 'SELF', 三方为 'VENDOR_XXX'
SELECT
    GET_JSON_OBJECT(extra_info, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(extra_info, '$.documentType') AS document_type,
    GET_JSON_OBJECT(extra_info, '$.cardType') AS card_type,
    COUNT(DISTINCT signature_id) AS pv,
    ROUND(COUNT(DISTINCT signature_id) * 100.0 /
        SUM(COUNT(DISTINCT signature_id)) OVER(), 2) AS pct
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '20260316' AND '20260415'
  AND solution_code LIKE '%OCR%'
GROUP BY
    GET_JSON_OBJECT(extra_info, '$.ocrVendor'),
    GET_JSON_OBJECT(extra_info, '$.documentType'),
    GET_JSON_OBJECT(extra_info, '$.cardType')
ORDER BY pv DESC
LIMIT 200;


-- ===========================
-- Part 2: 单点 Global OCR (通过 sub_sub_node 表)
-- ===========================

-- 2.1 SG 区域 - OCR sub_node 证件类型分布
SELECT
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(data, '$.countryCode') AS country_code,
    GET_JSON_OBJECT(data, '$.ocrProvider') AS ocr_provider,
    GET_JSON_OBJECT(data, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(data, '$.isInHouseSupported') AS is_inhouse,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '20260316' AND '20260415'
  AND node_type = 'DOCUMENT'
GROUP BY
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.cardType'),
    GET_JSON_OBJECT(data, '$.countryCode'),
    GET_JSON_OBJECT(data, '$.ocrProvider'),
    GET_JSON_OBJECT(data, '$.ocrVendor'),
    GET_JSON_OBJECT(data, '$.isInHouseSupported')
ORDER BY cnt DESC
LIMIT 200;

-- 2.2 ID 区域 - OCR sub_node 证件类型分布
SELECT
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(data, '$.countryCode') AS country_code,
    GET_JSON_OBJECT(data, '$.ocrProvider') AS ocr_provider,
    GET_JSON_OBJECT(data, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(data, '$.isInHouseSupported') AS is_inhouse,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.ods_advance_business_ekyc_transaction_sub_sub_node
WHERE pt BETWEEN '20260316' AND '20260415'
  AND node_type = 'DOCUMENT'
GROUP BY
    sub_node_type,
    GET_JSON_OBJECT(data, '$.documentType'),
    GET_JSON_OBJECT(data, '$.cardType'),
    GET_JSON_OBJECT(data, '$.countryCode'),
    GET_JSON_OBJECT(data, '$.ocrProvider'),
    GET_JSON_OBJECT(data, '$.ocrVendor'),
    GET_JSON_OBJECT(data, '$.isInHouseSupported')
ORDER BY cnt DESC
LIMIT 200;


-- ===========================
-- Part 3: Solution/IDV 中不支持的证件类型
-- ===========================

-- 3.1 Solution 流程中的证件类型分布 (SG)
SELECT
    f.solution_code,
    f.customer_id,
    GET_JSON_OBJECT(s.data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(s.data, '$.cardType') AS card_type,
    GET_JSON_OBJECT(s.data, '$.countryCode') AS country_code,
    GET_JSON_OBJECT(s.data, '$.ocrVendor') AS ocr_vendor,
    GET_JSON_OBJECT(s.data, '$.isInHouseSupported') AS is_inhouse,
    COUNT(DISTINCT f.uid) AS pv
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail f
JOIN adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node s
    ON f.uid = s.signature_id AND f.pt = s.pt
WHERE f.pt BETWEEN '20260316' AND '20260415'
  AND f.solution_code NOT LIKE '%OCR%'
  AND s.node_type = 'DOCUMENT'
  AND s.sub_node_type IN ('ID_OCR', 'GLOBAL_OCR', 'OCR')
GROUP BY
    f.solution_code,
    f.customer_id,
    GET_JSON_OBJECT(s.data, '$.documentType'),
    GET_JSON_OBJECT(s.data, '$.cardType'),
    GET_JSON_OBJECT(s.data, '$.countryCode'),
    GET_JSON_OBJECT(s.data, '$.ocrVendor'),
    GET_JSON_OBJECT(s.data, '$.isInHouseSupported')
ORDER BY pv DESC
LIMIT 200;

-- 3.2 Solution 中不支持的证件类型汇总 (按 vendor 区分自研 vs 三方)
SELECT
    CASE
        WHEN GET_JSON_OBJECT(s.data, '$.ocrVendor') IN ('INHOUSE', 'SELF', 'inhouse', 'self')
             OR GET_JSON_OBJECT(s.data, '$.isInHouseSupported') = 'true'
        THEN '自研支持'
        ELSE '自研不支持(三方)'
    END AS support_type,
    GET_JSON_OBJECT(s.data, '$.documentType') AS document_type,
    GET_JSON_OBJECT(s.data, '$.cardType') AS card_type,
    COUNT(DISTINCT s.signature_id) AS pv,
    ROUND(COUNT(DISTINCT s.signature_id) * 100.0 /
        SUM(COUNT(DISTINCT s.signature_id)) OVER(), 2) AS pct
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node s
WHERE s.pt BETWEEN '20260316' AND '20260415'
  AND s.node_type = 'DOCUMENT'
  AND s.sub_node_type IN ('ID_OCR', 'GLOBAL_OCR', 'OCR')
GROUP BY
    CASE
        WHEN GET_JSON_OBJECT(s.data, '$.ocrVendor') IN ('INHOUSE', 'SELF', 'inhouse', 'self')
             OR GET_JSON_OBJECT(s.data, '$.isInHouseSupported') = 'true'
        THEN '自研支持'
        ELSE '自研不支持(三方)'
    END,
    GET_JSON_OBJECT(s.data, '$.documentType'),
    GET_JSON_OBJECT(s.data, '$.cardType')
ORDER BY pv DESC
LIMIT 200;


-- ===========================
-- Part 4: 补充查询 - 特定表探索
-- ===========================

-- 4.1 尝试查询 Global OCR 专用表 (表名可能不存在，逐个尝试)
-- SELECT COUNT(*) FROM adv_guardian_data_core.dw_advance_business_global_ocr_transaction WHERE pt BETWEEN '20260316' AND '20260415';
-- SELECT COUNT(*) FROM adv_guardian_data_core.ods_sg_advance_business_global_ocr_result WHERE pt BETWEEN '20260316' AND '20260415';
-- SELECT COUNT(*) FROM adv_guardian_data_core.dw_ekyc_global_ocr_result WHERE pt BETWEEN '20260316' AND '20260415';

-- 4.2 检查 ekyc_txn 中与 OCR/document 相关的错误码
SELECT
    solution_code,
    idv_code,
    COUNT(DISTINCT signature_id) AS cnt
FROM adv_guardian_data_core.dw_advance_business_ekyc_transaction
WHERE pt BETWEEN '20260316' AND '20260415'
  AND (
    idv_code LIKE '%UNSUPPORT%'
    OR idv_code LIKE '%OCR%'
    OR idv_code LIKE '%DOC%'
    OR idv_code LIKE '%CARD%'
    OR idv_code LIKE '%UNKNOWN%'
  )
GROUP BY solution_code, idv_code
ORDER BY cnt DESC
LIMIT 200;

-- 4.3 每日趋势 - 自研不支持的证件类型占比
SELECT
    s.pt AS dt,
    COUNT(DISTINCT s.signature_id) AS total_doc_ocr,
    COUNT(DISTINCT CASE
        WHEN GET_JSON_OBJECT(s.data, '$.ocrVendor') NOT IN ('INHOUSE', 'SELF', 'inhouse', 'self')
             AND (GET_JSON_OBJECT(s.data, '$.isInHouseSupported') IS NULL
                  OR GET_JSON_OBJECT(s.data, '$.isInHouseSupported') = 'false')
        THEN s.signature_id
    END) AS unsupported_cnt,
    ROUND(COUNT(DISTINCT CASE
        WHEN GET_JSON_OBJECT(s.data, '$.ocrVendor') NOT IN ('INHOUSE', 'SELF', 'inhouse', 'self')
             AND (GET_JSON_OBJECT(s.data, '$.isInHouseSupported') IS NULL
                  OR GET_JSON_OBJECT(s.data, '$.isInHouseSupported') = 'false')
        THEN s.signature_id
    END) * 100.0 / COUNT(DISTINCT s.signature_id), 2) AS unsupported_pct
FROM adv_guardian_data_core.ods_sg_advance_business_ekyc_transaction_sub_sub_node s
WHERE s.pt BETWEEN '20260316' AND '20260415'
  AND s.node_type = 'DOCUMENT'
  AND s.sub_node_type IN ('ID_OCR', 'GLOBAL_OCR', 'OCR')
GROUP BY s.pt
ORDER BY s.pt;
