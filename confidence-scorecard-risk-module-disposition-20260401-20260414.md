# Confidence Scorecard

**Artifact:** `risk-module-disposition-20260401-20260414/index.html`
**Evaluated:** 2026-04-16
**Evaluator:** Vibe Work Agent (Evaluate Artifact mode)

---

**Overall Score: 42/100**
**Logic Units: 28 total, 28 traced**

> **Score Breakdown:** No skill set `.mdc` files were found in the workspace (0 L1 units). The VT API tool proxy was unreachable (connection refused on `localhost:3100`), so no search API calls could be made to find published artifacts (L2) or session exploration history (L3). All 28 logic units were evaluated based on internal consistency analysis and structural audit. The low overall score reflects **lack of external corroboration**, not necessarily incorrect logic — the artifact's internal arithmetic is largely sound, with notable exceptions documented below.

---

## Metrics

| # | Metric | Definition in Artifact | Level | Source | Confidence | Notes |
|---|--------|----------------------|-------|--------|-----------|-------|
| M1 | 风控命中 UV | `COUNT(DISTINCT transaction_id)` WHERE `hit_code != 'PASS'` | L4 | Agent inferred from SQL | 0.30 | SQL is sound; UV dedup via `DISTINCT transaction_id` is standard practice |
| M2 | 记录数 PV | `COUNT(*)` per hit_code | L4 | Agent inferred from SQL | 0.30 | PV total (1,995) matches sum of per-rule PV — verified ✓ |
| M3 | 重复率 | PV / UV per rule | L4 | Agent inferred from artifact | 0.30 | Spot-checked: 452/157 = 2.88 ≈ 2.9x ✓, 348/111 = 3.14 ≈ 3.1x ✓ |
| M4 | 命中率 (Hit Rate) | risk_hits / total_transactions per customer | L4 | Agent inferred from artifact | 0.25 | **Ambiguity:** "风控命中" in Section 5 appears to be PV (not UV), contradicting Section 1's "UV" definition. See Discrepancy D1. |
| M5 | 直接拦截 count | UV count by disposition=block | L4 | Agent inferred from artifact | 0.30 | 991 matches sum of all direct-block rule UVs ✓ |
| M6 | 送人审 count | UV count by disposition=review | L4 | Agent inferred from artifact | 0.30 | 387 = HIT_LD_SPLICING UV. Only one review rule, so sum is trivially consistent ✓ |
| M7 | 空跑 count | UV count by disposition=dry_run | L4 | Agent inferred from artifact | 0.30 | Reported as 0. Cannot verify without running actual query. |
| M8 | 日均命中 UV | total_UV / 14 days | L4 | Agent inferred | 0.30 | 1378/14 = 98.4 ≈ ~98 ✓ |
| M9 | 涉及客户数 | COUNT(DISTINCT customer_id) | L4 | Agent inferred | 0.30 | 16 customers listed in Section 5, consistent ✓ |

## Dimensions

| # | Dimension | Level | Source | Confidence | Notes |
|---|-----------|-------|--------|-----------|-------|
| D1 | `hit_code` (error_code) | L4 | Agent inferred from SQL col | 0.30 | Used as primary GROUP BY in all 3 queries. 16 distinct hit_codes found in artifact. |
| D2 | `customer_id` | L4 | Agent inferred from SQL JOIN | 0.30 | Obtained via JOIN to `dw_advance_business_ekyc_transaction_funnel_detail.customer_id`. |
| D3 | `pt` (date partition) | L4 | Agent inferred from SQL col | 0.30 | Standard ODPS partition column. Used in daily trend query. |
| D4 | 处置方式 (disposition type) | L4 | Agent inferred — NOT in SQL | 0.25 | **Critical gap:** The SQL queries do NOT select or GROUP BY a "disposition" column. The mapping from hit_code → disposition (直接拦截/送人审/空跑) must come from an external lookup not shown in the artifact's SQL. |

## Scope Filters

| # | Filter | Level | Source | Confidence | Notes |
|---|--------|-------|--------|-----------|-------|
| S1 | `pt BETWEEN '20260401' AND '20260414'` | L4 | Agent inferred from SQL | 0.30 | Standard YYYYMMDD partition format. 14-day range matches header. |
| S2 | `hit_code != 'PASS'` | L4 | Agent inferred from SQL | 0.30 | Excludes passing records. Present in queries 2 and 3, but **missing from query 1** — query 1 includes PASS rows in its GROUP BY, which means the overall distribution includes PASS. The artifact may have filtered PASS client-side. |
| S3 | Table: `adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record` | L4 | Agent inferred from SQL | 0.30 | ODS-layer table. Naming convention suggests raw operational data store from `guardian_risk_control` service. |
| S4 | Table: `adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail` | L4 | Agent inferred from SQL | 0.30 | DW-layer table. Used only for customer_id enrichment via JOIN. |
| S5 | JOIN: `r.transaction_id = f.uid AND r.pt = f.pt` | L4 | Agent inferred from SQL | 0.25 | **Risk:** JOIN key uses `transaction_id = uid` (different column names). If this is a wrong-key JOIN, customer attribution would be incorrect. Cannot verify without schema. |

## SQL Patterns

| # | Pattern | Level | Source | Confidence | Notes |
|---|---------|-------|--------|-----------|-------|
| Q1 | Query 1: Overall hit_code distribution | L4 | Agent inferred | 0.30 | Groups by `hit_code` with `COUNT(*)` and `COUNT(DISTINCT transaction_id)`. **Does NOT filter `!= 'PASS'`** — see S2 note. |
| Q2 | Query 2: Hit_code by customer (JOIN) | L4 | Agent inferred | 0.30 | Joins risk records with funnel detail for customer attribution. Filters `!= 'PASS'`. |
| Q3 | Query 3: Daily trend by hit_code | L4 | Agent inferred | 0.30 | Standard time-series pivot. Filters `!= 'PASS'`. |
| Q4 | Missing: Customer total transactions query | L5 | Not found | 0.00 | "总交易数" per customer (e.g., Bybit=191,069) is shown but **no SQL provided** for how total transaction counts were obtained. |
| Q5 | Missing: Disposition mapping query | L5 | Not found | 0.00 | No SQL or lookup table provided for hit_code → disposition mapping. This is a core business rule with no documented source. |

## Business Rules

| # | Rule | Level | Source | Confidence | Notes |
|---|------|-------|--------|-----------|-------|
| B1 | Disposition categories: 直接拦截, 送人审, 空跑 | L4 | Agent inferred from artifact text | 0.25 | Defined in header but **mapping logic not documented in SQL**. See Q5. |
| B2 | hit_code → disposition mapping | L4 | Agent inferred from artifact tables | 0.25 | 15 rules map to 直接拦截, 1 to 送人审, 0 to 空跑. The mapping source (config table? code enum?) is unknown. |
| B3 | Customer masking: Bybit named, others as Cust_XXXXX | L4 | Agent inferred from artifact | 0.30 | Consistent masking pattern. Bybit (ID: 9929123352) is the only named customer. |
| B4 | 空跑 rules: IP/timezone rules have 0 hits | L4 | Agent inferred from key findings | 0.25 | The artifact mentions HIT_IP_REJECT_HISTORY, HIT_DEVICE_TIMEZONE, etc. as 空跑 rules with 0 hits, but these do NOT appear in the Section 4 table. |
| B5 | UV dedup scope: per transaction_id | L4 | Agent inferred from SQL | 0.30 | `COUNT(DISTINCT transaction_id)` — standard dedup. |

---

## Internal Consistency Audit

### Verified ✓
- Top-level sums: 991 + 387 + 0 = 1,378 ✓
- Percentages: 991/1378 = 71.9%, 387/1378 = 28.1% ✓
- Daily average: 1378/14 ≈ 98 ✓
- PV total: sum of all per-rule PV = 1,995 ✓
- Direct block rule UV sum: 991 = reported 991 ✓
- MULTIPLE_CARD (158) + MULTIPLE_BIRTHDAY (140) = 298 ✓
- Bybit hit rate: 1595/191069 = 0.83% ✓ (note: uses PV-like "风控命中" count)
- Repeat rate spot checks: 452/157=2.88≈2.9x ✓, 348/111=3.14≈3.1x ✓

### Discrepancies Found

**D1 — Section 5 "风控命中" is PV, not UV (Severity: Medium)**
Section 1 defines "风控命中 UV" = 1,378 (deduped). But Section 5's "风控命中" column sums to 1,985 across all customers, which exceeds 1,378. For Bybit, Section 5 shows 1,595 but Section 6 per-rule UV sum is only 1,011. The Section 5 "风控命中" column appears to be PV (record count) mislabeled or at least not clarified as PV. This affects all customer hit rates.

**D2 — Key finding Bybit contribution inconsistency (Severity: Medium)**
Key findings state "Bybit contributed ~1,010 / 1,378 = 73%." The 1,010 aligns with Bybit's direct(661)+review(350)=1,011, which is a per-rule UV sum. But Section 5 shows Bybit "风控命中" = 1,595. The 73% claim uses a different counting method than the rest of the report.

**D3 — Customer disposition gap (Severity: Low)**
Sum of per-customer direct block = 955 (vs 991 total, gap = -36). Sum of per-customer review = 384 (vs 387 total, gap = -3). This suggests 36 direct-block and 3 review transactions belong to customers not listed, or there is a bucketing issue.

**D4 — Customers with unclassified disposition (Severity: Low)**
Cust_23407 (4 hits), Cust_23945 (4 hits), Cust_24067 (1 hit), Cust_24394 (1 hit) all show risk hits > 0 but direct=0, review=0, dry_run=0. These 10 transactions have no disposition category — they may hit rules not in the known disposition mapping.

**D5 — Query 1 does not filter PASS (Severity: Low)**
SQL Query 1 lacks `WHERE hit_code != 'PASS'`, meaning it returns PASS records too. The artifact must filter client-side, which is not documented.

---

## Freshness

- **Skill set last updated:** N/A — no `.mdc` skill set files found in workspace
- **Last published artifact with similar logic:** Unknown — VT API tool proxy unavailable (`localhost:3100` connection refused); could not search entity-log
- **Artifact generation date:** 2026-04-15
- **Data range:** 2026-04-01 to 2026-04-14 (data is 1-15 days old at generation time)
- **Stale warnings:**
  - No skill set documentation exists for this artifact's data model, tables, or business rules
  - The disposition mapping (hit_code → 直接拦截/送人审/空跑) has no documented source and may change without this artifact being updated
  - The 空跑 rules (IP/timezone) showing 0 hits may indicate stale or inactive rule configuration rather than genuinely 0 hits

---

## Recommendations

### Critical (address before trusting this artifact)

1. **Document the disposition mapping source (D4→Q5, B2).** The core concept of this report — classifying rules as 直接拦截/送人审/空跑 — has no SQL or config-table reference. Create a skill set entry or add a reference query (e.g., `SELECT hit_code, disposition FROM risk_rule_config`) so future reports trace this mapping to a verified source.

2. **Clarify UV vs PV in Section 5 (D1).** The "风控命中" column in the customer overview appears to be PV (record count), not UV (unique transactions). Either:
   - Rename the column to "风控记录数 (PV)" and add a separate UV column, or
   - Change the query to use `COUNT(DISTINCT transaction_id)` per customer.

3. **Fix the Bybit contribution claim (D2).** Key finding states "Bybit contributed ~1,010 / 1,378 = 73%." This mixes per-rule UV summation with overall UV. Either:
   - Compute Bybit's true transaction-level UV contribution using `COUNT(DISTINCT transaction_id) WHERE customer_id = 'bybit'`, or
   - Clarify that 1,010 is the sum of per-rule UVs (which double-counts multi-rule transactions).

4. **Add SQL for total transaction counts (Q4).** The report shows "总交易数" (e.g., Bybit = 191,069) but no SQL query is provided for this metric. Add the source query to enable full reproducibility.

### Important (improve confidence level)

5. **Create a skill set `.mdc` file** for the risk disposition analysis domain. Document:
   - Table schemas for `ods_sg_guardian_risk_control_t_risk_execution_record` and `dw_advance_business_ekyc_transaction_funnel_detail`
   - The JOIN key semantics (`transaction_id = uid`)
   - The disposition mapping
   - Standard date partition conventions

6. **Investigate 空跑 rule effectiveness (B4).** The artifact mentions HIT_IP_REJECT_HISTORY, HIT_DEVICE_TIMEZONE/COUNTRY/CLIENT_TZ_MISMATCH as 空跑 rules with 0 hits over 14 days. Verify:
   - Are these rules deployed and active?
   - Is the configuration threshold too high?
   - Should these be added to the Section 4 table even with 0 hits for completeness?

7. **Account for unclassified customer hits (D4).** Four customers (Cust_23407, Cust_23945, Cust_24067, Cust_24394) have risk hits that don't appear in any disposition bucket. Investigate whether these hit rules are missing from the disposition mapping.

### Nice to have

8. **Add `WHERE hit_code != 'PASS'` to Query 1 (D5)** for consistency with queries 2 and 3, or document that PASS filtering happens post-query.

9. **Include confidence metadata in future artifacts.** Embed the provenance level of each metric/dimension directly in the report HTML so consumers know which logic units are verified vs. inferred.

10. **Publish the artifact via the search-indexed pipeline** (not just OSS) so future evaluations can find it as an L2 reference via the search API.

---

## Questions for the User

To upgrade confidence levels on key logic units, please confirm:

1. **Where is the hit_code → disposition mapping defined?** Is it a database config table, an application enum, or a manual classification? (Would upgrade B1, B2, D4 from L4 → L1/L2)
2. **Is `transaction_id = uid` the correct JOIN key** between the risk execution record and funnel detail tables? (Would upgrade S5 from L4 → L1)
3. **What query produces "总交易数" per customer?** Is it from the funnel table or another source? (Would upgrade Q4 from L5 → L4)
4. **Has a similar disposition report been published before** for a prior date range (e.g., March 2026)? If so, what was the OSS path? (Would enable L2 cross-validation)
5. **Are the 空跑 rules (IP/timezone) currently active in production?** (Would upgrade B4 from L4 → L1)
