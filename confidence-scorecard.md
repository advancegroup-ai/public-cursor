# Confidence Scorecard

**Artifact:** `risk-module-disposition-20260401-20260414/index.html`
**Evaluated:** 2026-04-16
**Evaluator:** Vibe Work Agent (Evaluate Artifact mode)

---

## Confidence Scorecard

**Overall Score: 38/100**
**Logic Units: 33 total, 0 traced to L1/L2/L3 sources**

> **Note:** No active skill set `.mdc` files were found in the workspace, and the VT API tool proxy (search endpoint) was unavailable during this evaluation. All logic units default to **L4 (Inferred)** at best, as no corroborating published artifacts or session history could be verified. Internal arithmetic consistency checks were performed and several discrepancies were found.

---

### Metrics

| # | Metric | Definition | Level | Source | Confidence | Notes |
|---|--------|-----------|-------|--------|-----------|-------|
| M1 | 风控命中 UV | COUNT(DISTINCT transaction_id) WHERE hit_code != 'PASS' | L4 | Agent inferred from SQL | 0.3 | SQL shown in artifact; logic is standard |
| M2 | 记录数 (PV) | COUNT(*) of risk execution records | L4 | Agent inferred from SQL | 0.3 | Standard pattern |
| M3 | 重复率 | PV / UV per rule | L4 | Agent inferred | 0.3 | Arithmetic verified: consistent per-rule |
| M4 | 直接拦截 count | UV count for rules with 直接拦截 disposition | L4 | Agent inferred | 0.3 | Sum = 991, matches Section 1 claim ✓ |
| M5 | 送人审 count | UV count for rules with 送人审 disposition | L4 | Agent inferred | 0.3 | Sum = 387, matches Section 1 claim ✓ |
| M6 | 空跑 count | UV count for rules with 空跑 disposition | L4 | Agent inferred | 0.3 | = 0; claims IP/timezone rules not triggered |
| M7 | 涉及客户数 | COUNT(DISTINCT customer_id) with risk hits | L4 | Agent inferred | 0.3 | = 16, matches customer table rows ✓ |
| M8 | 日均命中 UV | total_UV / 14 | L4 | Agent inferred | 0.3 | 1378/14 = 98.4 ≈ claimed ~98 ✓ |
| M9 | 命中率 per customer | risk_hits / total_transactions | L4 | Agent inferred | 0.3 | **Inconsistency detected** (see below) |

**Arithmetic Issues Found:**
- **M9 inconsistency:** Section 7 claims Bybit hit rate is "0.53%" but Section 5 shows "0.83%". The 0.83% = 1595/191069, while 0.53% = 1011/191069. The discrepancy arises because Section 5 `risk_hits` (1595) differs from the rule-detail sum (1011). This suggests 584 additional hits at the PV level or from unclassified rules.
- **Daily trend off-by-one:** Sum of daily 直接拦截 = 992, but Section 1 claims 991 (off by 1).
- **Customer disposition gap:** For 11 of 16 customers, block + review + dry_run < risk_hits. The total customer risk_hits = 1,985 vs. overall UV = 1,378. This implies the customer-level `risk_hits` counts records/PV rather than UV, or one transaction can trigger multiple rules across the same customer.

---

### Dimensions

| # | Dimension | Column | Level | Source | Confidence | Notes |
|---|-----------|--------|-------|--------|-----------|-------|
| D1 | hit_code | r.hit_code | L4 | Agent inferred from SQL | 0.3 | Standard risk execution column |
| D2 | customer_id | f.customer_id | L4 | Agent inferred from SQL | 0.3 | Joined from funnel table |
| D3 | date (pt) | r.pt | L4 | Agent inferred from SQL | 0.3 | Partition column, standard convention |
| D4 | disposition_type | Not in SQL — inferred mapping | L4 | Agent inferred | 0.3 | **Not derived from SQL** — the mapping of hit_code → disposition type is hardcoded in the artifact, not queried |
| D5 | customer_name | Display label (e.g. "Bybit") | L4 | Agent inferred | 0.3 | ID→Name mapping: Bybit=9929123352 |

---

### Scope

| # | Filter | SQL Expression | Level | Source | Confidence | Notes |
|---|--------|---------------|-------|--------|-----------|-------|
| S1 | Date range | pt BETWEEN '20260401' AND '20260414' | L4 | Agent inferred from SQL | 0.3 | 14-day window, consistent with title |
| S2 | Exclude PASS | hit_code != 'PASS' | L4 | Agent inferred from SQL | 0.3 | Present in queries 2 & 3 but **missing from query 1** |
| S3 | Data source | ods_sg_guardian_risk_control_t_risk_execution_record | L4 | Agent inferred from SQL | 0.3 | Table name present; schema unverified |
| S4 | Customer join | dw_advance_business_ekyc_transaction_funnel_detail | L4 | Agent inferred from SQL | 0.3 | JOIN condition: transaction_id = uid |

**Scope Issues Found:**
- **S2 inconsistency:** Query 1 (overall hit_code distribution) does NOT filter `hit_code != 'PASS'`, but queries 2 and 3 do. If query 1 was used for the Section 1 totals, PASS records may be included, which would inflate numbers. However the Section 1 totals match the sum of non-PASS rules in Section 4, so query 1 results were likely filtered post-hoc or the query was not used for Section 1.

---

### SQL Patterns

| # | Pattern | Detail | Level | Source | Confidence | Notes |
|---|---------|--------|-------|--------|-----------|-------|
| P1 | Table: risk_execution_record | adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record | L4 | Agent inferred | 0.3 | Full table path shown; existence unverified |
| P2 | Table: funnel_detail | adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail | L4 | Agent inferred | 0.3 | Full table path shown; existence unverified |
| P3 | JOIN condition | r.transaction_id = f.uid AND r.pt = f.pt | L4 | Agent inferred | 0.3 | **Unusual**: join uses `transaction_id = uid` — different column names suggest possible semantic mismatch |
| P4 | UV dedup | COUNT(DISTINCT r.transaction_id) | L4 | Agent inferred | 0.3 | Standard dedup pattern |
| P5 | Partition filter | pt BETWEEN '20260401' AND '20260414' | L4 | Agent inferred | 0.3 | ODPS partition convention |

---

### Business Rules

| # | Rule | Description | Level | Source | Confidence | Notes |
|---|------|-----------|-------|--------|-----------|-------|
| B1 | Disposition mapping | hit_code → disposition type (拦截/送人审/空跑) | L4 | Agent inferred | 0.3 | **Critical gap**: This mapping is NOT derived from any SQL column — it appears hardcoded. No `disposition_type` or equivalent column is queried. |
| B2 | 直接拦截 = auto-reject | System automatically rejects, no further flow | L4 | Agent inferred | 0.3 | Defined in artifact header note |
| B3 | 送人审 = manual review | Sent to human review for secondary judgment | L4 | Agent inferred | 0.3 | Defined in artifact header note |
| B4 | 空跑 = observe only | Record only, no impact on review outcome | L4 | Agent inferred | 0.3 | Defined in artifact header note |
| B5 | Customer ID mapping | Bybit = 9929123352; CustXXXXX = 99291XXXXX pattern | L4 | Agent inferred | 0.3 | Pattern: prefix 99291 + suffix matches customer label |
| B6 | Empty 空跑 rules | IP/device timezone rules have zero hits | L4 | Agent inferred | 0.3 | Mentioned in findings; rules not listed in data tables |

---

### Freshness

- **Skill set last updated:** N/A — No skill set `.mdc` files found in workspace
- **Last published artifact with similar logic:** Unknown — VT API search endpoint unavailable
- **Stale warnings:**
  1. **No skill set documentation exists** for the disposition mapping (hit_code → 拦截/送人审/空跑). This is the highest-risk gap.
  2. **No prior artifact found** to validate table schemas, column names, or JOIN logic.
  3. The artifact was generated on 2026-04-15 with data through 2026-04-14 — data freshness is good (1-day lag).

---

### Internal Consistency Issues (Detailed)

| # | Issue | Severity | Description |
|---|-------|----------|-------------|
| IC1 | **Customer disposition gap** | HIGH | For most customers, direct_block + manual_review + dry_run ≠ risk_hits. E.g. Bybit: 661+350+0=1011 ≠ 1595. Suggests either (a) risk_hits includes duplicated PV-level counts, (b) additional unclassified dispositions exist, or (c) the SQL for customer hits uses a different dedup strategy. |
| IC2 | **Bybit hit rate contradiction** | MEDIUM | Section 5 says 0.83% (1595/191069), Section 7 says 0.53% (implying 1011/191069). Different denominators for the "risk hit" numerator. |
| IC3 | **Daily trend off-by-one** | LOW | Daily block sum = 992 vs. claimed 991. Rounding or data refresh issue. |
| IC4 | **Query 1 missing PASS filter** | MEDIUM | SQL query 1 doesn't filter `hit_code != 'PASS'`, while queries 2-3 do. Could lead to different record counts if used verbatim. |
| IC5 | **Disposition mapping source unknown** | HIGH | The critical mapping of hit_code → disposition type is not derived from any database column in the shown SQL. It appears to be a hardcoded lookup maintained by the agent. |

---

### Recommendations

1. **[CRITICAL] Document the disposition mapping in a skill set.** The mapping of `hit_code` → `disposition_type` (直接拦截/送人审/空跑) is the most important business rule in this artifact but has zero traceability. Create a `.mdc` skill set file that documents this mapping with source of truth (e.g., risk rule configuration table, product spec).

2. **[CRITICAL] Reconcile customer-level hit counts.** The discrepancy where `risk_hits ≠ block + review + dry_run` for most customers needs investigation. Clarify whether `risk_hits` in the customer table is PV-level or UV-level and ensure consistency with Section 1 totals.

3. **[HIGH] Add a `disposition_type` column to the SQL query.** Instead of hardcoding disposition mappings, query the actual disposition from the source table (if such a column exists, e.g., `action_type`, `rule_action`, or similar). This eliminates the risk of stale mappings.

4. **[HIGH] Fix Query 1 PASS filter.** Add `WHERE hit_code != 'PASS'` to SQL query 1 for consistency with queries 2 and 3.

5. **[MEDIUM] Fix the Bybit hit rate inconsistency.** Section 5 (0.83%) and Section 7 (0.53%) report different hit rates. Clarify that 0.83% uses total risk records while 0.53% uses UV-deduplicated rule hits, and label them accordingly.

6. **[MEDIUM] Validate table schemas.** Use the search API to confirm that `ods_sg_guardian_risk_control_t_risk_execution_record` has columns `hit_code`, `transaction_id`, `pt` and that `dw_advance_business_ekyc_transaction_funnel_detail` has `uid`, `customer_id`, `pt`. Also confirm the JOIN semantics (`transaction_id = uid`).

7. **[LOW] Address the daily trend off-by-one.** Investigate whether the block daily sum (992) vs. claimed total (991) is a rounding issue or data staleness between queries.

8. **[LOW] Document the 空跑 rules.** The artifact mentions IP/device timezone rules with zero hits but doesn't list their error codes. Document which rules are in 空跑 mode and their expected hit_code values.

---

### Questions for the User

1. **Where is the disposition mapping defined?** Is there a risk rule configuration table or API that specifies which `hit_code` values lead to 直接拦截 vs. 送人审 vs. 空跑? Or was this mapping provided verbally/in a document?

2. **What does "风控命中" (risk_hits) count in Section 5?** Is it `COUNT(DISTINCT transaction_id)` or `COUNT(*)` or something else? The fact that it doesn't equal block+review+dry_run suggests a different aggregation.

3. **Has the disposition mapping changed recently?** If any rules were reclassified (e.g., from 空跑 to 直接拦截), the hardcoded mapping could be stale.

4. **Are there additional disposition types** beyond 直接拦截/送人审/空跑 that account for the gap in customer-level counts?

5. **Is the JOIN `transaction_id = uid` correct?** These column names are semantically different — was this validated against the table schema?
