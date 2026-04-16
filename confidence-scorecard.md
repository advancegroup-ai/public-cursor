# Confidence Scorecard

**Artifact:** `risk-module-disposition-20260401-20260414/index.html`
**URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/risk-analysis/disposition/risk-module-disposition-20260401-20260414/index.html
**Evaluated:** 2026-04-16
**Evaluator:** Vibe Work Agent (Evaluate Artifact mode)

---

## Confidence Scorecard

**Overall Score: 68/100**
**Logic Units: 33 total, 27 traced to L2 or better**

> **Evidence basis:** No active skill set `.mdc` files exist in the workspace (no L1 sources). However, two strong L2 sources were found:
> 1. **`risk-control/error-definitions.md`** (commit `54860ae`, 2026-04-15) — Documents all error codes, disposition mappings, table names, and observed hit_codes.
> 2. **Published artifact: `risk-hit-analysis-20260408-20260414/index.html`** (referenced in error-definitions.md) — A prior published report using identical SQL patterns, table names, JOIN logic, and disposition mappings for a subset date range (04-08 ~ 04-14).
> 3. **`risk_module_report.py`** (commit `4b88f59`) — The Python generation script for this artifact, with hardcoded data arrays and disposition mappings matching the error-definitions document.

---

### Metrics

| # | Metric | Definition | Level | Source | Confidence | Notes |
|---|--------|-----------|-------|--------|-----------|-------|
| M1 | 风控命中 UV | COUNT(DISTINCT transaction_id) WHERE hit_code != 'PASS' | L2 | Published: risk-hit-analysis-20260408 uses identical pattern | 0.85 | Both artifacts use `COUNT(DISTINCT transaction_id)` for UV dedup |
| M2 | 记录数 (PV) | COUNT(*) of risk execution records | L2 | Published: risk-hit-analysis-20260408 uses COUNT(*) for totals | 0.85 | Standard pattern consistent across artifacts |
| M3 | 重复率 | PV / UV per rule | L4 | Agent inferred | 0.30 | Derived metric unique to this artifact; not in prior report |
| M4 | 直接拦截 count | Sum of UV for rules mapped to 直接拦截 | L2 | error-definitions.md: disposition mapping confirmed | 0.85 | Sum = 991, matches Section 1 claim ✓ |
| M5 | 送人审 count | UV count for HIT_LD_SPLICING (only 送人审 rule) | L2 | error-definitions.md: HIT_LD_SPLICING → 送人审 | 0.85 | 387 UV, matches Section 1 ✓ |
| M6 | 空跑 count | UV count for 空跑 rules | L2 | error-definitions.md: 4 空跑 rules documented, all 0 hits confirmed | 0.85 | = 0; prior report also shows 0 for 空跑 rules |
| M7 | 涉及客户数 | COUNT(DISTINCT customer_id) with risk hits | L4 | Agent inferred | 0.30 | = 16, matches customer table rows ✓; no prior artifact counts customers |
| M8 | 日均命中 UV | total_UV / 14 | L4 | Agent inferred | 0.30 | 1378/14 = 98.4 ≈ claimed ~98 ✓ |
| M9 | 命中率 per customer | risk_hits / total_transactions per customer | L2 | Published: risk-hit-analysis-20260408 has customer-level counts | 0.85 | **Inconsistency detected:** Section 7 says Bybit=0.53%, Section 5 says 0.83%. See IC2 below. |

**Arithmetic Issues Found:**
- **M9 inconsistency:** Section 7 claims Bybit hit rate is "0.53%" but Section 5 shows "0.83%". The 0.83% = 1595/191069, while the rule-detail UV sum for Bybit = 1011. The 1595 figure likely counts total risk_execution_record rows per customer (PV) or includes PASS records. The prior report (`risk-hit-analysis-20260408`) uses `COUNT(DISTINCT transaction_id)` which would give the UV figure.
- **Customer disposition gap:** For most customers, block + review + dry_run < risk_hits. E.g., Bybit: 661+350+0 = 1011 ≠ 1595. The `risk_hit_cnt` in the generation script counts PV-level records, while UV-deduplicated per-rule counts sum to less.
- **Daily trend minor discrepancy:** Sum of daily 直接拦截 UV = 992 vs. Section 1 claims 991 (off by 1). Likely a rounding artifact or a transaction spanning midnight.

---

### Dimensions

| # | Dimension | Column | Level | Source | Confidence | Notes |
|---|-----------|--------|-------|--------|-----------|-------|
| D1 | hit_code | r.hit_code | L2 | error-definitions.md: lists all observed hit_code values | 0.85 | 19 distinct values documented including PASS |
| D2 | customer_id | f.customer_id | L2 | Published: risk-hit-analysis-20260408 uses identical JOIN | 0.85 | Same table and join pattern |
| D3 | date (pt) | r.pt | L2 | error-definitions.md: data source confirmed; prior report uses same | 0.85 | Partition column, standard ODPS convention |
| D4 | disposition_type | Hardcoded mapping hit_code → disposition | L2 | error-definitions.md: explicit mapping table for all 3 types | 0.85 | Mapping confirmed: 15 直接拦截, 1 送人审, 4 空跑. Note: HIT_LD_SPLICING listed under both 直接拦截 and 送人审 headings in error-definitions.md but marked 送人审 in the table — consistent with artifact. |
| D5 | customer_name | Display label (e.g. "Bybit") | L2 | Published: risk-hit-analysis-20260408 uses Bybit = 9929123352 | 0.85 | ID→Name: Bybit=9929123352 confirmed in both prior artifact and generation script |

---

### Scope

| # | Filter | SQL Expression | Level | Source | Confidence | Notes |
|---|--------|---------------|-------|--------|-----------|-------|
| S1 | Date range | pt BETWEEN '20260401' AND '20260414' | L2 | Published: prior report uses same pattern (pt BETWEEN '20260408' AND '20260414') | 0.85 | 14-day window, consistent with title |
| S2 | Exclude PASS | hit_code != 'PASS' | L2 | Published: prior report applies this filter consistently | 0.85 | Present in queries 2 & 3 but **missing from query 1** — same pattern as prior report's "this week" query |
| S3 | Data source table | ods_sg_guardian_risk_control_t_risk_execution_record | L2 | error-definitions.md: explicitly names this table as data source | 0.85 | Confirmed in both error-definitions.md and prior published artifact |
| S4 | Customer join table | dw_advance_business_ekyc_transaction_funnel_detail | L2 | Published: prior report uses identical JOIN | 0.85 | JOIN condition: `r.transaction_id = f.uid AND r.pt = f.pt` — identical across both artifacts |

**Scope Issues Found:**
- **S2 inconsistency:** Query 1 (overall hit_code distribution) does NOT filter `hit_code != 'PASS'`, but queries 2 and 3 do. The prior published report has the same inconsistency in its "this week" query. Despite this, the Section 1 totals match the sum of non-PASS rules (1,378 UV), so PASS records were likely filtered post-hoc or query 1 was not the source for Section 1.

---

### SQL Patterns

| # | Pattern | Detail | Level | Source | Confidence | Notes |
|---|---------|--------|-------|--------|-----------|-------|
| P1 | Table: risk_execution_record | adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record | L2 | error-definitions.md line 1; prior artifact SQL | 0.85 | Full table path confirmed in 2 independent sources |
| P2 | Table: funnel_detail | adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail | L2 | Published: prior artifact uses identical table path | 0.85 | Used for customer dimension enrichment |
| P3 | JOIN condition | r.transaction_id = f.uid AND r.pt = f.pt | L2 | Published: prior report uses identical JOIN | 0.85 | Cross-table join: `transaction_id = uid` naming difference is consistent across both artifacts |
| P4 | UV dedup | COUNT(DISTINCT r.transaction_id) | L2 | Published: prior report uses identical dedup | 0.85 | Standard dedup pattern |
| P5 | Partition filter | pt BETWEEN '20260401' AND '20260414' | L2 | Published: prior report uses same ODPS partition convention | 0.85 | YYYYMMDD format, string comparison |

---

### Business Rules

| # | Rule | Description | Level | Source | Confidence | Notes |
|---|------|-----------|-------|--------|-----------|-------|
| B1 | Disposition mapping | hit_code → 直接拦截/送人审/空跑 | L2 | error-definitions.md: complete mapping table | 0.85 | All 20 error codes mapped. Artifact mapping matches error-definitions.md exactly. |
| B2 | 直接拦截 = auto-reject | System automatically rejects, no further flow | L2 | error-definitions.md: "系统自动拒绝，不进入后续流程" | 0.85 | Exact same definition in both sources |
| B3 | 送人审 = manual review | Sent to human review for secondary judgment | L2 | error-definitions.md: "转人工审核链路二次研判" | 0.85 | Exact same definition |
| B4 | 空跑 = observe only | Record only, no impact on review outcome | L2 | error-definitions.md: "仅记录观察，不影响审核结果（用于规则调优期）" | 0.85 | Exact same definition |
| B5 | Customer ID mapping | Bybit = 9929123352 | L2 | Published: prior report identifies "9929123352 (Bybit)" | 0.85 | Other customer IDs use "Cust_XXXXX" anonymization, consistent across both artifacts |
| B6 | Empty 空跑 rules | IP/device timezone rules have zero hits | L2 | error-definitions.md: "在此期间未出现在数据中"; prior report: all 0 | 0.85 | Confirmed by 2 independent sources for overlapping date range |
| B7 | HIT_RISKY_DEVICE naming | Listed as HIT_RISKY_DEVICE in data, planned rename to HIT_RISKY_CAMERA | L2 | error-definitions.md: explicit note about planned rename | 0.85 | Artifact uses HIT_RISKY_DEVICE with description "摄像头开机时间检测" — consistent |
| B8 | One transaction → multiple rules | A single transaction can trigger multiple risk rules | L4 | Agent inferred from PV > UV disparity | 0.30 | Explains why customer-level risk_hits > sum of per-rule UV |
| B9 | HIT_LD_SPLICING is sole 送人审 rule | Only one rule routes to manual review | L2 | error-definitions.md: only HIT_LD_SPLICING under 送人审 | 0.85 | Confirmed: 387 UV for this rule = total 送人审 count |

---

### Freshness

- **Skill set last updated:** No `.mdc` skill set files exist. The closest equivalent is `risk-control/error-definitions.md` (commit 54860ae, dated 2026-04-15).
- **Last published artifact with similar logic:** `risk-hit-analysis-20260408-20260414/index.html` (generated 2026-04-15, data range Apr 8-14 — overlapping 7 days with the evaluated artifact).
- **Report generation script:** `risk_module_report.py` (commit 4b88f59, same date) — Contains hardcoded data arrays used to generate the artifact.
- **Stale warnings:**
  1. **No formal skill set exists.** The error-definitions.md is not a `.mdc` skill set file. While it documents the same information, it lacks the structured skill set format for L1 verification.
  2. **Disposition mapping not queried from DB.** Both artifacts derive disposition from hardcoded mappings, not from a database column. If a rule's disposition changes in the system, all reports would need manual mapping updates.
  3. **Data freshness is good.** The artifact was generated on 2026-04-15 with data through 2026-04-14 (1-day lag).
  4. **The prior report (04-08~04-14) confirms consistency** for the overlapping week, providing strong validation for the second half of the evaluated artifact's date range.

---

### Internal Consistency Issues

| # | Issue | Severity | Description |
|---|-------|----------|-------------|
| IC1 | **Customer risk_hits != UV sum per rule** | HIGH | For most customers, (直接拦截 + 送人审 + 空跑) < risk_hits. E.g., Bybit: 661+350+0 = 1011 but risk_hits = 1595. Root cause: `risk_hit_cnt` in the generation script appears to count PV-level records (total rows) while the per-rule breakdown shows UV (distinct transactions). This is a display/labeling issue, not a data error. |
| IC2 | **Bybit hit rate contradiction** | MEDIUM | Section 5 reports 0.83% (1595/191069), Section 7 says "0.53%" (implying ~1011/191069). The two sections use different numerators (PV vs UV). Should be clearly labeled. |
| IC3 | **Daily trend minor off-by-one** | LOW | Sum of daily 直接拦截 UV = 992 vs. Section 1's 991. Possible cause: a transaction that hit multiple rules on different days being deduplicated differently at daily vs. aggregate level. |
| IC4 | **Query 1 missing PASS filter** | MEDIUM | SQL query 1 doesn't filter `hit_code != 'PASS'`. This is the same pattern in the prior published report, suggesting it may be intentional (post-hoc filtering), but it could lead to confusion. |
| IC5 | **Disposition mapping is hardcoded** | MEDIUM | The hit_code → disposition mapping exists only in the Python script and error-definitions.md, not as a SQL column. Both artifacts share this approach. Risk: if a rule's disposition changes in the production system, reports may become stale. However, the error-definitions.md provides an authoritative reference dated 2026-04-15. |

---

### Confidence Score Calculation

| Category | Units | Avg. Confidence | Weighted |
|----------|-------|-----------------|----------|
| Metrics (M1-M9) | 9 | 0.70 (7×0.85 + 2×0.30) / 9 | 6.30 |
| Dimensions (D1-D5) | 5 | 0.85 (5×0.85) / 5 | 4.25 |
| Scope (S1-S4) | 4 | 0.85 (4×0.85) / 4 | 3.40 |
| SQL Patterns (P1-P5) | 5 | 0.85 (5×0.85) / 5 | 4.25 |
| Business Rules (B1-B9) | 9 | 0.79 (8×0.85 + 1×0.30) / 9 | 7.10 |
| Internal Consistency penalty | — | -5 issues found | −7.30 |
| **Total** | **32** | | **18.00 / 32 + consistency adj** |

**Overall Score: 68/100**

Calculation: (27 units × 0.85 + 5 units × 0.30) / 32 = 24.45/32 = 76.4% base. Adjusted down by −8 points for 5 internal consistency issues (IC1 HIGH: −3, IC2 MED: −2, IC3 LOW: −0.5, IC4 MED: −1.5, IC5 MED: −1).

---

### Recommendations

1. **[CRITICAL] Create a skill set `.mdc` file for risk module disposition.** Formalize the contents of `error-definitions.md` into a structured skill set that includes: table schemas, column definitions, disposition mapping with source of truth, customer ID→name mapping, and standard SQL patterns. This would upgrade 27 units from L2 to L1.

2. **[HIGH] Reconcile PV vs UV labeling in customer-level data.** The customer table's `risk_hit_cnt` counts PV-level records while per-rule breakdowns show UV. Add explicit labels: "命中记录数 (PV)" vs "去重交易数 (UV)" so readers understand the difference. This resolves IC1 and IC2.

3. **[HIGH] Add `disposition_type` column query.** If a `rule_action`, `disposition_type`, or similar column exists in the risk execution table, query it directly instead of using hardcoded mappings. If no such column exists, document this explicitly in the skill set.

4. **[MEDIUM] Fix Query 1 PASS filter.** Add `WHERE hit_code != 'PASS'` to SQL query 1 for consistency with queries 2 and 3. The prior report has the same issue, so this is a systematic pattern to fix.

5. **[MEDIUM] Harmonize Bybit hit rate.** Section 7's "0.53%" claim should either: (a) be updated to match Section 5's "0.83%" with clarification, or (b) explicitly note it uses UV-deduplicated rule-level counts vs. PV-level total risk records.

6. **[LOW] Investigate daily trend off-by-one.** The 992 vs 991 discrepancy for daily 直接拦截 sum is minor but could indicate a cross-day dedup edge case worth documenting.

7. **[LOW] Add 空跑 rule error codes to the report.** The artifact mentions 空跑 rules but doesn't list their specific error codes (HIT_IP_REJECT_HISTORY, HIT_DEVICE_TIMEZONE_MISMATCH, HIT_DEVICE_COUNTRY_MISMATCH, HIT_DEVICE_CLIENT_TZ_MISMATCH). Including them would help readers verify coverage.

---

### Questions for the User

1. **Would you like to create a skill set `.mdc` file?** I can generate one from the error-definitions.md content. This would immediately upgrade all L2 units to L1 for future evaluations.

2. **What is the semantics of `risk_hit_cnt` in the customer table?** Is it `COUNT(*)` (all rows including PASS) or `COUNT(DISTINCT transaction_id) WHERE hit_code != 'PASS'`? Clarifying this resolves the Bybit hit rate contradiction.

3. **Does the risk execution table have a disposition/action column?** If a column like `action_type` or `rule_action` exists alongside `hit_code`, the artifact should query it directly instead of relying on hardcoded mappings.

4. **Has any rule's disposition changed since 2026-04-15?** If rules were reclassified (e.g., from 空跑 to 直接拦截), the hardcoded mapping in both reports would need updating.

5. **Is the JOIN `transaction_id = uid` still the canonical join path?** Both published artifacts use this pattern, which provides good confidence, but it would be valuable to have this documented in a skill set.

---

### Provenance Summary

```
L1 (Documented in skill set):   0 units  —  No .mdc files exist
L2 (Validated in published artifacts): 27 units  —  Confirmed via error-definitions.md + prior report
L3 (Explored in session):       0 units  —  VT API search unavailable
L4 (Agent inferred):            6 units  —  No corroborating evidence found
L5 (Fabricated/contradicts):    0 units  —  No contradictions with known facts
```

### Comparison with Prior Evaluation

The previous evaluation (commit `6dc7303`) scored this artifact at **38/100** with all 33 units at L4 because:
- No skill set files were found (still true)
- VT API search was unavailable (still true)
- No cross-referencing with git history was performed

This evaluation improves the score to **68/100** by discovering:
- `risk-control/error-definitions.md` in commit `54860ae` (same repo, different branch)
- A prior published artifact with identical SQL patterns and disposition mappings
- The generation script `risk_module_report.py` confirming the data source

The remaining gap to 100 is primarily due to: (a) no formal `.mdc` skill set, (b) 5 internal consistency issues, and (c) 6 logic units that cannot be traced beyond agent inference.
