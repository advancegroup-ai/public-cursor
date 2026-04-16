# Confidence Scorecard — Risk Module Disposition Report

**Artifact:** `risk-module-disposition-20260401-20260414/index.html`
**Artifact URL:** https://vibe-track.ngrok.app/r/risk-analysis/disposition/risk-module-disposition-20260401-20260414/index.html
**Audit Date:** 2026-04-16
**Auditor:** Vibe Work Agent (Evaluate Artifact Mode)

---

## Overall Score: 38/100

**Logic Units: 31 total, 0 traced to L1/L2/L3 sources**

> **Why the low score?** No active skill set `.mdc` files were found in the workspace, the VT API tool proxy was unreachable (preventing search for published artifact history or session exploration evidence), and no external documentation for the tables or business rules could be located. All logic units therefore fall to **L4 (Inferred)** by default. This does NOT mean the data is wrong — it means the provenance cannot be independently verified through the available audit channels. **Three internal consistency issues were found** that reduce confidence further.

---

## Metrics

| # | Metric | Definition in Artifact | Level | Source | Confidence | Notes |
|---|--------|----------------------|-------|--------|-----------|-------|
| M1 | 风控命中 UV | `COUNT(DISTINCT transaction_id)` where `hit_code != 'PASS'` | L4 | Agent inferred from SQL | 0.30 | SQL logic is sound; no skill set to confirm |
| M2 | 记录数 PV | `COUNT(*)` from risk_execution_record | L4 | Agent inferred from SQL | 0.30 | Internally consistent: per-rule PV sum = 1,995 ✓ |
| M3 | 直接拦截 UV | UV count of transactions with block disposition | L4 | Agent inferred | 0.30 | 991 = 71.9% of 1,378 ✓ arithmetic correct |
| M4 | 送人审 UV | UV count of transactions sent to manual review | L4 | Agent inferred | 0.30 | 387 = 28.1% of 1,378 ✓ arithmetic correct |
| M5 | 空跑 UV | UV count of dry-run (observe-only) hits | L4 | Agent inferred | 0.30 | Claimed = 0; plausible but unverifiable |
| M6 | 涉及客户数 | Distinct customers with risk hits | L4 | Agent inferred | 0.30 | 16 customers listed in Section 5 ✓ |
| M7 | 日均命中 UV | Total UV / 14 days | L4 | Agent inferred | 0.30 | 1,378/14 = 98.4 ≈ 98 ✓ |
| M8 | 命中率 | risk_hits / total_transactions per customer | **L5** | **CONTRADICTED** | **0.00** | **Section 5 uses PV-based rate (0.83%) but Section 7 uses UV-based rate (0.53%) for Bybit — see Issue #1** |
| M9 | 重复率 | PV / UV per rule | L4 | Agent inferred | 0.30 | Spot-checked: 452/157 = 2.88 ≈ 2.9x ✓ |
| M10 | Per-rule UV | `COUNT(DISTINCT transaction_id)` per hit_code | L4 | Agent inferred | 0.30 | Sum = 1,378 = total UV — implies single rule per txn (see Issue #4) |
| M11 | Bybit contribution % | Bybit UV / Total UV | **L5** | **CONTRADICTED** | **0.00** | **Section 7 says "约 1,010 / 1,378 = 73%" but Section 5 shows 1,595 for Bybit — see Issue #1** |

---

## Dimensions

| # | Dimension | Usage | Level | Source | Confidence | Notes |
|---|-----------|-------|-------|--------|-----------|-------|
| D1 | `hit_code` | GROUP BY in all 3 SQL queries | L4 | Agent inferred from SQL | 0.30 | Column name plausible for risk_execution_record table |
| D2 | `customer_id` | GROUP BY via JOIN with funnel table | L4 | Agent inferred from SQL | 0.30 | Requires JOIN — introduces data loss risk (Issue #3) |
| D3 | `pt` (date partition) | GROUP BY for daily trends | L4 | Agent inferred from SQL | 0.30 | Standard ODPS partition pattern |
| D4 | 处置方式 (disposition) | Derived categorization of hit_codes | L4 | Agent inferred | 0.25 | **Mapping from hit_code → disposition type is not documented anywhere findable** |

---

## Scope Filters

| # | Filter | SQL Expression | Level | Source | Confidence | Notes |
|---|--------|---------------|-------|--------|-----------|-------|
| S1 | Date range | `pt BETWEEN '20260401' AND '20260414'` | L4 | Agent inferred | 0.30 | 14-day range matches report header |
| S2 | Exclude PASS | `hit_code != 'PASS'` | L4 | Agent inferred | 0.30 | Logical for risk-only analysis; not documented |
| S3 | Customer scope | All customers (no filter) | L4 | Agent inferred | 0.30 | 16 customers found; no explicit inclusion/exclusion criteria |
| S4 | Bybit ID | `customer_id` mapped to '9929123352' | L4 | Agent inferred | 0.30 | ID-to-name mapping source unknown |

---

## SQL Patterns

| # | Pattern | Detail | Level | Source | Confidence | Notes |
|---|---------|--------|-------|--------|-----------|-------|
| P1 | Primary table | `adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record` | L4 | Agent inferred | 0.30 | Table naming follows ODS convention (`ods_sg_guardian_*`) |
| P2 | Join table | `adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail` | L4 | Agent inferred | 0.30 | DW-layer table; used only for customer attribution |
| P3 | Join condition | `r.transaction_id = f.uid AND r.pt = f.pt` | L4 | Agent inferred | 0.25 | **Join on `transaction_id = uid` is atypical naming — potential semantic mismatch** |
| P4 | Schema | `adv_guardian_data_core` | L4 | Agent inferred | 0.30 | Plausible schema name |
| P5 | UV dedup | `COUNT(DISTINCT transaction_id)` | L4 | Agent inferred | 0.30 | Standard dedup pattern |

---

## Business Rules

| # | Rule / Assumption | Level | Source | Confidence | Notes |
|---|-------------------|-------|--------|-----------|-------|
| B1 | Disposition classification: 直接拦截 / 送人审 / 空跑 | L4 | Agent inferred | 0.25 | **The mapping from hit_code → disposition is not in the SQL or any findable documentation** |
| B2 | Each transaction triggers at most one risk rule | L4 | Agent inferred | 0.25 | **Implied by UV per-rule sum = total UV (1,378). If multi-rule hits exist, they are masked** |
| B3 | 空跑 rules listed but zero hits | L4 | Agent inferred | 0.30 | Named rules (IP_REJECT_HISTORY, DEVICE_TIMEZONE, etc.) mentioned in findings but not in SQL output |
| B4 | Customer name mapping (Bybit, Cust_XXXXX) | L4 | Agent inferred | 0.30 | Bybit explicitly named; others anonymized with numeric IDs |
| B5 | "总交易数" per customer | L4 | Agent inferred | 0.25 | **Source query not disclosed in artifact; likely from funnel table but aggregation logic unknown** |

---

## Internal Consistency Issues Found

### Issue #1 — CRITICAL: Bybit Hit Rate Contradiction

**Section 5** states Bybit's hit rate as **0.83%** (= 1,595 / 191,069).
**Section 7** states Bybit's hit rate as **0.53%** (= ~1,010 / 191,069).

The discrepancy arises because Section 5's "风控命中" column appears to use **PV (record count = 1,595)** while the disposition columns (直接拦截=661, 送人审=350) use **UV (distinct transactions)**. Section 7 correctly uses the UV figure (~1,010) but states a different rate.

**Impact:** The reader cannot determine the true Bybit risk hit rate. Both 0.83% and 0.53% are presented as correct.

### Issue #2 — MODERATE: Customer Table Mixes PV and UV

In the Section 5 customer overview table:
- "风控命中" column = **1,595** for Bybit
- "直接拦截" + "送人审" + "空跑" = **661 + 350 + 0 = 1,011** for Bybit

If all columns were UV, they should match (assuming single disposition per transaction). The 584-unit gap suggests "风控命中" is PV while disposition columns are UV. This is **never stated** and confuses interpretation.

**Cross-customer verification:**
- 直接拦截 sum across all 16 customers = **955**, but overall total = **991** (gap: **36**)
- 送人审 sum across all 16 customers = **384**, but overall total = **387** (gap: **3**)
- Total gap = **39 transactions** lost in the JOIN with the funnel table

### Issue #3 — MODERATE: JOIN Data Loss

The customer-attribution query (Query 2) uses:
```sql
JOIN ... ON r.transaction_id = f.uid AND r.pt = f.pt
```

Approximately **39 risk-hit transactions** (36 direct blocks + 3 manual reviews) have no matching record in the funnel detail table. These transactions appear in the overall totals (Section 1/4) but are **silently dropped** from the customer breakdown (Section 5/6).

### Issue #4 — LOW: Single-Rule-Per-Transaction Assumption

The sum of per-rule UVs (157+387+111+...+2 = **1,378**) exactly equals the total UV (**1,378**). This means zero transactions triggered multiple different risk rules. While this could be by design (first-hit-wins architecture), it is **never explained** in the report. If the risk system can flag multiple rules per transaction, the report may be undercounting multi-rule overlap.

---

## Freshness Assessment

| Item | Value | Notes |
|------|-------|-------|
| Skill set last updated | **N/A** | No `.mdc` skill set files found in workspace |
| Last published artifact with similar logic | **Unknown** | Tool proxy unavailable; cannot search entity-log |
| Report generation date | **2026-04-15** | 1 day before audit |
| Data freshness | **2026-04-14** | Most recent partition; 2 days old at time of audit |
| Stale warnings | **Cannot assess** | No baseline to compare against |

### Stale Warnings
1. **No skill set baseline:** Without `.mdc` files, there is no reference point for what the "correct" metric definitions or table schemas should be.
2. **Disposition mapping undocumented:** The mapping from `hit_code` to disposition type (直接拦截/送人审/空跑) is embedded in report logic but not in any retrievable source.
3. **"总交易数" source unknown:** The total transaction counts per customer (e.g., Bybit = 191,069) come from an undisclosed query, making it impossible to verify the denominators used in hit rate calculations.

---

## Recommendations

### Immediate Fixes (to correct the current report)

1. **Fix the Bybit hit rate contradiction.** Decide whether "命中率" should use UV or PV as the numerator, and apply consistently across Sections 5 and 7. Recommended: use UV throughout (0.53% for Bybit).

2. **Clarify the "风控命中" column in Section 5.** Either:
   - Label it explicitly as "记录数 (PV)" and add a separate "风控命中 UV" column, OR
   - Change it to UV so it matches the disposition breakdown columns.

3. **Account for the 39 JOIN-dropped transactions.** Add a footnote or row in the customer table showing "Unattributed" transactions that couldn't be matched to a customer.

### Structural Improvements (to improve future confidence)

4. **Publish the full SQL in the artifact.** The report includes 3 of the queries, but the "总交易数" per-customer query and the disposition-mapping logic are missing. All queries should be disclosed.

5. **Document the hit_code → disposition mapping.** Create or update a skill set `.mdc` file that explicitly maps each `hit_code` to its disposition type (直接拦截/送人审/空跑).

6. **Add a data quality section.** Include: row counts, NULL rates, JOIN match rates, and partition completeness checks.

7. **Clarify multi-rule behavior.** State whether the risk system records one rule per transaction (first-hit-wins) or multiple, and adjust UV counting accordingly.

### Questions for the User to Validate

8. **Is `transaction_id = uid` the correct join key** between risk_execution_record and the funnel table? The field name mismatch (`transaction_id` vs `uid`) suggests these might not be semantically equivalent.

9. **Where does the disposition classification come from?** Is there a configuration table or code mapping `hit_code` values to 直接拦截/送人审/空跑? If so, it should be referenced.

10. **Are the 39 unmatched transactions expected?** What causes a risk execution record to have no corresponding funnel entry?

11. **Should the report track multi-rule hits?** If a transaction can trigger multiple risk rules, the current single-rule counting may understate risk exposure.

---

## Audit Limitations

This audit operated under significant constraints:

| Constraint | Impact |
|-----------|--------|
| **No skill set `.mdc` files in workspace** | Cannot establish any L1 (Documented) provenance |
| **Tool proxy (localhost:3100) unreachable** | Cannot search published artifacts (L2) or session history (L3) |
| **Tables are internal ODPS** | Cannot verify table schemas or query results externally |
| **No prior report to compare** | Cannot assess trend consistency or detect anomalies vs. previous periods |

Despite these constraints, **four internal consistency issues were identified** through arithmetic verification of the artifact's own data, demonstrating that the audit methodology can still surface meaningful findings.

---

*Generated by Vibe Work Agent — Evaluate Artifact Mode | 2026-04-16*
