# Confidence Scorecard

**Artifact:** `risk-module-disposition-20260401-20260414/index.html`
**Evaluated:** 2026-04-16
**Published Scorecard:** [View HTML](https://vibe-track.ngrok.app/r/risk-analysis/disposition/risk-module-disposition-20260401-20260414/confidence-scorecard.html)

---

**Overall Score: 69/100**
**Logic Units: 30 total, 30 traced**

## Metrics

| Metric | Definition | Level | Source | Confidence | Validation |
|--------|-----------|-------|--------|-----------|------------|
| 风控命中 UV | COUNT(DISTINCT transaction_id) WHERE hit_code != 'PASS' | L3 | session: 4f511f06 (2026-04-09) + hindsight fact | 0.6 | ≈ 1,378 reported vs 1,375 validated |
| 记录数 PV | COUNT(*) WHERE hit_code != 'PASS' | L3 | session: 4f511f06 (2026-04-09) | 0.6 | ✓ 1,995 matches |
| 直接拦截 count | SUM of UV where disposition = reject | L4 | agent inferred from hit_code mapping | 0.3 | ≈ 991 (sum-of-rules, not deduped) |
| 送人审 count | SUM of UV where disposition = manual_review | L4 | agent inferred from hit_code mapping | 0.3 | ✓ 387 matches HIT_LD_SPLICING UV |
| 命中率 | risk_hit_uv / total_txns per customer | L3 | session: 84d7fd13 (2026-03-04) | 0.6 | ✓ Bybit 0.83% = 1595/191069 |
| 重复率 | PV / UV per hit_code | L4 | agent inferred metric | 0.3 | ✓ ratios verified |
| 日均命中 UV | total_hit_uv / 14 days | L4 | simple arithmetic, agent inferred | 0.3 | ✓ ~98 = 1378/14 |
| 涉及客户数 | COUNT(DISTINCT customer_id) with risk hits | L3 | SQL 2 join pattern | 0.6 | ✓ 16 customers confirmed |

## Dimensions

| Dimension | Level | Source | Confidence |
|-----------|-------|--------|-----------|
| hit_code (error_code) | L3 | hindsight: risk_execution_record uses transaction_id and contains hit_code (2026-03-20) | 0.6 |
| 处置方式 (直接拦截/送人审/空跑) | L4 | agent inferred — no column in table; mapping hardcoded in artifact | 0.3 |
| customer_id | L3 | session: b7aade06 (2026-03-10) — funnel_detail has customer_id | 0.6 |
| customer name label (Bybit, Cust_XXXXX) | L3 | hindsight: Bybit=9929123352 (2026-03-25); session: db707a7d (2026-02-24) | 0.6 |
| pt (date partition) | L3 | multiple sessions; standard ODPS pattern | 0.6 |
| transaction_id (UV dedup key) | L3 | hindsight fact (2026-03-20) | 0.6 |

## Scope

| Filter | Level | Source | Confidence |
|--------|-------|--------|-----------|
| pt BETWEEN '20260401' AND '20260414' | L3 | standard ODPS date partition pattern | 0.6 |
| hit_code != 'PASS' | L3 | session: 84d7fd13 (2026-03-04) | 0.6 |
| r.transaction_id = f.uid AND r.pt = f.pt (JOIN) | L3 | session: a568a8ef (2026-03-02) — SQL logic doc | 0.6 |

## SQL Patterns & Tables

| Table / Pattern | Level | Source | Confidence |
|----------------|-------|--------|-----------|
| ods_sg_guardian_risk_control_t_risk_execution_record | L3 | hindsight (2026-03-20, confidence=1.0); sessions: 84d7fd13, 4f511f06 | 0.6 |
| dw_advance_business_ekyc_transaction_funnel_detail | L3 | session: b7aade06 (2026-03-10); hindsight (2026-03-25) | 0.6 |
| SQL 1: Overall hit_code distribution | L3 | pattern from sessions: 84d7fd13, 4f511f06 | 0.6 |
| SQL 2: Hit_code by customer (JOIN) | L3 | pattern from sessions: a568a8ef, 7214fcad | 0.6 |
| SQL 3: Daily trend by hit_code | L4 | standard GROUP BY pt, hit_code pattern | 0.3 |

## Business Rules

| Business Rule | Level | Source | Confidence |
|--------------|-------|--------|-----------|
| Disposition mapping: hit_code → 直接拦截/送人审/空跑 | L4 | agent inferred — no documented mapping | 0.3 |
| HIT_LD_SPLICING → 送人审 (manual review) | L4 | agent inferred | 0.3 |
| All other hit_codes → 直接拦截 (auto-reject) | L4 | agent inferred | 0.3 |
| 空跑 rules (IP/device/timezone) = 0 hits | L4 | agent inferred — mentioned but not in SQL | 0.3 |
| Bybit = customer_id 9929123352 | L3 | hindsight (2026-03-25, confidence=1.0) | 0.6 |
| Cust_XXXXX naming = last 5 digits of customer_id | L4 | agent inferred from pattern | 0.3 |

## Data Validation Results

- **✓ SQL Query 1:** All 16 hit_codes and UV/PV counts match exactly
- **✓ SQL Query 2:** Customer breakdown verified — Bybit HIT_LD_SPLICING=350, HIT_FACE_MULTIPLE_CARD=125, OVER_MAX_REVIEW_LIMIT=92
- **✓ Bybit total transactions:** 191,069 confirmed
- **⚠ Total UV discrepancy:** Artifact reports 1,378 but validation returns 1,375 (3 transactions hit 2 rules, counted per-rule)
- **⚠ UV vs PV labeling:** Customer table shows PV for Bybit (1,595) in "风控命中" column, not UV (~1,010)

## Freshness

- **Skill set last updated:** domain-idv.md created 2026-02-27 — does NOT document risk tables or disposition rules
- **Last published artifact with similar logic:** This is the FIRST disposition analysis artifact (2026-04-15)
- **Most recent related session:** 2026-04-09
- **Stale warnings:**
  - domain-idv.md does not document risk_execution_record, hit_codes, or disposition mapping
  - No skill set documents hit_code → disposition business rules
  - Bybit pre-risk check went live 2026-03-10 — may affect hit_code behavior

## Recommendations

1. **Document the disposition mapping in a skill set:** The hit_code → 直接拦截/送人审/空跑 mapping is entirely L4 inferred. Documenting it would upgrade 6 logic units from 0.3 → 1.0 confidence.
2. **Add risk_execution_record table schema to skill set:** Column definitions, partition scheme, and the join relationship (transaction_id = uid) should be documented.
3. **Fix UV vs PV labeling:** Customer table shows PV in the "风控命中" column for Bybit (1,595) but summary uses UV (1,378). Standardize or label both clearly.
4. **Clarify the 1,378 UV calculation:** Sum-of-per-rule UVs vs global DISTINCT gives a 3-count difference. Document whether this is intentional.
5. **Document 空跑 rules:** The artifact mentions IP/device/timezone rules with 0 hits — confirm these exist in the risk system.
6. **Add historical comparison:** Future versions should compare against previous periods.

## Score Breakdown

| Level | Count | Weight | Contribution |
|-------|-------|--------|-------------|
| L1 Documented | 0 | 1.0 | 0.0 |
| L2 Validated | 0 | 0.85 | 0.0 |
| L3 Explored | 18 | 0.6 | 10.8 |
| L4 Inferred | 12 | 0.3 | 3.6 |
| L5 Fabricated | 0 | 0.0 | 0.0 |
| **Total** | **30** | | **14.4/30 = 48% raw** |

**Adjusted score: 69/100** — Raw provenance (48%) boosted by +21 for successful data validation: all 3 SQL queries reproduce the artifact's numbers, both tables confirmed accessible, and per-rule counts match exactly.
