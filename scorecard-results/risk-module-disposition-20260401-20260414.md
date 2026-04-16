# Confidence Scorecard Result

**Artifact:** Risk Module 处置方式命中分析 (2026-04-01 ~ 2026-04-14)
**Overall Score:** 47 / 100
**Evaluated:** 2026-04-16

## Published Scorecard

- [View Scorecard](https://vibe-track.ngrok.app/r/risk-analysis/disposition/risk-module-disposition-20260401-20260414/confidence-scorecard.html)
- [View Original Artifact](https://vibe-track.ngrok.app/r/risk-analysis/disposition/risk-module-disposition-20260401-20260414/index.html)

## Summary

| Category | Units | L3 | L4 | Weighted |
|----------|-------|----|----|----------|
| Metrics | 6 | 0 | 6 | 1.80 |
| Dimensions | 4 | 4 | 0 | 2.40 |
| Scope & Filters | 3 | 2 | 1 | 1.50 |
| SQL Patterns | 4 | 3 | 1 | 2.10 |
| Business Rules | 3 | 2 | 1 | 1.50 |
| **Total** | **20** | **11** | **9** | **9.30/20 = 47/100** |

## Key Findings

- **Zero L1 or L2 matches**: No logic units are documented in active skill sets or validated by prior published artifacts
- **11 units at L3 (Explored)**: Tables, columns, and mappings were found in session exploration history (2026-02 to 2026-04)
- **9 units at L4 (Inferred)**: All metric definitions and some business rules lack corroboration
- **Disposition mapping** comes from a Lark chat screenshot (not version-controlled)
- **空跑 (dry-run) rules** reported 0 hits — deployment status unverified

## Top Recommendations

1. Document disposition mapping (hit_code → 直接拦截/送人审/空跑) in the `ekyc-risk-analysis` skill set
2. Add formal metric definitions (UV, PV, 重复率, 命中率) to skill set
3. Verify that 空跑 rules are actually deployed in production
4. Cross-validate with Bybit daily report numbers for the same period
