# Recall: Past Sessions on Bybit Forgery Detection Rules

**Search query:** "Bybit forgery detection rules"
**Search date:** 2026-04-14
**Total results found:** 30+ matching messages across 8 distinct sessions

---

## Session 1: IDV Bypass Fraud Detection — Deep Analysis
- **Session ID:** `735452ab-e111-47c4-959a-bacd76a559cb`
- **Date:** 2026-03-20
- **User:** simontt88
- **Status:** Most comprehensive session

### Key Highlights
- Systematic exploration of forgery detection rules beyond `face_reference_id + doc_number`
- **R1 Rule** (Same doc number + different face) identified as the most precise and valuable definitive fraud rule
- Rules R2–R12 explored but found to have high false-positive rates at the pure data level
- Image-level analysis pursued: same ID template background + different face/text = forgery indicator
- Multi-pronged strategy developed:
  1. **Batch-run R1 rule** on Bybit weekly data to find bypass cases
  2. **Image URL extraction** from `sub_sub_node` for OSS image verification
  3. **Combined signal scoring** — IP mismatch, ID_FORGERY flag, hosting IP signals
  4. **Face reference clustering** — detect same-face-different-document patterns
- DataVisor data analyzed for both document phase and liveness phase
- Visual cluster report with thumbnail images generated and uploaded to OSS
- Document capture environment vs. liveness face capture consistency detection explored

---

## Session 2: IDV Bypass Fraud Detection — Deep Analysis Report
- **Session ID:** `892977f0-c6e1-4780-8e1d-1897e840bbcc`
- **Date:** 2026-03-19
- **User:** simontt88
- **Status:** Comprehensive report generated

### Key Highlights
- **Problem Statement:** 43 known bypass cases reported for Bybit
- Parallel sub-agent analysis across multiple directions:
  1. Batch R1 rule execution on Bybit week data
  2. Image URL discovery from sub_sub_node → OSS path → download verification
  3. Combined signal scoring (IP mismatch, ID_FORGERY, hosting IP)
  4. Face reference clustering (same face, different documents)
- Final comprehensive analysis report produced

---

## Session 3: Bybit FastFail Migration Impact Assessment
- **Session ID:** `7214fcad-05d3-41ea-8225-7a33d6d91448`
- **Date:** 2026-02-21
- **User:** simontt88
- **Status:** Report and ad-hoc script created

### Key Highlights
- **Goal:** Migrate selected rules to FastFail (auto-reject on hit, skip manual review)
- **Data range:** 2026-02-06 → 2026-02-12 (7 days), KYC Scopes Only
- Ad-hoc analysis script written: `adhoc_fastfail_report.py`
- Path: `/mnt/nas/public2/simon/repos/aai-pipeline/projects/bybit_risk_monitor_daily/`
- HTML report generated with Chinese conclusions
- End-to-end pass rate consistency assessed pre/post migration

---

## Session 4: Bybit IDV Case Investigation
- **Session ID:** `6ebda0db-4996-419e-9b17-8b7c8fa310b6`
- **Date:** 2026-03-18
- **User:** simontt88
- **Status:** Case-level investigation

### Key Highlights
- Investigated specific IDV case ID `2ccfe82f726faa8e`
- Analyzed why this case bypassed detection
- Deep-dive into IDV model result data and domain skills

---

## Session 5: Bybit Manual Audit Rates by Country
- **Session ID:** `736c3808-4cbc-4c84-bdce-b41811904a91`
- **Date:** 2026-03-18
- **User:** simontt88
- **Status:** Country-level analysis

### Key Highlights
- Countries analyzed: CHN, IDN, RUS, THA — all showing 100% `passButNeedByCountryCodeIso3Map`
- Manual audit rate analysis for these countries
- Rule hit vs. non-hit ratio comparison

---

## Session 6: Forgery Cases Image & Embedding Analysis
- **Session ID:** `e6fe7061-655c-48f6-9697-eae47c990028`
- **Date:** 2026-03-19
- **User:** simontt88
- **Status:** Image pipeline work

### Key Highlights
- CSV data: 44 cases, mostly Bybit forgery bypass
- AAI pipeline image download flow used to obtain ID document images + liveness captures
- `advance-guardian-cv-service` codebase explored for embedding logic
- Saved to `projects/forgery_cases/`

---

## Session 7: Pre-Risk Control Rule Analysis
- **Session ID:** `845b762c-af4c-4bbb-a02a-512c3566e294`
- **Date:** 2026-03-16
- **User:** simontt88
- **Status:** Rule analysis with 11 modules

### Key Highlights
- Pre-risk control rules (前置风控规则) analysis across 11 modules
- Cross-hit analysis with Bybit's Risk module rules and Liveness reject modules

---

## Session 8: Comprehensive Forgery Detection Rules Summary
- **Session ID:** `96d3b55a-2282-4736-8ce6-b1db14c4770e`
- **Date:** 2026-03-19
- **User:** simontt88
- **Status:** Summary compiled

### Key Highlights
- **Analysis Scope:** 43 bypass cases from IDV Bypass Report (2026/03)
- Comprehensive summary of all fraud detection rules discovered
- Rules catalogued and evaluated for precision vs. recall

---

## Hindsight Memory (Learned Facts)

The system also recalled this learned fact:
> **simon** wants to batch-run R1 rule ('same doc number + different face') on Bybit week data, find image URLs, combine signals (IP mismatch, ID_FORGERY, hosting IP) for scoring, and cluster by face reference to find same-face-different-document patterns.
> *(Confidence: 1.0 | Date: 2026-03-20)*

---

## Timeline Summary

| Date | Session | Focus |
|------|---------|-------|
| 2026-02-21 | FastFail Migration | Rule migration impact assessment, ad-hoc report |
| 2026-03-16 | Pre-Risk Control | 11-module rule analysis |
| 2026-03-18 | IDV Case Investigation | Specific case `2ccfe82f726faa8e` bypass analysis |
| 2026-03-18 | Country Audit Rates | CHN/IDN/RUS/THA manual audit analysis |
| 2026-03-19 | Deep Analysis Report | 43 bypass cases, multi-direction sub-agent analysis |
| 2026-03-19 | Forgery Cases Images | Image download, embedding analysis, 44 CSV cases |
| 2026-03-19 | Rules Summary | Comprehensive rules catalogue |
| 2026-03-20 | Extended Rule Mining | R1-R12 rules, image-level template detection, clustering |

---

*Generated by Vibe Work Agent — Recall Mode*
