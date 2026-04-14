# Bybit Liveness Reports — Published Inventory

> Generated: 2026-04-14 | Source: VT Search API + OSS listing

---

## Published Reports

### 1. Bybit Liveness Weekly W15 (Demo4) ⭐ Primary
- **OSS Path:** `ekyc-liveness/idv/bybit-weekly-w15-demo4/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/idv/bybit-weekly-w15-demo4/index.html
- **Last Updated:** 2026-04-12
- **Has Recipe:** Yes (source: `demo1-bybit-1776009511`)
- **Artifact ID:** `rpt-7d64ad-tfa093-bybit-liveness-weekly-w15`

### 2. W15 Country Comparison Report
- **OSS Path:** `bybit/liveness-pass-rate/w15-2026-country-comparison/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/bybit/liveness-pass-rate/w15-2026-country-comparison/index.html
- **Last Updated:** 2026-04-14 (most recent)
- **Artifact ID:** `rpt-7d64ad-61f84e-index`

### 3. Bybit Weekly W15 (from W14 baseline)
- **OSS Path:** `ekyc-liveness/idv/bybit-weekly-w15-from-w14/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/idv/bybit-weekly-w15-from-w14/index.html

### 4. Bybit Weekly W15 (original)
- **OSS Path:** `ekyc-liveness/idv/bybit-weekly-w15/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/idv/bybit-weekly-w15/index.html

### 5. Bybit Weekly W15 (Cursor test)
- **OSS Path:** `ekyc-liveness/bybit/weekly-w15-cursor-test/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/bybit/weekly-w15-cursor-test/index.html

### 6. Bybit Weekly W14 — Vietnam Drilldown
- **OSS Path:** `ekyc-liveness/idv/bybit-weekly-w14-vnm-drilldown-20260412/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/idv/bybit-weekly-w14-vnm-drilldown-20260412/index.html
- **Also available:** subagent version at `ekyc-liveness/idv/bybit-weekly-w14-vnm-drilldown-20260412-subagent/index.html`

### 7. Bybit Weekly W14 — Vietnam Drilldown (original)
- **OSS Path:** `ekyc-liveness/idv/bybit-weekly-w14-vnm-drilldown/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc-liveness/idv/bybit-weekly-w14-vnm-drilldown/index.html

### 8. Bybit Liveness Analysis — Conversation Summary
- **OSS Path:** `bybit/liveness-analysis/conversation-summary/index.html`
- **Public URL:** https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/bybit/liveness-analysis/conversation-summary/index.html

---

## Automated Pipeline

| Name | Cron | Schedule | Last Updated |
|------|------|----------|--------------|
| Weekly Bybit Liveness Report Pipeline | `0 9 * * 1` | Every Monday 9:00 AM | 2026-04-12 |

Artifact ID: `schedule-weekly-bybit-liveness`

---

## Knowledge Context

- The Bybit daily report covers modules: **Risk, Liveness, DataVisor, Forgery, Watermark (legacy), DigitalForgery (new)**
- Metrics tracked: total PV/UV, IDV pass rates, reject counts per module, final status breakdown
- Report taxonomy follows: `{topic}/{solution}/{project[-dates]}/index.html`
- Pre-risk check module was integrated for Bybit on 2026-03-10
