import json
import subprocess

VT_TOKEN = "vt_sk_21089b3935576ce6c994c3aa2f14ad0d160b8eeafc583bd0b7583e520ecba2af"
VT_BASE = "https://vibe-track.ngrok.app"

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TikTok (TT) eKYC Analysis Report</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px;line-height:1.6}
.container{max-width:1200px;margin:0 auto}
h1{font-size:28px;font-weight:700;margin-bottom:6px;color:#fff}
.subtitle{color:#94a3b8;font-size:14px;margin-bottom:32px}
.accent{color:#3b82f6}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-bottom:32px}
.card{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155}
.card-label{font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8;margin-bottom:4px}
.card-value{font-size:28px;font-weight:700;color:#fff}
.card-sub{font-size:13px;color:#64748b;margin-top:4px}
.section{margin-bottom:32px}
.section-title{font-size:18px;font-weight:600;color:#fff;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.section-title .dot{width:8px;height:8px;border-radius:50%;background:#3b82f6}
table{width:100%;border-collapse:collapse;background:#1e293b;border-radius:12px;overflow:hidden}
thead{background:#334155}
th{padding:12px 16px;text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8;font-weight:600}
td{padding:12px 16px;font-size:14px;border-top:1px solid #1e293b}
tr:nth-child(even){background:#1e293b}
tr:nth-child(odd){background:#162032}
.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:600}
.badge-green{background:rgba(34,197,94,0.15);color:#22c55e}
.badge-yellow{background:rgba(234,179,8,0.15);color:#eab308}
.badge-red{background:rgba(239,68,68,0.15);color:#ef4444}
.badge-blue{background:rgba(59,130,246,0.15);color:#3b82f6}
.bar-container{display:flex;align-items:center;gap:8px}
.bar{height:20px;border-radius:4px;display:flex;overflow:hidden}
.bar-pass{background:#22c55e}
.bar-fail{background:#ef4444}
.bar-label{font-size:12px;color:#94a3b8;white-space:nowrap}
.trend-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:2px}
.trend-cell{background:#1e293b;padding:12px;text-align:center}
.trend-cell.header{background:#334155;font-size:12px;color:#94a3b8;font-weight:600;text-transform:uppercase}
.trend-month{font-size:11px;color:#94a3b8}
.trend-val{font-size:18px;font-weight:700;color:#fff}
.trend-delta{font-size:12px}
.up{color:#ef4444}
.down{color:#22c55e}
.insight-box{background:linear-gradient(135deg,#1e293b,#172040);border:1px solid #3b82f6;border-radius:12px;padding:20px;margin-bottom:16px}
.insight-box h3{color:#3b82f6;font-size:15px;margin-bottom:8px}
.insight-box p{color:#cbd5e1;font-size:14px}
.chart-row{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.chart-label{width:50px;text-align:right;font-size:13px;color:#94a3b8}
.chart-bar-wrap{flex:1;height:28px;background:#334155;border-radius:6px;overflow:hidden;position:relative}
.chart-bar{height:100%;border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:12px;font-weight:600;color:#fff;transition:width 0.3s}
.chart-val{font-size:13px;color:#e2e8f0;min-width:80px;text-align:right}
.footer{text-align:center;color:#475569;font-size:12px;margin-top:40px;padding-top:20px;border-top:1px solid #1e293b}
</style>
</head>
<body>
<div class="container">

<h1>TikTok (TT) <span class="accent">eKYC Analysis</span></h1>
<p class="subtitle">Q2 2025 (Apr - Jun) &middot; Document & Face Verification &middot; Generated 2026-04-14</p>

<!-- KPI Cards -->
<div class="grid">
  <div class="card">
    <div class="card-label">Document Verifications</div>
    <div class="card-value">2,026,842</div>
    <div class="card-sub">Apr - Jun 2025</div>
  </div>
  <div class="card">
    <div class="card-label">Face Verifications</div>
    <div class="card-value">843,765</div>
    <div class="card-sub">Apr - Jun 2025</div>
  </div>
  <div class="card">
    <div class="card-label">Overall Doc Pass Rate</div>
    <div class="card-value" style="color:#22c55e">94.62%</div>
    <div class="card-sub">1,917,809 passed</div>
  </div>
  <div class="card">
    <div class="card-label">Forgery Detected</div>
    <div class="card-value" style="color:#ef4444">109,012</div>
    <div class="card-sub">5.38% of all doc checks</div>
  </div>
</div>

<!-- KEY INSIGHTS -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Key Insights</div>
  <div class="insight-box">
    <h3>&#9888; Forgery Rate Rising</h3>
    <p>Document forgery rate increased from <strong>3.56%</strong> in April to <strong>6.73%</strong> in June (nearly doubled). IDN saw the largest spike: from 2.72% to 9.23% forgery rate. This trend warrants investigation into potential fraud rings or new attack patterns.</p>
  </div>
  <div class="insight-box">
    <h3>&#128200; Volume Growth</h3>
    <p>Total document verification volume grew <strong>37.7%</strong> from April (560K) to June (772K). IDN volume surged 2.6x (146K → 386K). THA remains the dominant market at 1.21M total verifications.</p>
  </div>
  <div class="insight-box">
    <h3>&#128100; Face Verification Gaps</h3>
    <p>Face verification pass rates are significantly lower than document-only: THA at <strong>76.3%</strong> (vs 96.4% doc), PHL at <strong>84.3%</strong> (vs 87.7% doc). The face liveness failure rate is low (~1.5-2%), meaning most failures come from face-document matching, not liveness attacks.</p>
  </div>
</div>

<!-- DOCUMENT VERIFICATION BY REGION -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Document Verification by Region</div>
  <table>
    <thead>
      <tr><th>Region</th><th>Customer ID</th><th class="num">Volume</th><th class="num">Passed</th><th class="num">Failed</th><th class="num">Pass Rate</th><th class="num">Forgery Fail</th><th class="num">Forgery %</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="badge badge-blue">THA</span></td><td>9929123016</td>
        <td class="num">1,214,487</td><td class="num">1,170,241</td><td class="num">44,246</td>
        <td class="num"><span class="badge badge-green">96.36%</span></td>
        <td class="num">44,237</td><td class="num">3.64%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">IDN</span></td><td>45296432</td>
        <td class="num">699,682</td><td class="num">648,733</td><td class="num">50,949</td>
        <td class="num"><span class="badge badge-green">92.72%</span></td>
        <td class="num">50,949</td><td class="num">7.28%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">PHL</span></td><td>9929122948</td>
        <td class="num">112,654</td><td class="num">98,835</td><td class="num">13,819</td>
        <td class="num"><span class="badge badge-yellow">87.73%</span></td>
        <td class="num">13,811</td><td class="num">12.26%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">MYS</span></td><td>9929123194</td>
        <td class="num">19</td><td class="num">4</td><td class="num">15</td>
        <td class="num"><span class="badge badge-red">21.05%</span></td>
        <td class="num">15</td><td class="num">78.95%</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- MONTHLY TRENDS -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Document Verification Monthly Trend (All Regions)</div>
  <table>
    <thead>
      <tr><th>Month</th><th class="num">Total</th><th class="num">Passed</th><th class="num">Pass Rate</th><th class="num">Forgery Detected</th><th class="num">Forgery Rate</th><th>Trend</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Apr 2025</td><td class="num">560,578</td><td class="num">540,627</td>
        <td class="num"><span class="badge badge-green">96.44%</span></td>
        <td class="num">19,937</td><td class="num">3.56%</td><td>—</td>
      </tr>
      <tr>
        <td>May 2025</td><td class="num">694,193</td><td class="num">657,091</td>
        <td class="num"><span class="badge badge-green">94.66%</span></td>
        <td class="num">37,102</td><td class="num">5.34%</td><td><span class="up">&#9650; +1.78pp forgery</span></td>
      </tr>
      <tr>
        <td>Jun 2025</td><td class="num">772,071</td><td class="num">720,095</td>
        <td class="num"><span class="badge badge-yellow">93.27%</span></td>
        <td class="num">51,973</td><td class="num">6.73%</td><td><span class="up">&#9650; +1.39pp forgery</span></td>
      </tr>
    </tbody>
  </table>
</div>

<!-- REGION MONTHLY DETAIL -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Document Verification by Region &times; Month</div>
  <table>
    <thead>
      <tr><th>Month</th><th>Region</th><th class="num">Volume</th><th class="num">Passed</th><th class="num">Pass Rate</th><th class="num">Forgery Fail</th><th class="num">Forgery %</th></tr>
    </thead>
    <tbody>
      <tr><td>Apr</td><td><span class="badge badge-blue">THA</span></td><td class="num">380,590</td><td class="num">369,156</td><td class="num">97.00%</td><td class="num">11,428</td><td class="num">3.00%</td></tr>
      <tr><td>Apr</td><td><span class="badge badge-blue">IDN</span></td><td class="num">145,955</td><td class="num">141,990</td><td class="num">97.28%</td><td class="num">3,965</td><td class="num">2.72%</td></tr>
      <tr><td>Apr</td><td><span class="badge badge-blue">PHL</span></td><td class="num">34,033</td><td class="num">29,481</td><td class="num">86.62%</td><td class="num">4,544</td><td class="num">13.35%</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">THA</span></td><td class="num">488,391</td><td class="num">467,339</td><td class="num">95.69%</td><td class="num">21,052</td><td class="num">4.31%</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">IDN</span></td><td class="num">167,591</td><td class="num">156,244</td><td class="num">93.23%</td><td class="num">11,347</td><td class="num">6.77%</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">PHL</span></td><td class="num">38,202</td><td class="num">33,508</td><td class="num">87.71%</td><td class="num">4,694</td><td class="num">12.29%</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">THA</span></td><td class="num">345,506</td><td class="num">333,746</td><td class="num">96.60%</td><td class="num">11,757</td><td class="num">3.40%</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">IDN</span></td><td class="num">386,136</td><td class="num">350,499</td><td class="num">90.77%</td><td class="num">35,637</td><td class="num">9.23%</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">PHL</span></td><td class="num">40,419</td><td class="num">35,846</td><td class="num">88.69%</td><td class="num">4,573</td><td class="num">11.31%</td></tr>
    </tbody>
  </table>
</div>

<!-- FORGERY DETECTION BREAKDOWN -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Forgery Detection (Front Image Model 1) by Region</div>
  <table>
    <thead>
      <tr><th>Region</th><th class="num">Total Checked</th><th class="num">SUCCESS</th><th class="num">ID_FORGERY_DETECTED</th><th class="num">Forgery Rate</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="badge badge-blue">IDN</span></td>
        <td class="num">698,547</td><td class="num">647,829</td><td class="num">50,718</td>
        <td class="num"><span class="badge badge-yellow">7.26%</span></td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">PHL</span></td>
        <td class="num">91,100</td><td class="num">79,418</td><td class="num">11,682</td>
        <td class="num"><span class="badge badge-red">12.82%</span></td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">THA</span></td>
        <td class="num">127</td><td class="num">87</td><td class="num">40</td>
        <td class="num"><span class="badge badge-red">31.50%</span></td>
      </tr>
    </tbody>
  </table>
  <p style="color:#64748b;font-size:13px;margin-top:8px">Note: THA has very few front-image forgery checks (127) - most THA verifications use a different detection pipeline. IDN and PHL heavily rely on the front-image forgery model.</p>
</div>

<!-- FACE VERIFICATION SECTION -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Face Verification (DOCUMENT_FACE) by Region</div>
  <table>
    <thead>
      <tr><th>Region</th><th class="num">Volume</th><th class="num">IDV Passed</th><th class="num">Pass Rate</th><th class="num">Liveness Pass</th><th class="num">Liveness Fail</th><th class="num">Liveness Fail %</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="badge badge-blue">THA</span></td>
        <td class="num">383,740</td><td class="num">292,940</td>
        <td class="num"><span class="badge badge-yellow">76.34%</span></td>
        <td class="num">377,986</td><td class="num">5,743</td><td class="num">1.50%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">PHL</span></td>
        <td class="num">331,522</td><td class="num">279,501</td>
        <td class="num"><span class="badge badge-yellow">84.31%</span></td>
        <td class="num">324,936</td><td class="num">6,583</td><td class="num">1.99%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">MYS</span></td>
        <td class="num">128,483</td><td class="num">116,235</td>
        <td class="num"><span class="badge badge-green">90.47%</span></td>
        <td class="num">124,766</td><td class="num">3,716</td><td class="num">2.89%</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">IDN</span></td>
        <td class="num">20</td><td class="num">0</td>
        <td class="num"><span class="badge badge-red">0.00%</span></td>
        <td class="num">16</td><td class="num">4</td><td class="num">20.00%</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- FACE MONTHLY TRENDS -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Face Verification Monthly Trends</div>
  <table>
    <thead>
      <tr><th>Month</th><th>Region</th><th class="num">Volume</th><th class="num">Passed</th><th class="num">Pass Rate</th><th class="num">Liveness Fail</th></tr>
    </thead>
    <tbody>
      <tr><td>Apr</td><td><span class="badge badge-blue">MYS</span></td><td class="num">62,621</td><td class="num">56,473</td><td class="num">90.18%</td><td class="num">1,888</td></tr>
      <tr><td>Apr</td><td><span class="badge badge-blue">PHL</span></td><td class="num">72,891</td><td class="num">60,637</td><td class="num">83.19%</td><td class="num">1,507</td></tr>
      <tr><td>Apr</td><td><span class="badge badge-blue">THA</span></td><td class="num">342</td><td class="num">229</td><td class="num">66.96%</td><td class="num">12</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">MYS</span></td><td class="num">55,918</td><td class="num">50,708</td><td class="num">90.68%</td><td class="num">1,564</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">PHL</span></td><td class="num">91,732</td><td class="num">77,018</td><td class="num">83.96%</td><td class="num">1,756</td></tr>
      <tr><td>May</td><td><span class="badge badge-blue">THA</span></td><td class="num">43,193</td><td class="num">31,970</td><td class="num">74.02%</td><td class="num">616</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">MYS</span></td><td class="num">9,944</td><td class="num">9,054</td><td class="num">91.05%</td><td class="num">264</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">PHL</span></td><td class="num">166,899</td><td class="num">141,846</td><td class="num">84.99%</td><td class="num">3,320</td></tr>
      <tr><td>Jun</td><td><span class="badge badge-blue">THA</span></td><td class="num">340,205</td><td class="num">260,741</td><td class="num">76.64%</td><td class="num">5,115</td></tr>
    </tbody>
  </table>
</div>

<!-- VOLUME VISUALIZATION -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Document Volume by Region (Proportional)</div>
  <div class="chart-row">
    <div class="chart-label">THA</div>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa)">1,214,487</div>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-label">IDN</div>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width:57.6%;background:linear-gradient(90deg,#8b5cf6,#a78bfa)">699,682</div>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-label">PHL</div>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width:9.3%;background:linear-gradient(90deg,#f59e0b,#fbbf24)">112,654</div>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-label">MYS</div>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width:0.5%;min-width:40px;background:linear-gradient(90deg,#ef4444,#f87171)">19</div>
    </div>
  </div>
</div>

<!-- SUMMARY -->
<div class="section">
  <div class="section-title"><span class="dot"></span>Summary &amp; Recommendations</div>
  <div class="card" style="border-color:#3b82f6">
    <p style="font-size:14px;color:#cbd5e1;line-height:1.8">
      <strong style="color:#fff">1. Forgery rate is trending up significantly</strong> — from 3.56% (Apr) to 6.73% (Jun). IDN is the primary driver with a 3.4x increase in forgery detections (3,965 → 35,637). Action: investigate whether this is driven by new fraud patterns or model sensitivity changes.<br><br>
      <strong style="color:#fff">2. IDN volume surge</strong> — IDN verification volume grew 2.6x in Q2. Combined with rising forgery rates, this region needs focused attention. Consider tightening thresholds or adding manual review for high-risk cases.<br><br>
      <strong style="color:#fff">3. PHL has consistently high forgery rates</strong> — averaging ~12% across all 3 months. This is the highest stable rate among active regions. Review the PHL document types and consider region-specific model tuning.<br><br>
      <strong style="color:#fff">4. Face verification gap</strong> — The gap between document-only pass rate (~94.6%) and document+face pass rate (~80.9%) suggests ~14% of users pass document checks but fail at face matching. This could indicate identity borrowing or document sharing.<br><br>
      <strong style="color:#fff">5. THA face verification ramp-up</strong> — THA went from 342 face checks in Apr to 340,205 in Jun, indicating a recent rollout. The 76.3% pass rate may stabilize as the system matures.<br><br>
      <strong style="color:#fff">6. MYS winding down</strong> — MYS face verification dropped from 62K (Apr) to 10K (Jun), while document checks remain minimal (19 total). This suggests MYS may be transitioning away or in a testing phase.
    </p>
  </div>
</div>

<div class="footer">
  TikTok (TT) eKYC Analysis &middot; Data: q2_tt_document_annotation_data &amp; q2_tt_document_face_annotation_data &middot; Period: Q2 2025
</div>

</div>
</body>
</html>"""

import requests

payload = {
    "channel": "oss_html",
    "oss_key": "reports/tt-ekyc-q2-analysis/index.html",
    "html": html
}

resp = requests.post(
    f"{VT_BASE}/api/tools/deliver",
    headers={
        "Authorization": f"Bearer {VT_TOKEN}",
        "Content-Type": "application/json"
    },
    json=payload,
    timeout=30
)
print(resp.status_code)
print(resp.text)
