#!/usr/bin/env python3
"""Generate and publish Bybit Liveness Weekly Report."""

import json
import subprocess

DATA = [
    {"real_region": "NGA", "total_pv": 10650, "passed": 7838, "pass_rate": 73.6},
    {"real_region": "RUS", "total_pv": 7094, "passed": 5793, "pass_rate": 81.66},
    {"real_region": "EGY", "total_pv": 3755, "passed": 3118, "pass_rate": 83.04},
    {"real_region": "CHN", "total_pv": 2130, "passed": 1864, "pass_rate": 87.51},
    {"real_region": "BRA", "total_pv": 1858, "passed": 1602, "pass_rate": 86.22},
    {"real_region": "UKR", "total_pv": 1824, "passed": 1537, "pass_rate": 84.27},
    {"real_region": "IDN", "total_pv": 1252, "passed": 1022, "pass_rate": 81.63},
    {"real_region": "BGD", "total_pv": 883, "passed": 718, "pass_rate": 81.31},
    {"real_region": "KEN", "total_pv": 489, "passed": 414, "pass_rate": 84.66},
    {"real_region": "VEN", "total_pv": 412, "passed": 374, "pass_rate": 90.78},
    {"real_region": "PAK", "total_pv": 353, "passed": 312, "pass_rate": 88.39},
    {"real_region": "VNM", "total_pv": 240, "passed": 225, "pass_rate": 93.75},
    {"real_region": "MEX", "total_pv": 159, "passed": 149, "pass_rate": 93.71},
    {"real_region": "AGO", "total_pv": 128, "passed": 119, "pass_rate": 92.97},
    {"real_region": "ETH", "total_pv": 66, "passed": 49, "pass_rate": 74.24},
    {"real_region": "UZB", "total_pv": 55, "passed": 55, "pass_rate": 100.0},
    {"real_region": "ZAF", "total_pv": 36, "passed": 31, "pass_rate": 86.11},
    {"real_region": "TJK", "total_pv": 35, "passed": 34, "pass_rate": 97.14},
    {"real_region": "LKA", "total_pv": 35, "passed": 32, "pass_rate": 91.43},
    {"real_region": "COL", "total_pv": 35, "passed": 33, "pass_rate": 94.29},
    {"real_region": "PER", "total_pv": 33, "passed": 29, "pass_rate": 87.88},
    {"real_region": "IND", "total_pv": 27, "passed": 27, "pass_rate": 100.0},
    {"real_region": "DZA", "total_pv": 21, "passed": 20, "pass_rate": 95.24},
    {"real_region": "TUN", "total_pv": 20, "passed": 17, "pass_rate": 85.0},
    {"real_region": "MAR", "total_pv": 20, "passed": 20, "pass_rate": 100.0},
    {"real_region": "TUR", "total_pv": 20, "passed": 19, "pass_rate": 95.0},
    {"real_region": "ARG", "total_pv": 18, "passed": 13, "pass_rate": 72.22},
    {"real_region": "BLR", "total_pv": 18, "passed": 8, "pass_rate": 44.44},
    {"real_region": "RWA", "total_pv": 18, "passed": 17, "pass_rate": 94.44},
    {"real_region": "TKM", "total_pv": 18, "passed": 15, "pass_rate": 83.33},
    {"real_region": "SDN", "total_pv": 16, "passed": 15, "pass_rate": 93.75},
    {"real_region": "TWN", "total_pv": 14, "passed": 0, "pass_rate": 0.0},
    {"real_region": "KGZ", "total_pv": 12, "passed": 12, "pass_rate": 100.0},
    {"real_region": "ZWE", "total_pv": 9, "passed": 9, "pass_rate": 100.0},
    {"real_region": "CUB", "total_pv": 9, "passed": 9, "pass_rate": 100.0},
    {"real_region": "AZE", "total_pv": 9, "passed": 9, "pass_rate": 100.0},
    {"real_region": "CHL", "total_pv": 9, "passed": 7, "pass_rate": 77.78},
    {"real_region": "YEM", "total_pv": 7, "passed": 1, "pass_rate": 14.29},
    {"real_region": "KAZ", "total_pv": 6, "passed": 6, "pass_rate": 100.0},
    {"real_region": "CMR", "total_pv": 5, "passed": 5, "pass_rate": 100.0},
    {"real_region": "SYR", "total_pv": 5, "passed": 5, "pass_rate": 100.0},
    {"real_region": "JOR", "total_pv": 5, "passed": 5, "pass_rate": 100.0},
    {"real_region": "ECU", "total_pv": 5, "passed": 5, "pass_rate": 100.0},
    {"real_region": "USA", "total_pv": 4, "passed": 1, "pass_rate": 25.0},
    {"real_region": "MDA", "total_pv": 4, "passed": 4, "pass_rate": 100.0},
    {"real_region": "THA", "total_pv": 4, "passed": 1, "pass_rate": 25.0},
    {"real_region": "AFG", "total_pv": 4, "passed": 3, "pass_rate": 75.0},
    {"real_region": "BDI", "total_pv": 3, "passed": 3, "pass_rate": 100.0},
    {"real_region": "PSE", "total_pv": 3, "passed": 3, "pass_rate": 100.0},
    {"real_region": "AUS", "total_pv": 3, "passed": 1, "pass_rate": 33.33},
]

total_volume = sum(r["total_pv"] for r in DATA)
total_passed = sum(r["passed"] for r in DATA)
overall_rate = round(total_passed * 100.0 / total_volume, 2)
countries_below_80 = [r for r in DATA if r["pass_rate"] < 80]
top5 = DATA[:5]


def build_table_rows():
    rows = ""
    for i, r in enumerate(DATA):
        rate = r["pass_rate"]
        is_low = rate < 80
        row_bg = "#1e1b2e" if i % 2 == 0 else "#151229"
        rate_color = "#ef4444" if is_low else "#4ade80"
        rate_badge_bg = "rgba(239,68,68,0.15)" if is_low else "rgba(74,222,128,0.12)"
        flag = " &#9888;" if is_low else ""
        rows += f"""<tr style="background:{row_bg};">
  <td style="padding:10px 16px;border-bottom:1px solid #1e293b;">{i+1}</td>
  <td style="padding:10px 16px;border-bottom:1px solid #1e293b;font-weight:600;">{r['real_region']}</td>
  <td style="padding:10px 16px;border-bottom:1px solid #1e293b;text-align:right;">{r['total_pv']:,}</td>
  <td style="padding:10px 16px;border-bottom:1px solid #1e293b;text-align:right;">{r['passed']:,}</td>
  <td style="padding:10px 16px;border-bottom:1px solid #1e293b;text-align:right;">
    <span style="background:{rate_badge_bg};color:{rate_color};padding:3px 10px;border-radius:12px;font-weight:700;">{rate}%{flag}</span>
  </td>
</tr>
"""
    return rows


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bybit Liveness Weekly Report (Apr 7-11, 2026)</title>
</head>
<body style="margin:0;padding:0;background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;line-height:1.6;">

<div style="max-width:960px;margin:0 auto;padding:32px 20px;">

  <!-- Header -->
  <div style="text-align:center;margin-bottom:40px;">
    <h1 style="margin:0 0 8px;font-size:28px;color:#ffffff;">Bybit Liveness Weekly Report</h1>
    <p style="margin:0;color:#94a3b8;font-size:16px;">Apr 7 &ndash; 11, 2026 &nbsp;&middot;&nbsp; Week 15</p>
    <div style="width:60px;height:3px;background:#3b82f6;margin:16px auto 0;border-radius:2px;"></div>
  </div>

  <!-- Summary Cards -->
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:32px;">
    <div style="flex:1;min-width:200px;background:#1e293b;border-radius:12px;padding:20px 24px;border:1px solid #334155;">
      <div style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Total Volume</div>
      <div style="font-size:32px;font-weight:700;color:#ffffff;">{total_volume:,}</div>
    </div>
    <div style="flex:1;min-width:200px;background:#1e293b;border-radius:12px;padding:20px 24px;border:1px solid #334155;">
      <div style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Total Passed</div>
      <div style="font-size:32px;font-weight:700;color:#4ade80;">{total_passed:,}</div>
    </div>
    <div style="flex:1;min-width:200px;background:#1e293b;border-radius:12px;padding:20px 24px;border:1px solid #334155;">
      <div style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Overall Pass Rate</div>
      <div style="font-size:32px;font-weight:700;color:#3b82f6;">{overall_rate}%</div>
    </div>
    <div style="flex:1;min-width:200px;background:#1e293b;border-radius:12px;padding:20px 24px;border:1px solid {'#ef4444' if len(countries_below_80) > 0 else '#334155'};">
      <div style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Below 80% Regions</div>
      <div style="font-size:32px;font-weight:700;color:#ef4444;">{len(countries_below_80)}</div>
    </div>
  </div>

  <!-- Alerts -->
  <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:10px;padding:16px 20px;margin-bottom:32px;">
    <div style="font-weight:700;color:#ef4444;margin-bottom:8px;">&#9888; Low Pass Rate Regions (below 80%)</div>
    <div style="color:#fca5a5;font-size:14px;">
      {', '.join(f"<strong>{r['real_region']}</strong> ({r['pass_rate']}%)" for r in sorted(countries_below_80, key=lambda x: x['pass_rate']))}
    </div>
  </div>

  <!-- Table -->
  <div style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;margin-bottom:32px;">
    <div style="padding:16px 20px;border-bottom:1px solid #334155;">
      <h2 style="margin:0;font-size:18px;color:#ffffff;">Liveness Detection by Country</h2>
      <p style="margin:4px 0 0;font-size:13px;color:#64748b;">50 countries &middot; Sorted by volume</p>
    </div>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#0f172a;">
            <th style="padding:12px 16px;text-align:left;color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #334155;">#</th>
            <th style="padding:12px 16px;text-align:left;color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #334155;">Country</th>
            <th style="padding:12px 16px;text-align:right;color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #334155;">Volume</th>
            <th style="padding:12px 16px;text-align:right;color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #334155;">Passed</th>
            <th style="padding:12px 16px;text-align:right;color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #334155;">Pass Rate</th>
          </tr>
        </thead>
        <tbody>
{build_table_rows()}
        </tbody>
        <tfoot>
          <tr style="background:#0f172a;font-weight:700;">
            <td colspan="2" style="padding:12px 16px;border-top:2px solid #334155;color:#ffffff;">TOTAL</td>
            <td style="padding:12px 16px;border-top:2px solid #334155;text-align:right;color:#ffffff;">{total_volume:,}</td>
            <td style="padding:12px 16px;border-top:2px solid #334155;text-align:right;color:#ffffff;">{total_passed:,}</td>
            <td style="padding:12px 16px;border-top:2px solid #334155;text-align:right;">
              <span style="background:rgba(59,130,246,0.15);color:#3b82f6;padding:3px 10px;border-radius:12px;font-weight:700;">{overall_rate}%</span>
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <div style="text-align:center;color:#475569;font-size:12px;padding-top:16px;border-top:1px solid #1e293b;">
    Generated by Cursor Cloud Agent &middot; Data source: adv_guardian_data_core &middot; Customer ID: 9929123352
  </div>

</div>
</body>
</html>"""

print(f"HTML generated: {len(html)} bytes")
print(f"Summary: {total_volume:,} total volume, {total_passed:,} passed, {overall_rate}% overall rate")
print(f"Countries below 80%: {len(countries_below_80)}")

VT_TOKEN = "vt_sk_21089b3935576ce6c994c3aa2f14ad0d160b8eeafc583bd0b7583e520ecba2af"
VT_BASE = "https://vibe-track.ngrok.app"

payload = {
    "channel": "oss_html",
    "oss_key": "ekyc-liveness/bybit/weekly-w15-cursor-test/index.html",
    "html": html,
}

import urllib.request

req = urllib.request.Request(
    f"{VT_BASE}/api/tools/deliver",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {VT_TOKEN}",
        "Content-Type": "application/json",
    },
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
        print(f"\nPublish result: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"Error publishing: {e}")
