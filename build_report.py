#!/usr/bin/env python3
"""Build a self-contained HTML report for Bybit Liveness W15 2026."""
import json

with open('/workspace/analysis_output.json') as f:
    data = json.load(f)

w15 = data['w15_overall']
w14 = data['w14_overall']
delta_pp = data['delta_pp']
country_stats = data['country_stats']
alerts = data['alerts']

# Top countries by volume (min 10 sessions in W15)
significant = [c for c in country_stats if c['w15_total'] >= 10]
top_by_volume = sorted(significant, key=lambda x: x['w15_total'], reverse=True)

# Daily data for chart
daily_w15 = [
    {"date": "Apr 7", "rate": 95.75, "vol": 5929},
    {"date": "Apr 8", "rate": 95.21, "vol": 6004},
    {"date": "Apr 9", "rate": 95.66, "vol": 6391},
    {"date": "Apr 10", "rate": 96.00, "vol": 6531},
    {"date": "Apr 11", "rate": 95.33, "vol": 6921},
    {"date": "Apr 12", "rate": 95.39, "vol": 6729},
    {"date": "Apr 13", "rate": 95.69, "vol": 6424},
]
daily_w14 = [
    {"date": "Mar 31", "rate": 96.93, "vol": 6666},
    {"date": "Apr 1", "rate": 95.62, "vol": 7707},
    {"date": "Apr 2", "rate": 95.98, "vol": 7144},
    {"date": "Apr 3", "rate": 96.23, "vol": 6669},
    {"date": "Apr 4", "rate": 96.02, "vol": 5866},
    {"date": "Apr 5", "rate": 95.44, "vol": 5672},
    {"date": "Apr 6", "rate": 95.42, "vol": 5935},
]

# Build country table rows
def build_country_rows():
    rows = []
    for c in top_by_volume[:25]:
        w15_pr = f"{c['w15_pass_rate']:.1f}%" if c['w15_pass_rate'] is not None else "N/A"
        w14_pr = f"{c['w14_pass_rate']:.1f}%" if c['w14_pass_rate'] is not None else "N/A"
        
        if c['delta_pp'] is not None:
            delta = c['delta_pp']
            if delta < -2:
                delta_class = "alert-drop"
                delta_str = f"{delta:+.1f}pp"
            elif delta < 0:
                delta_class = "minor-drop"
                delta_str = f"{delta:+.1f}pp"
            elif delta > 2:
                delta_class = "good-rise"
                delta_str = f"{delta:+.1f}pp"
            else:
                delta_class = "neutral"
                delta_str = f"{delta:+.1f}pp"
        else:
            delta_class = "neutral"
            delta_str = "NEW"
        
        rows.append(f"""<tr>
            <td>{c['region']}</td>
            <td class="num">{c['w15_total']:,}</td>
            <td class="num">{c['w15_pass']:,}</td>
            <td class="num">{c['w15_fail']:,}</td>
            <td class="num">{w15_pr}</td>
            <td class="num">{w14_pr}</td>
            <td class="num {delta_class}">{delta_str}</td>
        </tr>""")
    return "\n".join(rows)

# Build alert rows
def build_alert_rows():
    if not alerts:
        return '<tr><td colspan="5" style="text-align:center;color:#94a3b8;">No countries with >2pp drop detected</td></tr>'
    rows = []
    sorted_alerts = sorted(alerts, key=lambda x: x['delta_pp'])
    for a in sorted_alerts:
        rows.append(f"""<tr>
            <td><span class="alert-badge">{a['region']}</span></td>
            <td class="num">{a['w15_total']:,}</td>
            <td class="num">{a['w15_pass_rate']:.1f}%</td>
            <td class="num">{a['w14_pass_rate']:.1f}%</td>
            <td class="num alert-drop">{a['delta_pp']:+.1f}pp</td>
        </tr>""")
    return "\n".join(rows)

# Build bar chart data for top 10 countries
top10 = top_by_volume[:10]
bar_labels = json.dumps([c['region'] for c in top10])
bar_w15 = json.dumps([c['w15_pass_rate'] if c['w15_pass_rate'] else 0 for c in top10])
bar_w14 = json.dumps([c['w14_pass_rate'] if c['w14_pass_rate'] else 0 for c in top10])

# Daily chart data
daily_labels_w15 = json.dumps([d['date'] for d in daily_w15])
daily_rates_w15 = json.dumps([d['rate'] for d in daily_w15])
daily_labels_w14 = json.dumps([d['date'] for d in daily_w14])
daily_rates_w14 = json.dumps([d['rate'] for d in daily_w14])

# Volume data for daily
daily_vols_w15 = json.dumps([d['vol'] for d in daily_w15])
daily_vols_w14 = json.dumps([d['vol'] for d in daily_w14])

# SQL used
sql_query = """SELECT region, liveness_result1, COUNT(*) as cnt 
FROM idv_aai_liveness_details 
WHERE customer_id = 9929123352 
  AND pt >= '20260407' AND pt <= '20260413'
GROUP BY region, liveness_result1 
ORDER BY region, liveness_result1"""

delta_color = "#ef4444" if delta_pp < 0 else "#22c55e"
delta_icon = "↓" if delta_pp < 0 else "↑"

vol_delta = w15['total'] - w14['total']
vol_delta_pct = vol_delta / w14['total'] * 100

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bybit Liveness Weekly Report — W15 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    --accent-light: #60a5fa;
    --border: #334155;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
  
  /* Header */
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), #8b5cf6, var(--accent));
  }}
  .header-tag {{
    display: inline-block;
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-light);
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .header .subtitle {{
    color: var(--text-secondary);
    font-size: 15px;
  }}
  .header .meta {{
    margin-top: 12px;
    color: var(--text-muted);
    font-size: 13px;
  }}
  
  /* KPI Cards */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }}
  .kpi-label {{
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
  }}
  .kpi-delta {{
    font-size: 14px;
    margin-top: 6px;
    font-weight: 500;
  }}
  .kpi-detail {{
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }}
  
  /* Alert Box */
  .alert-box {{
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .alert-box h3 {{
    color: var(--danger);
    font-size: 15px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .alert-box h3::before {{
    content: '⚠';
    font-size: 18px;
  }}
  
  /* Section */
  .section {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .section h2 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section h2 .icon {{ font-size: 20px; }}
  
  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  thead th {{
    text-align: left;
    padding: 10px 12px;
    background: rgba(59, 130, 246, 0.08);
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
  }}
  tbody td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.5);
    color: var(--text-primary);
  }}
  tbody tr:hover {{
    background: rgba(59, 130, 246, 0.04);
  }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .alert-drop {{ color: var(--danger); font-weight: 600; }}
  .minor-drop {{ color: #fb923c; }}
  .good-rise {{ color: var(--success); font-weight: 600; }}
  .neutral {{ color: var(--text-secondary); }}
  .alert-badge {{
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    color: var(--danger);
    padding: 2px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
  }}
  
  /* Charts */
  .chart-container {{
    position: relative;
    height: 320px;
    margin-top: 8px;
  }}
  .charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }}
  @media (max-width: 768px) {{
    .charts-row {{ grid-template-columns: 1fr; }}
  }}
  
  /* Collapsible SQL */
  details {{
    margin-top: 16px;
  }}
  details summary {{
    cursor: pointer;
    color: var(--text-muted);
    font-size: 13px;
    padding: 8px 0;
  }}
  details summary:hover {{
    color: var(--accent-light);
  }}
  pre {{
    background: #0c1222;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    font-size: 12px;
    color: #93c5fd;
    line-height: 1.5;
    margin-top: 8px;
  }}
  
  /* Footer */
  .footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    padding: 24px 0 16px;
    border-top: 1px solid var(--border);
    margin-top: 16px;
  }}
</style>
</head>
<body>
<div class="container">
  
  <!-- Header -->
  <div class="header">
    <div class="header-tag">Weekly Report</div>
    <h1>Bybit Liveness — W15 2026</h1>
    <div class="subtitle">April 7 – 13, 2026 &nbsp;|&nbsp; Compared with W14 (Mar 31 – Apr 6)</div>
    <div class="meta">Source: idv_aai_liveness_details &nbsp;·&nbsp; Customer ID: 9929123352 &nbsp;·&nbsp; Generated: Apr 15, 2026</div>
  </div>
  
  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">W15 Pass Rate</div>
      <div class="kpi-value" style="color: var(--accent-light);">{w15['pass_rate']:.1f}%</div>
      <div class="kpi-delta" style="color: {delta_color};">{delta_icon} {abs(delta_pp):.2f}pp vs W14</div>
      <div class="kpi-detail">W14: {w14['pass_rate']:.1f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Sessions</div>
      <div class="kpi-value">{w15['total']:,}</div>
      <div class="kpi-delta" style="color: {'#ef4444' if vol_delta < 0 else '#22c55e'};">{vol_delta:+,} ({vol_delta_pct:+.1f}%)</div>
      <div class="kpi-detail">W14: {w14['total']:,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Pass / Fail</div>
      <div class="kpi-value" style="font-size: 24px;">{w15['pass']:,} <span style="color:var(--text-muted);font-size:16px;">/</span> <span style="color:var(--danger);font-size:22px;">{w15['fail']:,}</span></div>
      <div class="kpi-detail">Null/Pending: {w15['null']:,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Countries w/ &gt;2pp Drop</div>
      <div class="kpi-value" style="color: {'var(--danger)' if len(alerts) > 0 else 'var(--success)'};">{len(alerts)}</div>
      <div class="kpi-detail">{'⚠ Requires attention' if len(alerts) > 0 else '✓ All countries stable'}</div>
    </div>
  </div>
  
  <!-- Alert Section -->
  {'<div class="alert-box"><h3>Countries with >2pp Pass Rate Drop (min 10 sessions)</h3>' + """
  <table>
    <thead><tr><th>Country</th><th>W15 Volume</th><th>W15 Rate</th><th>W14 Rate</th><th>Delta</th></tr></thead>
    <tbody>""" + build_alert_rows() + """</tbody>
  </table></div>""" if alerts else ''}
  
  <!-- Charts Row -->
  <div class="charts-row">
    <div class="section">
      <h2><span class="icon">📈</span> Daily Pass Rate Trend</h2>
      <div class="chart-container">
        <canvas id="dailyChart"></canvas>
      </div>
    </div>
    <div class="section">
      <h2><span class="icon">📊</span> Top 10 Countries — Pass Rate</h2>
      <div class="chart-container">
        <canvas id="countryChart"></canvas>
      </div>
    </div>
  </div>
  
  <!-- Country Detail Table -->
  <div class="section">
    <h2><span class="icon">🌍</span> Country Breakdown (Top 25 by Volume)</h2>
    <table>
      <thead>
        <tr>
          <th>Country</th>
          <th style="text-align:right;">W15 Total</th>
          <th style="text-align:right;">Pass</th>
          <th style="text-align:right;">Fail</th>
          <th style="text-align:right;">W15 Rate</th>
          <th style="text-align:right;">W14 Rate</th>
          <th style="text-align:right;">Delta</th>
        </tr>
      </thead>
      <tbody>
        {build_country_rows()}
      </tbody>
    </table>
  </div>
  
  <!-- SQL Query -->
  <div class="section">
    <details>
      <summary>📋 View SQL Query Used</summary>
      <pre>{sql_query}</pre>
    </details>
  </div>
  
  <div class="footer">
    Bybit Liveness Weekly Report · Generated by Vibe Work Agent · Data from ODPS idv_aai_liveness_details
  </div>
</div>

<script>
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(51, 65, 85, 0.5)';

// Daily Pass Rate Chart
const dailyCtx = document.getElementById('dailyChart').getContext('2d');
new Chart(dailyCtx, {{
  type: 'line',
  data: {{
    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
    datasets: [{{
      label: 'W15 (Apr 7-13)',
      data: {daily_rates_w15},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      borderWidth: 2,
      fill: true,
      tension: 0.3,
      pointRadius: 4,
      pointBackgroundColor: '#3b82f6',
    }}, {{
      label: 'W14 (Mar 31-Apr 6)',
      data: {daily_rates_w14},
      borderColor: '#64748b',
      backgroundColor: 'rgba(100, 116, 139, 0.05)',
      borderWidth: 2,
      borderDash: [5, 5],
      fill: false,
      tension: 0.3,
      pointRadius: 3,
      pointBackgroundColor: '#64748b',
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ padding: 16, usePointStyle: true, pointStyle: 'circle' }} }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{ return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + '%'; }}
        }}
      }}
    }},
    scales: {{
      y: {{
        min: 94,
        max: 98,
        ticks: {{ callback: function(v) {{ return v + '%'; }} }},
        grid: {{ color: 'rgba(51, 65, 85, 0.3)' }}
      }},
      x: {{
        grid: {{ display: false }}
      }}
    }}
  }}
}});

// Country Bar Chart
const countryCtx = document.getElementById('countryChart').getContext('2d');
new Chart(countryCtx, {{
  type: 'bar',
  data: {{
    labels: {bar_labels},
    datasets: [{{
      label: 'W15',
      data: {bar_w15},
      backgroundColor: 'rgba(59, 130, 246, 0.7)',
      borderColor: '#3b82f6',
      borderWidth: 1,
      borderRadius: 4,
    }}, {{
      label: 'W14',
      data: {bar_w14},
      backgroundColor: 'rgba(100, 116, 139, 0.4)',
      borderColor: '#64748b',
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ padding: 16, usePointStyle: true, pointStyle: 'rect' }} }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{ return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'; }}
        }}
      }}
    }},
    scales: {{
      y: {{
        min: 80,
        max: 100,
        ticks: {{ callback: function(v) {{ return v + '%'; }} }},
        grid: {{ color: 'rgba(51, 65, 85, 0.3)' }}
      }},
      x: {{
        grid: {{ display: false }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

with open('/workspace/report.html', 'w') as f:
    f.write(html)

print(f"Report generated: /workspace/report.html")
print(f"Size: {len(html):,} bytes")
