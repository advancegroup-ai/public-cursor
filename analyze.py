#!/usr/bin/env python3
import json

w15_data = [{"region":"AGO","result":"null","cnt":1},{"region":"AGO","result":"fail","cnt":9},{"region":"AGO","result":"pass","cnt":185},{"region":"ARE","result":"fail","cnt":1},{"region":"ARE","result":"pass","cnt":1},{"region":"ARG","result":"null","cnt":1},{"region":"ARG","result":"fail","cnt":1},{"region":"ARG","result":"pass","cnt":16},{"region":"AUS","result":"pass","cnt":3},{"region":"AZE","result":"pass","cnt":7},{"region":"BDI","result":"null","cnt":1},{"region":"BDI","result":"pass","cnt":3},{"region":"BEL","result":"pass","cnt":1},{"region":"BGD","result":"null","cnt":5},{"region":"BGD","result":"fail","cnt":48},{"region":"BGD","result":"pass","cnt":1215},{"region":"BLR","result":"pass","cnt":3},{"region":"BRA","result":"null","cnt":283},{"region":"BRA","result":"fail","cnt":48},{"region":"BRA","result":"pass","cnt":2170},{"region":"CAN","result":"pass","cnt":1},{"region":"CHL","result":"pass","cnt":11},{"region":"CHN","result":"null","cnt":14},{"region":"CHN","result":"fail","cnt":386},{"region":"CHN","result":"pass","cnt":2077},{"region":"COL","result":"fail","cnt":2},{"region":"COL","result":"pass","cnt":41},{"region":"CYP","result":"pass","cnt":1},{"region":"CZE","result":"pass","cnt":1},{"region":"DZA","result":"null","cnt":1},{"region":"DZA","result":"pass","cnt":20},{"region":"EGY","result":"null","cnt":25},{"region":"EGY","result":"fail","cnt":148},{"region":"EGY","result":"pass","cnt":5025},{"region":"ETH","result":"null","cnt":1},{"region":"ETH","result":"fail","cnt":2},{"region":"ETH","result":"pass","cnt":87},{"region":"GEO","result":"pass","cnt":1},{"region":"GHA","result":"fail","cnt":1},{"region":"HUN","result":"pass","cnt":1},{"region":"IDN","result":"null","cnt":200},{"region":"IDN","result":"fail","cnt":51},{"region":"IDN","result":"pass","cnt":1453},{"region":"IND","result":"fail","cnt":1},{"region":"IND","result":"pass","cnt":13},{"region":"IRQ","result":"fail","cnt":1},{"region":"IRQ","result":"pass","cnt":1},{"region":"JOR","result":"pass","cnt":1},{"region":"KAZ","result":"pass","cnt":1},{"region":"KEN","result":"null","cnt":1},{"region":"KEN","result":"fail","cnt":18},{"region":"KEN","result":"pass","cnt":661},{"region":"LBY","result":"pass","cnt":1},{"region":"LKA","result":"fail","cnt":3},{"region":"LKA","result":"pass","cnt":47},{"region":"MAR","result":"fail","cnt":1},{"region":"MAR","result":"pass","cnt":19},{"region":"MDA","result":"pass","cnt":2},{"region":"MEX","result":"fail","cnt":4},{"region":"MEX","result":"pass","cnt":204},{"region":"MOZ","result":"pass","cnt":1},{"region":"MWI","result":"pass","cnt":1},{"region":"MYS","result":"pass","cnt":3},{"region":"NGA","result":"null","cnt":3528},{"region":"NGA","result":"fail","cnt":387},{"region":"NGA","result":"pass","cnt":11617},{"region":"NIC","result":"pass","cnt":1},{"region":"NPL","result":"pass","cnt":1},{"region":"PAK","result":"null","cnt":3},{"region":"PAK","result":"fail","cnt":10},{"region":"PAK","result":"pass","cnt":462},{"region":"PER","result":"fail","cnt":4},{"region":"PER","result":"pass","cnt":38},{"region":"POL","result":"pass","cnt":3},{"region":"PSE","result":"pass","cnt":3},{"region":"RUS","result":"null","cnt":93},{"region":"RUS","result":"fail","cnt":526},{"region":"RUS","result":"pass","cnt":10050},{"region":"RWA","result":"fail","cnt":1},{"region":"RWA","result":"pass","cnt":19},{"region":"SAU","result":"pass","cnt":1},{"region":"SDN","result":"fail","cnt":1},{"region":"SDN","result":"pass","cnt":20},{"region":"SWZ","result":"pass","cnt":1},{"region":"SYR","result":"pass","cnt":2},{"region":"TJK","result":"pass","cnt":1},{"region":"TUN","result":"fail","cnt":2},{"region":"TUN","result":"pass","cnt":23},{"region":"TUR","result":"fail","cnt":2},{"region":"TUR","result":"pass","cnt":24},{"region":"TWN","result":"fail","cnt":1},{"region":"TWN","result":"pass","cnt":17},{"region":"UKR","result":"null","cnt":18},{"region":"UKR","result":"fail","cnt":125},{"region":"UKR","result":"pass","cnt":2508},{"region":"URY","result":"pass","cnt":1},{"region":"USA","result":"fail","cnt":3},{"region":"USA","result":"pass","cnt":4},{"region":"VEN","result":"fail","cnt":8},{"region":"VEN","result":"pass","cnt":534},{"region":"VNM","result":"null","cnt":1},{"region":"VNM","result":"fail","cnt":8},{"region":"VNM","result":"pass","cnt":276},{"region":"YEM","result":"pass","cnt":7},{"region":"ZAF","result":"pass","cnt":48},{"region":"ZMB","result":"pass","cnt":3},{"region":"ZWE","result":"pass","cnt":8}]

w14_data = [{"region":"AGO","result":"fail","cnt":15},{"region":"AGO","result":"pass","cnt":187},{"region":"ARE","result":"pass","cnt":1},{"region":"ARG","result":"pass","cnt":18},{"region":"ARM","result":"pass","cnt":6},{"region":"AUS","result":"pass","cnt":1},{"region":"BEN","result":"pass","cnt":1},{"region":"BFA","result":"pass","cnt":4},{"region":"BGD","result":"null","cnt":6},{"region":"BGD","result":"fail","cnt":36},{"region":"BGD","result":"pass","cnt":1065},{"region":"BLR","result":"pass","cnt":4},{"region":"BOL","result":"pass","cnt":1},{"region":"BRA","result":"null","cnt":187},{"region":"BRA","result":"fail","cnt":54},{"region":"BRA","result":"pass","cnt":1307},{"region":"CHL","result":"pass","cnt":1},{"region":"CHN","result":"null","cnt":2},{"region":"CHN","result":"fail","cnt":179},{"region":"CHN","result":"pass","cnt":1213},{"region":"COL","result":"fail","cnt":2},{"region":"COL","result":"pass","cnt":69},{"region":"DZA","result":"pass","cnt":32},{"region":"ECU","result":"pass","cnt":2},{"region":"EGY","result":"null","cnt":37},{"region":"EGY","result":"fail","cnt":218},{"region":"EGY","result":"pass","cnt":5370},{"region":"ETH","result":"null","cnt":1},{"region":"ETH","result":"fail","cnt":1},{"region":"ETH","result":"pass","cnt":62},{"region":"GEO","result":"pass","cnt":1},{"region":"GHA","result":"pass","cnt":1},{"region":"HUN","result":"pass","cnt":1},{"region":"IDN","result":"null","cnt":293},{"region":"IDN","result":"fail","cnt":48},{"region":"IDN","result":"pass","cnt":1711},{"region":"IND","result":"pass","cnt":17},{"region":"IRQ","result":"fail","cnt":1},{"region":"JOR","result":"pass","cnt":1},{"region":"KEN","result":"null","cnt":4},{"region":"KEN","result":"fail","cnt":15},{"region":"KEN","result":"pass","cnt":761},{"region":"KOR","result":"fail","cnt":1},{"region":"LKA","result":"fail","cnt":4},{"region":"LKA","result":"pass","cnt":79},{"region":"LSO","result":"pass","cnt":3},{"region":"MAR","result":"pass","cnt":14},{"region":"MDA","result":"pass","cnt":3},{"region":"MEX","result":"null","cnt":4},{"region":"MEX","result":"fail","cnt":5},{"region":"MEX","result":"pass","cnt":200},{"region":"MOZ","result":"pass","cnt":5},{"region":"MYS","result":"pass","cnt":15},{"region":"NGA","result":"null","cnt":3492},{"region":"NGA","result":"fail","cnt":382},{"region":"NGA","result":"pass","cnt":12499},{"region":"PAK","result":"fail","cnt":18},{"region":"PAK","result":"pass","cnt":333},{"region":"PER","result":"pass","cnt":48},{"region":"PHL","result":"pass","cnt":2},{"region":"PSE","result":"pass","cnt":2},{"region":"RUS","result":"null","cnt":91},{"region":"RUS","result":"fail","cnt":448},{"region":"RUS","result":"pass","cnt":10267},{"region":"RWA","result":"pass","cnt":9},{"region":"SAU","result":"pass","cnt":1},{"region":"SDN","result":"fail","cnt":3},{"region":"SDN","result":"pass","cnt":8},{"region":"SWZ","result":"pass","cnt":1},{"region":"SYR","result":"pass","cnt":1},{"region":"THA","result":"null","cnt":2},{"region":"THA","result":"fail","cnt":3},{"region":"THA","result":"pass","cnt":13},{"region":"TJK","result":"pass","cnt":1},{"region":"TUN","result":"fail","cnt":1},{"region":"TUN","result":"pass","cnt":24},{"region":"TUR","result":"pass","cnt":22},{"region":"TWN","result":"fail","cnt":12},{"region":"TWN","result":"pass","cnt":24},{"region":"UKR","result":"null","cnt":13},{"region":"UKR","result":"fail","cnt":215},{"region":"UKR","result":"pass","cnt":3690},{"region":"UZB","result":"pass","cnt":2},{"region":"VEN","result":"null","cnt":7},{"region":"VEN","result":"fail","cnt":12},{"region":"VEN","result":"pass","cnt":539},{"region":"VNM","result":"fail","cnt":2},{"region":"VNM","result":"pass","cnt":157},{"region":"ZAF","result":"null","cnt":2},{"region":"ZAF","result":"fail","cnt":1},{"region":"ZAF","result":"pass","cnt":40},{"region":"ZWE","result":"pass","cnt":3}]

# W15 daily data
w15_daily = [{"pt":20260407,"result":"null","cnt":499},{"pt":20260407,"result":"fail","cnt":231},{"pt":20260407,"result":"pass","cnt":5199},{"pt":20260408,"result":"null","cnt":529},{"pt":20260408,"result":"fail","cnt":262},{"pt":20260408,"result":"pass","cnt":5213},{"pt":20260409,"result":"null","cnt":651},{"pt":20260409,"result":"fail","cnt":249},{"pt":20260409,"result":"pass","cnt":5491},{"pt":20260410,"result":"null","cnt":578},{"pt":20260410,"result":"fail","cnt":238},{"pt":20260410,"result":"pass","cnt":5715},{"pt":20260411,"result":"null","cnt":534},{"pt":20260411,"result":"fail","cnt":298},{"pt":20260411,"result":"pass","cnt":6089},{"pt":20260412,"result":"null","cnt":720},{"pt":20260412,"result":"fail","cnt":277},{"pt":20260412,"result":"pass","cnt":5732},{"pt":20260413,"result":"null","cnt":665},{"pt":20260413,"result":"fail","cnt":248},{"pt":20260413,"result":"pass","cnt":5511}]

# W14 daily data
w14_daily = [{"pt":20260331,"result":"null","cnt":514},{"pt":20260331,"result":"fail","cnt":189},{"pt":20260331,"result":"pass","cnt":5963},{"pt":20260401,"result":"null","cnt":658},{"pt":20260401,"result":"fail","cnt":309},{"pt":20260401,"result":"pass","cnt":6740},{"pt":20260402,"result":"null","cnt":603},{"pt":20260402,"result":"fail","cnt":263},{"pt":20260402,"result":"pass","cnt":6278},{"pt":20260403,"result":"null","cnt":601},{"pt":20260403,"result":"fail","cnt":229},{"pt":20260403,"result":"pass","cnt":5839},{"pt":20260404,"result":"null","cnt":508},{"pt":20260404,"result":"fail","cnt":213},{"pt":20260404,"result":"pass","cnt":5145},{"pt":20260405,"result":"null","cnt":605},{"pt":20260405,"result":"fail","cnt":231},{"pt":20260405,"result":"pass","cnt":4836},{"pt":20260406,"result":"null","cnt":652},{"pt":20260406,"result":"fail","cnt":242},{"pt":20260406,"result":"pass","cnt":5041}]

def process_week(data):
    countries = {}
    for row in data:
        region = row['region']
        result = row['result']
        cnt = row['cnt']
        if region not in countries:
            countries[region] = {'pass': 0, 'fail': 0, 'null': 0, 'total': 0}
        if result == 'pass':
            countries[region]['pass'] += cnt
        elif result == 'fail':
            countries[region]['fail'] += cnt
        else:
            countries[region]['null'] += cnt
        countries[region]['total'] += cnt
    return countries

w15 = process_week(w15_data)
w14 = process_week(w14_data)

w15_total = sum(c['total'] for c in w15.values())
w15_pass = sum(c['pass'] for c in w15.values())
w15_fail = sum(c['fail'] for c in w15.values())
w15_null = sum(c['null'] for c in w15.values())

w14_total = sum(c['total'] for c in w14.values())
w14_pass = sum(c['pass'] for c in w14.values())
w14_fail = sum(c['fail'] for c in w14.values())
w14_null = sum(c['null'] for c in w14.values())

w15_pass_rate = w15_pass / (w15_pass + w15_fail) * 100
w14_pass_rate = w14_pass / (w14_pass + w14_fail) * 100

print(f"=== Overall Stats ===")
print(f"W15: total={w15_total}, pass={w15_pass}, fail={w15_fail}, null={w15_null}, pass_rate={w15_pass_rate:.2f}%")
print(f"W14: total={w14_total}, pass={w14_pass}, fail={w14_fail}, null={w14_null}, pass_rate={w14_pass_rate:.2f}%")
print(f"Delta: {w15_pass_rate - w14_pass_rate:+.2f}pp")
print()

all_regions = set(list(w15.keys()) + list(w14.keys()))
country_stats = []
for r in sorted(all_regions):
    w15c = w15.get(r, {'pass': 0, 'fail': 0, 'null': 0, 'total': 0})
    w14c = w14.get(r, {'pass': 0, 'fail': 0, 'null': 0, 'total': 0})
    w15_decided = w15c['pass'] + w15c['fail']
    w14_decided = w14c['pass'] + w14c['fail']
    w15_pr = w15c['pass'] / w15_decided * 100 if w15_decided > 0 else None
    w14_pr = w14c['pass'] / w14_decided * 100 if w14_decided > 0 else None
    delta = None
    if w15_pr is not None and w14_pr is not None:
        delta = w15_pr - w14_pr
    country_stats.append({
        'region': r,
        'w15_total': w15c['total'], 'w15_pass': w15c['pass'], 'w15_fail': w15c['fail'], 'w15_null': w15c['null'],
        'w15_pass_rate': round(w15_pr, 2) if w15_pr is not None else None,
        'w14_total': w14c['total'], 'w14_pass': w14c['pass'], 'w14_fail': w14c['fail'], 'w14_null': w14c['null'],
        'w14_pass_rate': round(w14_pr, 2) if w14_pr is not None else None,
        'delta_pp': round(delta, 2) if delta is not None else None,
        'alert': delta is not None and delta < -2
    })

print("=== Countries with >2pp Drop (min 10 sessions) ===")
alerts = [c for c in country_stats if c['alert'] and c['w15_total'] >= 10]
for a in sorted(alerts, key=lambda x: x['delta_pp']):
    print(f"  {a['region']}: W15={a['w15_pass_rate']:.1f}% W14={a['w14_pass_rate']:.1f}% delta={a['delta_pp']:+.1f}pp (n={a['w15_total']})")

print()
print("=== Top 15 Countries by Volume (W15) ===")
top = sorted(country_stats, key=lambda x: x['w15_total'], reverse=True)[:15]
for c in top:
    pr_str = f"{c['w15_pass_rate']:.1f}%" if c['w15_pass_rate'] is not None else 'N/A'
    d_str = f"{c['delta_pp']:+.1f}pp" if c['delta_pp'] is not None else 'NEW'
    print(f"  {c['region']}: total={c['w15_total']}, pass_rate={pr_str}, delta={d_str}")

# Daily pass rate
print()
print("=== Daily Pass Rate (W15) ===")
daily_w15 = {}
for row in w15_daily:
    pt = str(row['pt'])
    if pt not in daily_w15:
        daily_w15[pt] = {'pass': 0, 'fail': 0, 'null': 0}
    daily_w15[pt][row['result']] += row['cnt']

for d in sorted(daily_w15.keys()):
    p = daily_w15[d]['pass']
    f = daily_w15[d]['fail']
    pr = p / (p + f) * 100
    print(f"  {d}: pass={p}, fail={f}, null={daily_w15[d]['null']}, rate={pr:.2f}%")

print()
print("=== Daily Pass Rate (W14) ===")
daily_w14 = {}
for row in w14_daily:
    pt = str(row['pt'])
    if pt not in daily_w14:
        daily_w14[pt] = {'pass': 0, 'fail': 0, 'null': 0}
    daily_w14[pt][row['result']] += row['cnt']

for d in sorted(daily_w14.keys()):
    p = daily_w14[d]['pass']
    f = daily_w14[d]['fail']
    pr = p / (p + f) * 100
    print(f"  {d}: pass={p}, fail={f}, null={daily_w14[d]['null']}, rate={pr:.2f}%")

# Output JSON for report building
output = {
    'w15_overall': {'total': w15_total, 'pass': w15_pass, 'fail': w15_fail, 'null': w15_null, 'pass_rate': round(w15_pass_rate, 2)},
    'w14_overall': {'total': w14_total, 'pass': w14_pass, 'fail': w14_fail, 'null': w14_null, 'pass_rate': round(w14_pass_rate, 2)},
    'delta_pp': round(w15_pass_rate - w14_pass_rate, 2),
    'country_stats': country_stats,
    'alerts': alerts,
    'top_countries': top
}
with open('/workspace/analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nOutput saved to /workspace/analysis_output.json")
