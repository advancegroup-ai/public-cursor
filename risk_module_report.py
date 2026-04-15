#!/usr/bin/env python3
"""Generate Risk Module Disposition Analysis HTML report and publish to OSS."""

import json
import subprocess
import sys

VT_TOKEN = "vt_sk_21089b3935576ce6c994c3aa2f14ad0d160b8eeafc583bd0b7583e520ecba2af"
VT_BASE = "https://vibe-track.ngrok.app"

# ============================================================
# DATA (collected from queries)
# ============================================================

# Overall by hit_code
OVERALL_DATA = [
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","total_cnt":452,"unique_txns":157},
    {"hit_code":"HIT_LD_SPLICING","total_cnt":387,"unique_txns":387},
    {"hit_code":"OVER_MAX_REVIEW_LIMIT","total_cnt":348,"unique_txns":111},
    {"hit_code":"HIT_FACE_MULTIPLE_CARD","total_cnt":158,"unique_txns":158},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","total_cnt":140,"unique_txns":140},
    {"hit_code":"HIT_FACE_PUNISHMENT","total_cnt":106,"unique_txns":106},
    {"hit_code":"HIT_INJECTION_ICON_DETECTED","total_cnt":92,"unique_txns":16},
    {"hit_code":"HIT_RISKY_DEVICE","total_cnt":85,"unique_txns":85},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","total_cnt":68,"unique_txns":68},
    {"hit_code":"HIT_ID_DUPLICATE","total_cnt":55,"unique_txns":55},
    {"hit_code":"HIT_FACE_DUPLICATE","total_cnt":51,"unique_txns":51},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","total_cnt":30,"unique_txns":30},
    {"hit_code":"HIT_FACE_BLACK_LIST","total_cnt":14,"unique_txns":5},
    {"hit_code":"OVER_MAX_HIT_LIMIT","total_cnt":4,"unique_txns":4},
    {"hit_code":"HIT_CRED_BLACK_LIST","total_cnt":3,"unique_txns":3},
    {"hit_code":"HIT_FACE_FREQUENCY_LIMIT","total_cnt":2,"unique_txns":2},
]

# By customer x hit_code
CUSTOMER_HITCODE_DATA = [
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","customer_id":9929123352,"cnt":379,"unique_txns":87},
    {"hit_code":"HIT_LD_SPLICING","customer_id":9929123352,"cnt":350,"unique_txns":350},
    {"hit_code":"OVER_MAX_REVIEW_LIMIT","customer_id":9929123352,"cnt":308,"unique_txns":92},
    {"hit_code":"HIT_FACE_MULTIPLE_CARD","customer_id":9929123352,"cnt":125,"unique_txns":125},
    {"hit_code":"HIT_INJECTION_ICON_DETECTED","customer_id":9929123352,"cnt":92,"unique_txns":16},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929123352,"cnt":85,"unique_txns":85},
    {"hit_code":"HIT_RISKY_DEVICE","customer_id":9929123352,"cnt":75,"unique_txns":75},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","customer_id":9929123352,"cnt":52,"unique_txns":52},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929123352,"cnt":41,"unique_txns":41},
    {"hit_code":"HIT_FACE_DUPLICATE","customer_id":9929123352,"cnt":34,"unique_txns":34},
    {"hit_code":"HIT_ID_DUPLICATE","customer_id":9929123352,"cnt":31,"unique_txns":31},
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","customer_id":9929119403,"cnt":27,"unique_txns":27},
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","customer_id":9929123961,"cnt":22,"unique_txns":22},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","customer_id":9929123352,"cnt":19,"unique_txns":19},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929123795,"cnt":19,"unique_txns":19},
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","customer_id":9929124132,"cnt":18,"unique_txns":18},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929123961,"cnt":16,"unique_txns":16},
    {"hit_code":"OVER_MAX_REVIEW_LIMIT","customer_id":9929116935,"cnt":16,"unique_txns":4},
    {"hit_code":"OVER_MAX_REVIEW_LIMIT","customer_id":9929123961,"cnt":14,"unique_txns":14},
    {"hit_code":"HIT_FACE_MULTIPLE_CARD","customer_id":9929116935,"cnt":14,"unique_txns":14},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929123795,"cnt":12,"unique_txns":12},
    {"hit_code":"HIT_ID_DUPLICATE","customer_id":9929123961,"cnt":12,"unique_txns":12},
    {"hit_code":"HIT_FACE_MULTIPLE_CARD","customer_id":9929124132,"cnt":12,"unique_txns":12},
    {"hit_code":"HIT_LD_SPLICING","customer_id":9929122151,"cnt":11,"unique_txns":11},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","customer_id":9929124132,"cnt":11,"unique_txns":11},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929119403,"cnt":11,"unique_txns":11},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929120638,"cnt":11,"unique_txns":11},
    {"hit_code":"HIT_ID_DUPLICATE","customer_id":9929124132,"cnt":10,"unique_txns":10},
    {"hit_code":"HIT_LD_SPLICING","customer_id":9929123961,"cnt":9,"unique_txns":9},
    {"hit_code":"HIT_FACE_BLACK_LIST","customer_id":9929120638,"cnt":9,"unique_txns":3},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929116935,"cnt":9,"unique_txns":9},
    {"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","customer_id":9929116935,"cnt":8,"unique_txns":8},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929119403,"cnt":8,"unique_txns":8},
    {"hit_code":"HIT_FACE_PUNISHMENT","customer_id":9929116935,"cnt":8,"unique_txns":8},
    {"hit_code":"HIT_LD_SPLICING","customer_id":9929124262,"cnt":7,"unique_txns":7},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929124132,"cnt":7,"unique_txns":7},
    {"hit_code":"HIT_LD_SPLICING","customer_id":9929123761,"cnt":7,"unique_txns":7},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929119349,"cnt":6,"unique_txns":6},
    {"hit_code":"HIT_FACE_DUPLICATE","customer_id":9929123961,"cnt":6,"unique_txns":6},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929120638,"cnt":5,"unique_txns":5},
    {"hit_code":"HIT_FACE_BLACK_LIST","customer_id":9929119367,"cnt":5,"unique_txns":2},
    {"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","customer_id":9929116935,"cnt":5,"unique_txns":2},
    {"hit_code":"HIT_FACE_DUPLICATE","customer_id":9929116935,"cnt":5,"unique_txns":5},
    {"hit_code":"HIT_RISKY_DEVICE","customer_id":9929119349,"cnt":4,"unique_txns":4},
    {"hit_code":"HIT_RISKY_DEVICE","customer_id":9929116935,"cnt":4,"unique_txns":4},
    {"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","customer_id":9929122151,"cnt":4,"unique_txns":4},
    {"hit_code":"OVER_MAX_HIT_LIMIT","customer_id":9929123352,"cnt":4,"unique_txns":4},
]

# Customer total transactions
CUSTOMER_TOTALS = [
    {"customer_id":9929123352,"total_txns":191069,"risk_hit_cnt":1595},
    {"customer_id":9929123961,"total_txns":1627,"risk_hit_cnt":80},
    {"customer_id":9929116935,"total_txns":1123,"risk_hit_cnt":74},
    {"customer_id":9929124132,"total_txns":6689,"risk_hit_cnt":62},
    {"customer_id":9929119403,"total_txns":350,"risk_hit_cnt":54},
    {"customer_id":9929123795,"total_txns":87,"risk_hit_cnt":31},
    {"customer_id":9929120638,"total_txns":106,"risk_hit_cnt":26},
    {"customer_id":9929122151,"total_txns":190,"risk_hit_cnt":15},
    {"customer_id":9929119349,"total_txns":279,"risk_hit_cnt":14},
    {"customer_id":9929123761,"total_txns":977,"risk_hit_cnt":10},
    {"customer_id":9929119367,"total_txns":375,"risk_hit_cnt":7},
    {"customer_id":9929124262,"total_txns":4011,"risk_hit_cnt":7},
    {"customer_id":9929123407,"total_txns":141,"risk_hit_cnt":4},
    {"customer_id":9929123945,"total_txns":387,"risk_hit_cnt":4},
    {"customer_id":9929124067,"total_txns":1,"risk_hit_cnt":1},
    {"customer_id":9929124394,"total_txns":1454,"risk_hit_cnt":1},
]

# Daily trend by hit_code
DAILY_TREND = [{"pt":20260401,"hit_code":"HIT_LD_SPLICING","unique_txns":39},{"pt":20260401,"hit_code":"HIT_RISKY_DEVICE","unique_txns":12},{"pt":20260401,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":10},{"pt":20260401,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":10},{"pt":20260401,"hit_code":"HIT_ID_DUPLICATE","unique_txns":5},{"pt":20260401,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":4},{"pt":20260401,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":4},{"pt":20260401,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":3},{"pt":20260401,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":2},{"pt":20260401,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":1},{"pt":20260401,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":1},{"pt":20260402,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":40},{"pt":20260402,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":34},{"pt":20260402,"hit_code":"HIT_LD_SPLICING","unique_txns":33},{"pt":20260402,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":29},{"pt":20260402,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":24},{"pt":20260402,"hit_code":"HIT_RISKY_DEVICE","unique_txns":18},{"pt":20260402,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":10},{"pt":20260402,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":7},{"pt":20260402,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":6},{"pt":20260402,"hit_code":"HIT_FACE_BLACK_LIST","unique_txns":5},{"pt":20260402,"hit_code":"HIT_ID_DUPLICATE","unique_txns":5},{"pt":20260402,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":2},{"pt":20260402,"hit_code":"HIT_CRED_BLACK_LIST","unique_txns":2},{"pt":20260402,"hit_code":"HIT_FACE_FREQUENCY_LIMIT","unique_txns":2},{"pt":20260403,"hit_code":"HIT_LD_SPLICING","unique_txns":25},{"pt":20260403,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":17},{"pt":20260403,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":15},{"pt":20260403,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":10},{"pt":20260403,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":8},{"pt":20260403,"hit_code":"HIT_ID_DUPLICATE","unique_txns":7},{"pt":20260403,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":7},{"pt":20260403,"hit_code":"HIT_RISKY_DEVICE","unique_txns":5},{"pt":20260403,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":3},{"pt":20260403,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":3},{"pt":20260403,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":1},{"pt":20260404,"hit_code":"HIT_LD_SPLICING","unique_txns":24},{"pt":20260404,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":17},{"pt":20260404,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":11},{"pt":20260404,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":8},{"pt":20260404,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":6},{"pt":20260404,"hit_code":"HIT_RISKY_DEVICE","unique_txns":5},{"pt":20260404,"hit_code":"HIT_ID_DUPLICATE","unique_txns":5},{"pt":20260404,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":4},{"pt":20260404,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":4},{"pt":20260404,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":2},{"pt":20260404,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260405,"hit_code":"HIT_LD_SPLICING","unique_txns":21},{"pt":20260405,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":19},{"pt":20260405,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":8},{"pt":20260405,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":4},{"pt":20260405,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":4},{"pt":20260405,"hit_code":"HIT_RISKY_DEVICE","unique_txns":4},{"pt":20260405,"hit_code":"HIT_ID_DUPLICATE","unique_txns":3},{"pt":20260405,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":3},{"pt":20260405,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":2},{"pt":20260405,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":1},{"pt":20260406,"hit_code":"HIT_LD_SPLICING","unique_txns":38},{"pt":20260406,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":17},{"pt":20260406,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":10},{"pt":20260406,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":8},{"pt":20260406,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":8},{"pt":20260406,"hit_code":"HIT_RISKY_DEVICE","unique_txns":5},{"pt":20260406,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":5},{"pt":20260406,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":4},{"pt":20260406,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":3},{"pt":20260406,"hit_code":"HIT_ID_DUPLICATE","unique_txns":2},{"pt":20260406,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260406,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":1},{"pt":20260407,"hit_code":"HIT_LD_SPLICING","unique_txns":31},{"pt":20260407,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":11},{"pt":20260407,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":7},{"pt":20260407,"hit_code":"HIT_RISKY_DEVICE","unique_txns":7},{"pt":20260407,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":7},{"pt":20260407,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":6},{"pt":20260407,"hit_code":"HIT_ID_DUPLICATE","unique_txns":5},{"pt":20260407,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":5},{"pt":20260407,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":5},{"pt":20260407,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":3},{"pt":20260407,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260407,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":1},{"pt":20260408,"hit_code":"HIT_LD_SPLICING","unique_txns":26},{"pt":20260408,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":17},{"pt":20260408,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":11},{"pt":20260408,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":8},{"pt":20260408,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":4},{"pt":20260408,"hit_code":"HIT_ID_DUPLICATE","unique_txns":3},{"pt":20260408,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":3},{"pt":20260408,"hit_code":"HIT_RISKY_DEVICE","unique_txns":2},{"pt":20260408,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":2},{"pt":20260408,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":1},{"pt":20260409,"hit_code":"HIT_LD_SPLICING","unique_txns":23},{"pt":20260409,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":19},{"pt":20260409,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":18},{"pt":20260409,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":12},{"pt":20260409,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":12},{"pt":20260409,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":10},{"pt":20260409,"hit_code":"HIT_RISKY_DEVICE","unique_txns":8},{"pt":20260409,"hit_code":"HIT_ID_DUPLICATE","unique_txns":7},{"pt":20260409,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":6},{"pt":20260409,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":5},{"pt":20260409,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":4},{"pt":20260409,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260410,"hit_code":"HIT_LD_SPLICING","unique_txns":28},{"pt":20260410,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":15},{"pt":20260410,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":12},{"pt":20260410,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":8},{"pt":20260410,"hit_code":"HIT_ID_DUPLICATE","unique_txns":5},{"pt":20260410,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":5},{"pt":20260410,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":4},{"pt":20260410,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":3},{"pt":20260410,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":3},{"pt":20260410,"hit_code":"HIT_RISKY_DEVICE","unique_txns":2},{"pt":20260410,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260410,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":1},{"pt":20260411,"hit_code":"HIT_LD_SPLICING","unique_txns":18},{"pt":20260411,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":14},{"pt":20260411,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":11},{"pt":20260411,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":10},{"pt":20260411,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":10},{"pt":20260411,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":5},{"pt":20260411,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":5},{"pt":20260411,"hit_code":"HIT_ID_DUPLICATE","unique_txns":4},{"pt":20260411,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":3},{"pt":20260411,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":2},{"pt":20260411,"hit_code":"HIT_RISKY_DEVICE","unique_txns":2},{"pt":20260411,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":2},{"pt":20260411,"hit_code":"OVER_MAX_HIT_LIMIT","unique_txns":1},{"pt":20260412,"hit_code":"HIT_LD_SPLICING","unique_txns":33},{"pt":20260412,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":12},{"pt":20260412,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":12},{"pt":20260412,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":8},{"pt":20260412,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":7},{"pt":20260412,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":7},{"pt":20260412,"hit_code":"HIT_RISKY_DEVICE","unique_txns":4},{"pt":20260412,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":3},{"pt":20260412,"hit_code":"OVER_MAX_HIT_LIMIT","unique_txns":2},{"pt":20260412,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":2},{"pt":20260412,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":2},{"pt":20260412,"hit_code":"HIT_ID_DUPLICATE","unique_txns":1},{"pt":20260413,"hit_code":"HIT_LD_SPLICING","unique_txns":28},{"pt":20260413,"hit_code":"OVER_MAX_REVIEW_LIMIT","unique_txns":25},{"pt":20260413,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":11},{"pt":20260413,"hit_code":"HIT_RISKY_DEVICE","unique_txns":9},{"pt":20260413,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":6},{"pt":20260413,"hit_code":"HIT_FACE_DUPLICATE","unique_txns":5},{"pt":20260413,"hit_code":"HIT_FACE_PUNISHMENT","unique_txns":5},{"pt":20260413,"hit_code":"HIT_ID_DUPLICATE","unique_txns":3},{"pt":20260413,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":1},{"pt":20260413,"hit_code":"OVER_MAX_HIT_LIMIT","unique_txns":1},{"pt":20260413,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":1},{"pt":20260414,"hit_code":"HIT_LD_SPLICING","unique_txns":20},{"pt":20260414,"hit_code":"HIT_FACE_MULTIPLE_CARD","unique_txns":12},{"pt":20260414,"hit_code":"HIT_FACE_MULTIPLE_BIRTHDAY","unique_txns":8},{"pt":20260414,"hit_code":"HIT_VIRTUAL_CAMERA_DETECTED","unique_txns":4},{"pt":20260414,"hit_code":"HIT_ID_FACE_MULTIPLE_ID_NUMBER","unique_txns":4},{"pt":20260414,"hit_code":"HIT_INJECTION_ICON_DETECTED","unique_txns":2},{"pt":20260414,"hit_code":"HIT_RISKY_DEVICE","unique_txns":2},{"pt":20260414,"hit_code":"HIT_ID_FACE_MULTIPLE_BIRTHDAY","unique_txns":1},{"pt":20260414,"hit_code":"HIT_CRED_BLACK_LIST","unique_txns":1}]

# ============================================================
# MAPPINGS
# ============================================================

DISPOSITION = {
    "HIT_FACE_BLACK_LIST": "直接拦截",
    "HIT_FACE_PUNISHMENT": "直接拦截",
    "HIT_FACE_FREQUENCY_LIMIT": "直接拦截",
    "OVER_MAX_HIT_LIMIT": "直接拦截",
    "OVER_MAX_REVIEW_LIMIT": "直接拦截",
    "HIT_FACE_MULTIPLE_CARD": "直接拦截",
    "HIT_FACE_MULTIPLE_BIRTHDAY": "直接拦截",
    "HIT_ID_FACE_MULTIPLE_ID_NUMBER": "直接拦截",
    "HIT_ID_FACE_MULTIPLE_BIRTHDAY": "直接拦截",
    "HIT_VIRTUAL_CAMERA_DETECTED": "直接拦截",
    "HIT_INJECTION_ICON_DETECTED": "直接拦截",
    "HIT_RISKY_DEVICE": "直接拦截",
    "HIT_ID_DUPLICATE": "直接拦截",
    "HIT_FACE_DUPLICATE": "直接拦截",
    "HIT_CRED_BLACK_LIST": "直接拦截",
    "HIT_LD_SPLICING": "送人审",
    "HIT_IP_REJECT_HISTORY": "空跑",
    "HIT_DEVICE_TIMEZONE_MISMATCH": "空跑",
    "HIT_DEVICE_COUNTRY_MISMATCH": "空跑",
    "HIT_DEVICE_CLIENT_TZ_MISMATCH": "空跑",
}

RULE_DESC = {
    "HIT_FACE_BLACK_LIST": "人脸黑名单",
    "HIT_FACE_PUNISHMENT": "人脸请求冷却中",
    "HIT_FACE_FREQUENCY_LIMIT": "人脸请求频率过高",
    "OVER_MAX_HIT_LIMIT": "人脸高频命中风控规则",
    "OVER_MAX_REVIEW_LIMIT": "人脸被多次人审拒绝",
    "HIT_FACE_MULTIPLE_CARD": "人脸关联多个证件",
    "HIT_FACE_MULTIPLE_BIRTHDAY": "人脸关联多个生日",
    "HIT_ID_FACE_MULTIPLE_ID_NUMBER": "证件人脸多个身份证号",
    "HIT_ID_FACE_MULTIPLE_BIRTHDAY": "证件人脸多个生日",
    "HIT_VIRTUAL_CAMERA_DETECTED": "命中虚拟相机检测",
    "HIT_INJECTION_ICON_DETECTED": "环境图注入图标检测",
    "HIT_RISKY_DEVICE": "摄像头开机时间检测",
    "HIT_LD_SPLICING": "环境图多图层检测",
    "HIT_ID_DUPLICATE": "证件背面重复检测",
    "HIT_FACE_DUPLICATE": "活体远脸重复检测",
    "HIT_CRED_BLACK_LIST": "证件黑名单",
    "HIT_IP_REJECT_HISTORY": "IP 高危",
    "HIT_DEVICE_TIMEZONE_MISMATCH": "设备+时区变更",
    "HIT_DEVICE_COUNTRY_MISMATCH": "设备+IP国家变更",
    "HIT_DEVICE_CLIENT_TZ_MISMATCH": "设备+前端时区变更",
}

CUSTOMER_NAMES = {
    9929123352: "Bybit",
    9929124002: "MEXC",
    9929122963: "DTC Pay",
    45296432: "TT (PIPO)",
    9929122076: "TT (PIPO)",
    9929123016: "TT (PIPO)",
    9929122948: "TT (PIPO)",
    9929123194: "TT (PIPO)",
    9929116935: "Cust_16935",
    9929119403: "Cust_19403",
    9929123961: "Cust_23961",
    9929124132: "Cust_24132",
    9929119349: "Cust_19349",
    9929122151: "Cust_22151",
    9929120638: "Cust_20638",
    9929123795: "Cust_23795",
    9929119367: "Cust_19367",
    9929124262: "Cust_24262",
    9929123761: "Cust_23761",
    9929123407: "Cust_23407",
    9929123945: "Cust_23945",
    9929124394: "Cust_24394",
    9929124067: "Cust_24067",
}

DISP_COLORS = {
    "直接拦截": "#ef4444",
    "送人审": "#f59e0b",
    "空跑": "#6b7280",
}

def cname(cid):
    return CUSTOMER_NAMES.get(cid, str(cid))

def build_html():
    # Prepare daily disposition trend
    dates = sorted(set(str(d["pt"]) for d in DAILY_TREND))
    date_labels = [f"{d[4:6]}-{d[6:8]}" for d in dates]

    # Aggregate daily by disposition
    daily_disp = {}
    for d in DAILY_TREND:
        dt = str(d["pt"])
        disp = DISPOSITION.get(d["hit_code"], "未分类")
        daily_disp.setdefault(disp, {})
        daily_disp[disp][dt] = daily_disp[disp].get(dt, 0) + d["unique_txns"]

    # Daily by top rules
    top_rules = ["HIT_LD_SPLICING","HIT_VIRTUAL_CAMERA_DETECTED","OVER_MAX_REVIEW_LIMIT",
                 "HIT_FACE_MULTIPLE_CARD","HIT_FACE_MULTIPLE_BIRTHDAY","HIT_FACE_PUNISHMENT",
                 "HIT_RISKY_DEVICE"]
    daily_rules = {}
    for d in DAILY_TREND:
        if d["hit_code"] in top_rules:
            daily_rules.setdefault(d["hit_code"], {})
            daily_rules[d["hit_code"]][str(d["pt"])] = d["unique_txns"]

    rule_colors = {
        "HIT_LD_SPLICING": "#f59e0b",
        "HIT_VIRTUAL_CAMERA_DETECTED": "#ef4444",
        "OVER_MAX_REVIEW_LIMIT": "#8b5cf6",
        "HIT_FACE_MULTIPLE_CARD": "#3b82f6",
        "HIT_FACE_MULTIPLE_BIRTHDAY": "#10b981",
        "HIT_FACE_PUNISHMENT": "#ec4899",
        "HIT_RISKY_DEVICE": "#f97316",
    }

    # Build chart datasets for disposition trend
    disp_datasets = []
    for disp_name in ["直接拦截", "送人审", "空跑"]:
        if disp_name in daily_disp:
            vals = [daily_disp[disp_name].get(d, 0) for d in dates]
            disp_datasets.append({
                "label": disp_name,
                "data": vals,
                "borderColor": DISP_COLORS.get(disp_name, "#999"),
                "backgroundColor": DISP_COLORS.get(disp_name, "#999") + "33",
                "fill": True,
                "tension": 0.3,
            })

    # Build chart datasets for rule trend
    rule_datasets = []
    for rule in top_rules:
        if rule in daily_rules:
            vals = [daily_rules[rule].get(d, 0) for d in dates]
            rule_datasets.append({
                "label": RULE_DESC.get(rule, rule),
                "data": vals,
                "borderColor": rule_colors.get(rule, "#999"),
                "tension": 0.3,
                "pointRadius": 3,
            })

    # Build customer breakdown table rows
    customer_rows = ""
    for ct in CUSTOMER_TOTALS:
        cid = ct["customer_id"]
        name = cname(cid)
        total = ct["total_txns"]
        risk_cnt = ct["risk_hit_cnt"]
        hit_rate = risk_cnt / total * 100 if total > 0 else 0

        # Get hit_codes for this customer
        cust_hits = [r for r in CUSTOMER_HITCODE_DATA if r["customer_id"] == cid]
        # Group by disposition
        disp_cnts = {}
        for ch in cust_hits:
            disp = DISPOSITION.get(ch["hit_code"], "未分类")
            disp_cnts[disp] = disp_cnts.get(disp, 0) + ch["unique_txns"]

        intercept = disp_cnts.get("直接拦截", 0)
        review = disp_cnts.get("送人审", 0)
        dryrun = disp_cnts.get("空跑", 0)

        top3 = sorted(cust_hits, key=lambda x: -x["unique_txns"])[:3]
        top3_str = ", ".join([f"{RULE_DESC.get(r['hit_code'], r['hit_code'])}({r['unique_txns']})" for r in top3])

        color = "#ef4444" if hit_rate > 5 else ("#f59e0b" if hit_rate > 1 else "#10b981")
        customer_rows += f"""<tr>
            <td>{name}</td>
            <td style="text-align:right">{total:,}</td>
            <td style="text-align:right">{risk_cnt}</td>
            <td style="text-align:right;color:{color};font-weight:600">{hit_rate:.2f}%</td>
            <td style="text-align:right;color:#ef4444">{intercept}</td>
            <td style="text-align:right;color:#f59e0b">{review}</td>
            <td style="text-align:right;color:#6b7280">{dryrun}</td>
            <td style="font-size:12px">{top3_str}</td>
        </tr>"""

    # Overall rule detail table
    rule_rows = ""
    for r in OVERALL_DATA:
        disp = DISPOSITION.get(r["hit_code"], "未分类")
        disp_color = DISP_COLORS.get(disp, "#999")
        desc = RULE_DESC.get(r["hit_code"], r["hit_code"])
        rule_rows += f"""<tr>
            <td><code>{r['hit_code']}</code></td>
            <td>{desc}</td>
            <td style="text-align:center"><span class="badge" style="background:{disp_color}">{disp}</span></td>
            <td style="text-align:right">{r['unique_txns']}</td>
            <td style="text-align:right">{r['total_cnt']}</td>
            <td style="text-align:right">{r['total_cnt']/r['unique_txns']:.1f}x</td>
        </tr>"""

    # Per-customer detail tables (top 5 customers)
    top_customers = [9929123352, 9929123961, 9929116935, 9929124132, 9929119403]
    customer_detail_sections = ""
    for cid in top_customers:
        name = cname(cid)
        ct = next((c for c in CUSTOMER_TOTALS if c["customer_id"] == cid), None)
        if not ct:
            continue
        cust_hits = sorted([r for r in CUSTOMER_HITCODE_DATA if r["customer_id"] == cid], key=lambda x: -x["unique_txns"])
        if not cust_hits:
            continue

        rows = ""
        for ch in cust_hits:
            disp = DISPOSITION.get(ch["hit_code"], "未分类")
            disp_color = DISP_COLORS.get(disp, "#999")
            desc = RULE_DESC.get(ch["hit_code"], ch["hit_code"])
            rows += f"""<tr>
                <td><code>{ch['hit_code']}</code></td>
                <td>{desc}</td>
                <td style="text-align:center"><span class="badge" style="background:{disp_color}">{disp}</span></td>
                <td style="text-align:right">{ch['unique_txns']}</td>
            </tr>"""

        total = ct["total_txns"]
        risk = ct["risk_hit_cnt"]
        rate = risk / total * 100 if total > 0 else 0
        customer_detail_sections += f"""
        <h3>{name} <span style="color:#94a3b8;font-weight:400">(ID: {cid})</span></h3>
        <div style="display:flex;gap:20px;margin-bottom:12px">
            <div class="mini-stat">总交易: <strong>{total:,}</strong></div>
            <div class="mini-stat">风控命中: <strong>{risk}</strong></div>
            <div class="mini-stat">命中率: <strong style="color:{'#ef4444' if rate>5 else '#f59e0b' if rate>1 else '#10b981'}">{rate:.2f}%</strong></div>
        </div>
        <table class="data-table">
            <thead><tr><th>Error Code</th><th>规则说明</th><th>处置方式</th><th>命中数(UV)</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    # SQL queries for collapsible section
    sql_queries = """
-- 1. Overall hit_code distribution (2 weeks)
SELECT hit_code, COUNT(*) AS cnt, COUNT(DISTINCT transaction_id) AS unique_txns
FROM adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record
WHERE pt BETWEEN '20260401' AND '20260414'
GROUP BY hit_code ORDER BY cnt DESC;

-- 2. Hit_code by customer
SELECT r.hit_code, f.customer_id, COUNT(*) AS cnt, COUNT(DISTINCT r.transaction_id) AS unique_txns
FROM adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record r
JOIN adv_guardian_data_core.dw_advance_business_ekyc_transaction_funnel_detail f
  ON r.transaction_id = f.uid AND r.pt = f.pt
WHERE r.pt BETWEEN '20260401' AND '20260414' AND r.hit_code != 'PASS'
GROUP BY r.hit_code, f.customer_id ORDER BY cnt DESC;

-- 3. Daily trend by hit_code
SELECT r.pt, r.hit_code, COUNT(DISTINCT r.transaction_id) AS unique_txns
FROM adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record r
WHERE r.pt BETWEEN '20260401' AND '20260414' AND r.hit_code != 'PASS'
GROUP BY r.pt, r.hit_code ORDER BY r.pt;
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Risk Module 处置方式命中分析 | 2026-04-01 ~ 2026-04-14</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; line-height:1.6; }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px 20px; }}
  h1 {{ font-size:28px; font-weight:700; margin-bottom:4px; }}
  h2 {{ font-size:20px; font-weight:600; margin:32px 0 16px; color:#f8fafc; border-left:4px solid #3b82f6; padding-left:12px; }}
  h3 {{ font-size:17px; font-weight:600; margin:24px 0 8px; color:#cbd5e1; }}
  .subtitle {{ color:#94a3b8; font-size:14px; margin-bottom:24px; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin:20px 0 32px; }}
  .summary-card {{ background:#1e293b; border-radius:12px; padding:20px; border:1px solid #334155; }}
  .summary-card .label {{ font-size:13px; color:#94a3b8; margin-bottom:4px; }}
  .summary-card .value {{ font-size:28px; font-weight:700; }}
  .summary-card .sub {{ font-size:12px; color:#64748b; margin-top:4px; }}
  .chart-container {{ background:#1e293b; border-radius:12px; padding:20px; border:1px solid #334155; margin:16px 0; }}
  .chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:900px) {{ .chart-row {{ grid-template-columns:1fr; }} }}
  .data-table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; font-size:13px; }}
  .data-table th {{ background:#334155; color:#cbd5e1; padding:10px 12px; text-align:left; font-weight:600; position:sticky; top:0; }}
  .data-table td {{ padding:8px 12px; border-bottom:1px solid #1e293b; }}
  .data-table tr:hover {{ background:#1e293b; }}
  .data-table code {{ background:#334155; padding:2px 6px; border-radius:4px; font-size:11px; color:#93c5fd; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; color:#fff; }}
  .mini-stat {{ background:#1e293b; padding:8px 16px; border-radius:8px; font-size:13px; color:#94a3b8; }}
  .mini-stat strong {{ color:#f8fafc; }}
  details {{ margin:16px 0; }}
  summary {{ cursor:pointer; color:#3b82f6; font-weight:600; padding:8px 0; }}
  pre {{ background:#0f172a; border:1px solid #334155; border-radius:8px; padding:16px; overflow-x:auto; font-size:12px; color:#94a3b8; }}
  .legend-inline {{ display:flex; flex-wrap:wrap; gap:16px; margin:8px 0 16px; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; font-size:13px; color:#94a3b8; }}
  .legend-dot {{ width:12px; height:12px; border-radius:50%; }}
  .note {{ background:#1e293b; border-left:4px solid #f59e0b; padding:12px 16px; margin:16px 0; border-radius:0 8px 8px 0; font-size:13px; color:#fbbf24; }}
</style>
</head>
<body>
<div class="container">
  <h1>Risk Module 处置方式命中分析</h1>
  <div class="subtitle">数据范围: 2026-04-01 ~ 2026-04-14 (14天) | 生成时间: 2026-04-15 | 数据源: risk_execution_record</div>

  <div class="note">
    <strong>处置方式说明:</strong>
    <strong style="color:#ef4444">直接拦截</strong> = 系统自动拒绝，不进入后续流程 &nbsp;|&nbsp;
    <strong style="color:#f59e0b">送人审</strong> = 转人工审核链路二次研判 &nbsp;|&nbsp;
    <strong style="color:#6b7280">空跑</strong> = 仅记录观察，不影响审核结果
  </div>

  <h2>1. 总览</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="label">风控命中 UV (去重交易数)</div>
      <div class="value" style="color:#ef4444">1,378</div>
      <div class="sub">含重复记录 1,995 条</div>
    </div>
    <div class="summary-card">
      <div class="label">直接拦截</div>
      <div class="value" style="color:#ef4444">991</div>
      <div class="sub">71.9% of all risk hits</div>
    </div>
    <div class="summary-card">
      <div class="label">送人审</div>
      <div class="value" style="color:#f59e0b">387</div>
      <div class="sub">28.1% of all risk hits</div>
    </div>
    <div class="summary-card">
      <div class="label">空跑</div>
      <div class="value" style="color:#6b7280">0</div>
      <div class="sub">IP/设备时区规则暂未产生命中</div>
    </div>
    <div class="summary-card">
      <div class="label">涉及客户数</div>
      <div class="value" style="color:#3b82f6">16</div>
      <div class="sub">有风控命中的客户</div>
    </div>
    <div class="summary-card">
      <div class="label">日均命中 UV</div>
      <div class="value" style="color:#818cf8">~98</div>
      <div class="sub">1,378 / 14 days</div>
    </div>
  </div>

  <h2>2. 处置方式每日趋势</h2>
  <div class="chart-container">
    <canvas id="dispTrend" height="280"></canvas>
  </div>

  <h2>3. Top 规则每日趋势</h2>
  <div class="chart-container">
    <canvas id="ruleTrend" height="300"></canvas>
  </div>

  <h2>4. 全量规则命中明细</h2>
  <div class="legend-inline">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>直接拦截</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>送人审</div>
    <div class="legend-item"><div class="legend-dot" style="background:#6b7280"></div>空跑</div>
  </div>
  <div style="overflow-x:auto">
  <table class="data-table">
    <thead><tr>
      <th>Error Code</th><th>风控报错原因</th><th>处置方式</th>
      <th style="text-align:right">命中数(UV)</th><th style="text-align:right">记录数(PV)</th><th style="text-align:right">重复率</th>
    </tr></thead>
    <tbody>{rule_rows}</tbody>
  </table>
  </div>

  <h2>5. 各客户命中概览</h2>
  <div style="overflow-x:auto">
  <table class="data-table">
    <thead><tr>
      <th>客户</th><th style="text-align:right">总交易数</th><th style="text-align:right">风控命中</th>
      <th style="text-align:right">命中率</th>
      <th style="text-align:right;color:#ef4444">直接拦截</th>
      <th style="text-align:right;color:#f59e0b">送人审</th>
      <th style="text-align:right;color:#6b7280">空跑</th>
      <th>Top 3 规则</th>
    </tr></thead>
    <tbody>{customer_rows}</tbody>
  </table>
  </div>

  <h2>6. 重点客户详情</h2>
  {customer_detail_sections}

  <h2>7. 关键发现</h2>
  <div style="background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155;margin:12px 0">
    <ul style="padding-left:20px;line-height:2">
      <li><strong style="color:#f59e0b">HIT_LD_SPLICING (送人审)</strong> 是命中量最高的规则，14天命中 387 UV，日均约 28 笔。该规则为唯一的"送人审"类型，建议关注人审后续处理效率。</li>
      <li><strong style="color:#ef4444">HIT_VIRTUAL_CAMERA_DETECTED</strong> 是直接拦截中命中量最高的规则 (157 UV / 452 PV)，重复率 2.9x，说明同一用户反复尝试使用虚拟相机。Bybit 贡献了 87 UV (55%)。</li>
      <li><strong style="color:#8b5cf6">OVER_MAX_REVIEW_LIMIT</strong> 命中 111 UV / 348 PV，重复率 3.1x，表明被拒用户仍在持续尝试。4月13日出现异常峰值 (25 UV)，建议排查。</li>
      <li><strong>人脸关联类规则</strong> (MULTIPLE_CARD + MULTIPLE_BIRTHDAY) 合计 298 UV，是身份欺诈的典型信号。</li>
      <li><strong>空跑规则</strong> (HIT_IP_REJECT_HISTORY, HIT_DEVICE_TIMEZONE/COUNTRY/CLIENT_TZ_MISMATCH) 在当前两周 <span style="color:#10b981">未产生任何命中</span>，可能需要确认规则是否已上线或阈值设置是否合理。</li>
      <li><strong>Bybit</strong> 贡献了绝大多数风控命中 (约 1,010 / 1,378 = 73%)，但考虑到其交易量 (191K)，命中率仅 0.53%，属于正常范围。</li>
      <li><strong>高命中率客户</strong>: Cust_23795 (35.6%), Cust_20638 (24.5%), Cust_19403 (15.4%) 命中率显著偏高，建议重点关注这些客户的业务质量。</li>
    </ul>
  </div>

  <details>
    <summary>📄 取数 SQL 查询</summary>
    <pre>{sql_queries}</pre>
  </details>

  <div style="margin-top:40px;padding-top:16px;border-top:1px solid #334155;color:#64748b;font-size:12px;text-align:center">
    Risk Module Disposition Analysis | Generated by Vibe Track | 2026-04-15
  </div>
</div>

<script>
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#334155';

  new Chart(document.getElementById('dispTrend'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(date_labels)},
      datasets: {json.dumps(disp_datasets)}
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: '处置方式每日命中趋势 (UV)', color: '#f8fafc', font: {{ size: 15 }} }},
        legend: {{ position: 'top' }}
      }},
      scales: {{
        y: {{ beginAtZero: true, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ color: '#1e293b' }} }}
      }}
    }}
  }});

  new Chart(document.getElementById('ruleTrend'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(date_labels)},
      datasets: {json.dumps(rule_datasets)}
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: 'Top 规则每日命中趋势 (UV)', color: '#f8fafc', font: {{ size: 15 }} }},
        legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }}
      }},
      scales: {{
        y: {{ beginAtZero: true, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ color: '#1e293b' }} }}
      }}
    }}
  }});
</script>
</body>
</html>"""
    return html


def publish_to_oss(html_content):
    payload = {
        "channel": "oss_html",
        "oss_key": "risk-analysis/disposition/risk-module-disposition-20260401-20260414/index.html",
        "html": html_content,
    }
    result = subprocess.run(
        ["curl", "-s", f"{VT_BASE}/api/tools/deliver",
         "-H", f"Authorization: Bearer {VT_TOKEN}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout


if __name__ == "__main__":
    print("Building HTML report...")
    html = build_html()
    print(f"HTML size: {len(html)} bytes")

    print("Publishing to OSS...")
    result = publish_to_oss(html)
    print(f"Publish result: {result}")
