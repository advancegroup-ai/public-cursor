# 风控规则错误码定义 (Risk Control Error Code Definitions)

> 数据来源: `adv_guardian_data_core.ods_sg_guardian_risk_control_t_risk_execution_record`
> 最近更新: 2026-04-15

## 处置方式说明

| 处置方式 | 说明 |
|---------|------|
| **直接拦截** | 系统自动拒绝，不进入后续流程 |
| **送人审** | 转人工审核链路二次研判 |
| **空跑** | 仅记录观察，不影响审核结果（用于规则调优期） |

## 错误码清单

### 直接拦截规则

| Error Code | 风控报错原因 | 处置方式 |
|-----------|------------|---------|
| `HIT_FACE_BLACK_LIST` | 人脸黑名单 | 直接拦截 |
| `HIT_FACE_PUNISHMENT` | 人脸请求冷却中 | 直接拦截 |
| `HIT_FACE_FREQUENCY_LIMIT` | 人脸请求频率过高 | 直接拦截 |
| `OVER_MAX_HIT_LIMIT` | 人脸高频命中风控规则 | 直接拦截 |
| `OVER_MAX_REVIEW_LIMIT` | 人脸被多次人审拒绝 | 直接拦截 |
| `HIT_FACE_MULTIPLE_CARD` | 人脸关联多个证件（同证不同号） | 直接拦截 |
| `HIT_FACE_MULTIPLE_NAME` | 人脸关联多个姓名 | 直接拦截 |
| `HIT_FACE_MULTIPLE_BIRTHDAY` | 人脸关联多个生日 | 直接拦截 |
| `HIT_ID_FACE_MULTIPLE_NAME` | 证件人脸多个姓名 | 直接拦截 |
| `HIT_ID_FACE_MULTIPLE_ID_NUMBER` | 证件人脸多个身份证号 | 直接拦截 |
| `HIT_ID_FACE_MULTIPLE_BIRTHDAY` | 证件人脸多个生日 | 直接拦截 |
| `HIT_VIRTUAL_CAMERA_DETECTED` | 命中虚拟相机检测 | 直接拦截 |
| `HIT_INJECTION_ICON_DETECTED` | 环境图注入图标检测 | 直接拦截 |
| `HIT_RISKY_CAMERA` | 摄像头开机时间检测 | 直接拦截 |
| `HIT_LD_SPLICING` | 环境图多图层检测 | 送人审 |
| `HIT_ID_DUPLICATE` | 证件背面重复检测 | 直接拦截 |
| `HIT_FACE_DUPLICATE` | 活体远脸重复检测 | 直接拦截 |
| `HIT_CRED_BLACK_LIST` | 证件黑名单 | 直接拦截 |

> **注**: `HIT_RISKY_CAMERA` 在数据表中当前以 `HIT_RISKY_DEVICE` 出现，计划更名为 `HIT_RISKY_CAMERA`。

### 送人审规则

| Error Code | 风控报错原因 | 处置方式 |
|-----------|------------|---------|
| `HIT_LD_SPLICING` | 环境图多图层检测 | 送人审 |

### 空跑规则（仅记录，不影响审核结果）

| Error Code | 风控报错原因 | 处置方式 |
|-----------|------------|---------|
| `HIT_IP_REJECT_HISTORY` | IP 高危 | 空跑 |
| `HIT_DEVICE_TIMEZONE_MISMATCH` | 换了设备且时区也变了 | 空跑 |
| `HIT_DEVICE_COUNTRY_MISMATCH` | 换了设备且 IP 国家也变了 | 空跑 |
| `HIT_DEVICE_CLIENT_TZ_MISMATCH` | 换了设备且前端上报时区也变了 | 空跑 |

## 数据库中已观测的所有 hit_code

以下为 `risk_record` 表中 2026-03-01 ~ 04-14 期间实际出现过的所有 `hit_code` 值：

```
HIT_CRED_BLACK_LIST
HIT_FACE_BLACK_LIST
HIT_FACE_DUPLICATE
HIT_FACE_FREQUENCY_LIMIT
HIT_FACE_MULTIPLE_BIRTHDAY
HIT_FACE_MULTIPLE_CARD
HIT_FACE_MULTIPLE_NAME
HIT_FACE_PUNISHMENT
HIT_ID_DUPLICATE
HIT_ID_FACE_MULTIPLE_BIRTHDAY
HIT_ID_FACE_MULTIPLE_ID_NUMBER
HIT_ID_FACE_MULTIPLE_NAME
HIT_INJECTION_ICON_DETECTED
HIT_LD_SPLICING
HIT_RISKY_DEVICE
HIT_VIRTUAL_CAMERA_DETECTED
OVER_MAX_HIT_LIMIT
OVER_MAX_REVIEW_LIMIT
PASS
```

**空跑规则** (`HIT_IP_REJECT_HISTORY`, `HIT_DEVICE_TIMEZONE_MISMATCH`, `HIT_DEVICE_COUNTRY_MISMATCH`, `HIT_DEVICE_CLIENT_TZ_MISMATCH`) 在此期间 **未出现** 在数据中。

## 近期命中分析报告

- **报告链接**: [风控规则命中分析 (2026-04-08 ~ 04-14)](https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/risk-control/error-definitions/risk-hit-analysis-20260408-20260414/index.html)
