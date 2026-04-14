# Bybit Liveness Detection — Key Findings

Based on past conversation search results, here are the key findings:

1. **Critical SC07 vulnerability (0% rejection rate):** The Bybit SC07 flow (`DOCUMENT_DATABASE_FACE`) had a **0% rejection rate** for two weeks (2026-03-10 to 2026-03-23), meaning liveness detection was completely ineffective at preventing any fraudulent transactions in that flow.

2. **Forged document IDV bypass investigation:** A research effort was initiated to investigate forged document IDV bypass detection, involving downloading ID document and liveness capture data, understanding IDV embedding logic (`id_back_vectorize`, `id_face`, `face`), testing processed vectors, and running detection logic specifically for Bybit data from 2026-03-12 to 2026-03-18.

3. **Log data scoping:** The `frontend-log` logstore in Aliyun SLS primarily contains Liveness Detection data (`LIVENESS_DETECTION_H5` / `LIVENESS_DETECTION_PC`) but does **not** contain Bybit customer data or IDV bizTypes — other logstores like `sdk-log` or `ekyc-frontend-risk-log` are needed for Bybit-specific fingerprint and IDV data.

---

*A weekly Bybit liveness report pipeline (`schedule-weekly-bybit-liveness`) is also in place, running every Monday morning.*
