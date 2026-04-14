# Binance (9929123517) Customer Data Analysis

**Period**: April 1–13, 2026
**Report URL**: https://prod-guardian-cv.oss-ap-southeast-5.aliyuncs.com/ekyc/binance/customer-overview-20260414/index.html

## Summary

- **Customer ID**: 9929123517
- **Solution**: LIVENESS_DETECTION (standalone liveness, not full IDV)
- **Region**: PHL (Philippines only)
- **Daily Volume**: ~594 PV / ~256 UV per day
- **PV Pass Rate**: ~35% overall (Mobile ~57%, PC ~22%)
- **UV Pass Rate**: ~62% overall

## Key Findings

1. PC H5 liveness pass rate is extremely low (~22%) vs Mobile (~57%)
2. Rising drop-off trend: sessions with no face upload jumped from ~23/day to ~223/day
3. No risk rules configured — all rejections come from liveness modules
4. Top rejectors: ENSEMBLER_V2_H5_MODEL (62% of rejects), FACE_FORGERY (52%)
5. OBS_INJECTION + CAMERA_FP_PC pattern in ~20% of rejects (PC injection attacks)

## Recommendations

1. **Urgent**: Investigate drop-off spike (Apr 11-13)
2. **Short-term**: Review PC H5 thresholds — may be too aggressive for PHL
3. **Short-term**: Analyze ENSEMBLER_V2_H5_MODEL false rejections on PC
4. **Monitor**: UV pass rate trending down (67% → 58.5%)
