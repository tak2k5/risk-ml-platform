# riskops: phần dùng chung của hai mô hình rủi ro

| Module | Dùng ở đâu |
|---|---|
| `drift.py`: PSI, KS, `summarize` cho một lô; `StreamDrift` cho cửa sổ trượt trên luồng | `credit/pipelines/monitor.py` (theo lịch), API tín dụng và consumer gian lận (liên tục) |
| `prom.py`: gauge `risk_feature_psi`, `risk_score_psi`, `risk_drift_alert` có nhãn `model` | API tín dụng, consumer gian lận, dashboard `monitoring/grafana/risk.json` |
| `registry.py`: ghi lần chạy, đăng ký phiên bản, alias champion/challenger, tải mô hình về | `fraud/src/fraudrt/train.py`, `fraud/src/fraudrt/promote.py`, nạp mô hình theo `MODEL_URI` |

Không cài như gói riêng: Makefile và Dockerfile thêm `riskops/src` vào `PYTHONPATH`.

```bash
PYTHONPATH=src python -m pytest
```
