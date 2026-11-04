"""Gọi thử API đang chạy: /health rồi /predict với hồ sơ mẫu. Dùng trong CI và sau khi deploy.

python scripts/smoke_test.py http://localhost:8000
"""

import json
import sys
import urllib.request

EXAMPLE = {
    "CNT_CHILDREN": 1,
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 600000,
    "AMT_ANNUITY": 30000,
    "AMT_GOODS_PRICE": 540000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "DAYS_ID_PUBLISH": -3000,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55,
    "REGION_RATING_CLIENT": 2,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Laborers",
}


def call(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main(base: str) -> None:
    health = call(f"{base}/health")
    assert health["status"] == "ok", health
    pred = call(f"{base}/predict", EXAMPLE)
    assert 0 <= pred["probability_default"] <= 1, pred
    print("OK", json.dumps(pred, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
