"""Phát lại giao dịch theo thứ tự thời gian vào topic Redpanda/Kafka "transactions".

    python -m fraudrt.producer --data data/paysim.csv --rate 500        # 500 giao dịch/giây
    python -m fraudrt.producer --synthetic --rate 200 --limit 5000
Khóa của message = tài khoản gửi, nên mọi giao dịch của một tài khoản vào cùng partition, đúng thứ tự.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .data import load_paysim, make_synthetic


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paysim.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rate", type=float, default=200, help="giao dịch mỗi giây")
    ap.add_argument("--start", type=float, default=0.85, help="bắt đầu từ phần dữ liệu nào (0.85 = phần test)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topic", default="transactions")
    ap.add_argument("--brokers", default=os.environ.get("BROKERS", "localhost:19092"))
    a = ap.parse_args(argv)
    from confluent_kafka import Producer

    df = make_synthetic() if a.synthetic else load_paysim(a.data)
    df = df.iloc[int(len(df) * a.start) :]
    if a.limit:
        df = df.iloc[: a.limit]
    p = Producer({"bootstrap.servers": a.brokers, "linger.ms": 5, "enable.idempotence": True})
    t0 = time.perf_counter()
    for i, tx in enumerate(df.to_dict("records")):
        p.produce(a.topic, key=str(tx["orig"]), value=json.dumps(tx, default=float))
        p.poll(0)
        wait = t0 + (i + 1) / a.rate - time.perf_counter()  # giữ đều tốc độ
        if wait > 0:
            time.sleep(wait)
    p.flush(30)
    print(f"đã gửi {len(df)} giao dịch trong {time.perf_counter() - t0:.1f} giây")


if __name__ == "__main__":
    main()
