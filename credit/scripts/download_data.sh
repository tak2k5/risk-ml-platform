#!/usr/bin/env bash
# Tải dữ liệu Home Credit Default Risk từ Kaggle.
# Yêu cầu: đã tham gia cuộc thi (bấm "Join competition" trên Kaggle) và có ~/.kaggle/kaggle.json
# (hoặc biến môi trường KAGGLE_USERNAME, KAGGLE_KEY; trên Codespaces đặt làm Codespaces secret)
#   https://www.kaggle.com/competitions/home-credit-default-risk
# Dữ liệu chỉ để học, KHÔNG commit lên GitHub (đã chặn trong .gitignore).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
pip install -q kaggle
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data
cd data && (unzip -o application_train.csv.zip && rm application_train.csv.zip) 2>/dev/null || true
ls -lh application_train.csv
