#!/usr/bin/env bash
# Tải PaySim (Kaggle: ealaxi/paysim1, ~470 MB sau giải nén) vào data/paysim.csv.
# Cần Kaggle API: đặt ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token), hoặc hai biến môi trường
# KAGGLE_USERNAME, KAGGLE_KEY (trên Codespaces: thêm làm Codespaces secret).
# Repo KHÔNG chứa dữ liệu: điều khoản Kaggle không cho phân phối lại, ai dùng tự tải.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
pip install -q kaggle
if [ -f data/paysim.csv ]; then echo "đã có data/paysim.csv"; exit 0; fi
kaggle datasets download -d ealaxi/paysim1 -p data --unzip
mv data/PS_*_log.csv data/paysim.csv
echo "xong: data/paysim.csv ($(wc -l < data/paysim.csv) dòng)"
