#!/usr/bin/env bash
# Chạy một lần khi tạo Codespace (postCreateCommand): công cụ hệ thống + thư viện Python để chạy test.
# Chạy lại tay được: bash .devcontainer/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -qq && sudo apt-get install -y -qq make unzip > /dev/null
python -m pip install -q -U pip

# Codespaces không có GPU. Nếu repo dùng PyTorch, cài trước bản CPU (khoảng 200 MB) để pip không kéo bản CUDA
# (vài GB, tốn ổ đĩa mà không dùng được). Phần cần GPU chạy trên Colab như README hướng dẫn.
if grep -rqsE "^(torch|torchvision|ultralytics|sentence-transformers|transformers|faster-whisper)" \
     --include="requirements*.txt" . ; then
  python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

make install
echo
echo "Codespace đã sẵn sàng. Thử: make test"
