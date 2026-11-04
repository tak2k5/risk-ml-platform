#!/usr/bin/env bash
# Triển khai canary cho phiên bản mới, tăng dần lưu lượng; lỗi thì quay lui.
#   bash deploy/scripts/canary.sh v2
set -euo pipefail
NEW_TAG=${1:?Cần tag mới, ví dụ v2}

docker build -t credit-risk:"$NEW_TAG" -f Dockerfile ..
kind load docker-image credit-risk:"$NEW_TAG" --name credit-risk

for WEIGHT in 10 50; do
  echo "Canary nhận ${WEIGHT}% lưu lượng"
  helm upgrade cr deploy/helm/credit-risk --reuse-values --wait \
    --set canary.enabled=true --set canary.image.tag="$NEW_TAG" --set canary.weight="$WEIGHT"
  # Kiểm tra: 50 yêu cầu phải thành công hết; nếu không thì tắt canary (quay lui)
  for i in $(seq 1 50); do
    if ! python scripts/smoke_test.py http://localhost:8080 > /dev/null; then
      echo "Lỗi ở mức ${WEIGHT}%, quay lui"
      helm upgrade cr deploy/helm/credit-risk --reuse-values --set canary.enabled=false
      exit 1
    fi
  done
done

echo "Ổn định, chuyển toàn bộ sang $NEW_TAG và tắt canary"
helm upgrade cr deploy/helm/credit-risk --reuse-values --wait --set image.tag="$NEW_TAG" --set canary.enabled=false
