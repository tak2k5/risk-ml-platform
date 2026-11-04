#!/usr/bin/env bash
# Dựng toàn bộ hệ thống trên laptop bằng kind (Kubernetes trong Docker), miễn phí.
# Cần: docker, kind, kubectl, helm. Chạy từ thư mục credit/:  bash deploy/scripts/kind-up.sh
set -euo pipefail

CLUSTER=credit-risk
TAG=${TAG:-local}

echo "1) Tạo cụm kind (bỏ qua nếu đã có)"
kind get clusters | grep -qx "$CLUSTER" || kind create cluster --config deploy/kind/kind-config.yaml

echo "2) Cài ingress-nginx (để chia lưu lượng canary) và metrics-server (để HPA đọc được CPU)"
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/kind/deploy.yaml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml
# kind dùng chứng chỉ tự ký cho kubelet, metrics-server cần cờ này mới chạy được
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' || true
kubectl wait -n ingress-nginx --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=180s

echo "3) Build image và nạp vào cụm (không cần registry)"
[ -f models/model.joblib ] || PYTHONPATH=src:../riskops/src python -m credit_risk.train --synthetic 20000 --no-mlflow
docker build -t credit-risk:"$TAG" -f Dockerfile ..
kind load docker-image credit-risk:"$TAG" --name "$CLUSTER"

echo "4) Cài Helm chart"
helm upgrade --install cr deploy/helm/credit-risk --set image.tag="$TAG" --wait --timeout 5m
kubectl get pods,svc,hpa,ingress

echo "5) Gọi thử qua Ingress"
python scripts/smoke_test.py http://localhost:8080
