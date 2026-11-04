#!/usr/bin/env bash
# Cài kind, kubectl, Helm, k6 vào Codespace cho Bước 1.5 (Kubernetes và đo tải). Chỉ cần chạy một lần:
#   bash .devcontainer/k8s-tools.sh
# Phiên bản cố định để kết quả lặp lại được.
set -euo pipefail
KIND=v0.24.0 KUBECTL=v1.31.2 HELM=v3.16.2 K6=v0.54.0
tmp=$(mktemp -d)
curl -fsSL -o "$tmp/kind" "https://github.com/kubernetes-sigs/kind/releases/download/$KIND/kind-linux-amd64"
curl -fsSL -o "$tmp/kubectl" "https://dl.k8s.io/release/$KUBECTL/bin/linux/amd64/kubectl"
curl -fsSL "https://get.helm.sh/helm-$HELM-linux-amd64.tar.gz" | tar -xz -C "$tmp" linux-amd64/helm
curl -fsSL "https://github.com/grafana/k6/releases/download/$K6/k6-$K6-linux-amd64.tar.gz" | tar -xz -C "$tmp"
sudo install -m 0755 "$tmp/kind" "$tmp/kubectl" "$tmp/linux-amd64/helm" "$tmp/k6-$K6-linux-amd64/k6" /usr/local/bin/
rm -rf "$tmp"
kind version && kubectl version --client && helm version --short && k6 version
