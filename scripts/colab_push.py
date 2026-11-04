"""Đẩy file báo cáo từ Colab lên GitHub, để không phải tải về rồi kéo vào Codespace.

    python ../scripts/colab_push.py rag/reports "RAG: kết quả trên Colab T4"

Cần biến môi trường GH_TOKEN: fine-grained token chỉ cho repo này, quyền Contents: Read and write
(ô cuối notebook lấy từ Colab Secrets). Token không bao giờ được in ra.

Chỉ thêm các file trong thư mục được chỉ định; .gitignore đã chặn dữ liệu, mô hình và báo cáo giả lập,
nên lệnh này không thể vô tình đẩy những thứ đó.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def token_url(url: str, token: str) -> str:
    """https://github.com/a/b.git -> https://x-access-token:<token>@github.com/a/b.git"""
    if not url.startswith("https://"):
        raise SystemExit(f"Remote phải là https, đang là: {url}")
    return "https://x-access-token:" + token + "@" + re.sub(r"^https://([^@/]*@)?", "", url)


def owner_of(url: str) -> str:
    m = re.search(r"github\.com[/:]([^/]+)/", url)
    return m.group(1) if m else "colab"


def hide(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def git(root: Path, *args: str, token: str = "", check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    out = hide(p.stdout + p.stderr, token)
    if check and p.returncode != 0:
        raise SystemExit(f"git {args[0]} lỗi:\n{out}")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    folder, message = argv
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        print("Chưa có GH_TOKEN trong Colab Secrets. Kết quả vẫn nằm trên Drive; tải về và đẩy bằng Codespace.")
        return 1
    root = Path(git(Path("."), "rev-parse", "--show-toplevel").strip())
    url = git(root, "remote", "get-url", "origin").strip()
    owner = owner_of(url)
    git(root, "config", "user.name", owner)
    git(root, "config", "user.email", f"{owner}@users.noreply.github.com")

    git(root, "add", "--", folder)
    staged = git(root, "diff", "--cached", "--name-only").split()
    if not staged:
        print(f"Không có file mới trong {folder} để đẩy.")
        return 0
    print("Sẽ đẩy:", *staged, sep="\n  ")
    git(root, "commit", "-q", "-m", message)
    push = token_url(url, token)
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    # bạn có thể đã đẩy thay đổi khác từ Codespace trong lúc notebook chạy: lấy về trước rồi mới đẩy
    git(root, "pull", "-q", "--rebase", push, branch, token=token)
    git(root, "push", "-q", push, f"HEAD:{branch}", token=token)
    print(f"Đã đẩy lên GitHub ({owner}, nhánh {branch}). Mở repo trên github.com để xem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
