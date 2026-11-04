import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_push as cp  # noqa: E402


def test_token_url_va_che_token():
    url = cp.token_url("https://github.com/kiet/repo.git", "ghp_abc")
    assert url == "https://x-access-token:ghp_abc@github.com/kiet/repo.git"
    assert (
        cp.token_url("https://old@github.com/kiet/repo.git", "t") == "https://x-access-token:t@github.com/kiet/repo.git"
    )
    assert cp.hide(f"fatal: {url}", "ghp_abc") == "fatal: https://x-access-token:***@github.com/kiet/repo.git"
    assert cp.owner_of("https://github.com/kiet/repo.git") == "kiet"
    with pytest.raises(SystemExit):
        cp.token_url("git@github.com:kiet/repo.git", "t")


def test_khong_co_token_thi_khong_lam_gi(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert cp.main(["reports", "msg"]) == 1
    assert "GH_TOKEN" in capsys.readouterr().out


def test_day_chi_thu_muc_bao_cao(tmp_path, monkeypatch):
    run = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)  # noqa: E731
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    run("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    run("clone", "-q", str(remote), str(work), cwd=tmp_path)
    (work / "rag" / "reports").mkdir(parents=True)
    (work / "rag" / "data").mkdir()
    (work / "README.md").write_text("x")
    run("-c", "user.name=t", "-c", "user.email=t@t", "add", "README.md", cwd=work)
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=work)
    run("push", "-q", "origin", "main", cwd=work)
    (work / "rag" / "reports" / "ablation.md").write_text("| a |")
    (work / "rag" / "data" / "raw.pdf").write_text("không được đẩy")

    # remote cục bộ không phải https: thay hàm tạo URL để thử phần còn lại
    monkeypatch.setattr(cp, "token_url", lambda url, token: str(remote))
    monkeypatch.setenv("GH_TOKEN", "t0k")
    monkeypatch.chdir(work)
    assert cp.main(["rag/reports", "RAG: kết quả"]) == 0
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
                           capture_output=True, text=True).stdout.split()  # fmt: skip
    assert files == ["README.md", "rag/reports/ablation.md"]
    assert cp.main(["rag/reports", "lần 2"]) == 0  # không có gì mới: không lỗi
