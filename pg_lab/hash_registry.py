"""hash_registry.py — 生成 evidence/base_sha256.txt（v4：证据链自包含版）。

为什么要有这个文件：手工维护哈希会陈旧（v3 的 base_sha256.txt 里提交脚本的哈希
停留在 1415 行版本，而实际脚本已经变了）。这里一次算全、可复跑：

    python pg_lab/hash_registry.py

登记内容：
  1. 官方基线（包内副本 evidence/upstream/）——含包内路径、官方来源路径、行数、sha256；
  2. 提交脚本与补丁；
  3. 包内所有 evidence 文件的 sha256（防"证据文件被换过"）；
  4. 「补丁能否在包内直接 apply」的自检结果。
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EV = os.path.join(ROOT, "evidence")
UP = os.path.join(EV, "upstream")

FILES = [
    # (包内路径, 官方来源路径, 说明)
    (os.path.join(UP, "root_train_gpt.py"),
     "openai/parameter-golf @ main : train_gpt.py",
     "本方案的基线。提交脚本 = 这份文件 + evidence/submission_script.patch"),
    (os.path.join(UP, "record_2026-03-19_SlidingWindowEval_train_gpt.py"),
     "openai/parameter-golf @ main : records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py",
     "滑窗评测机制的出处（Matthew Li）。只借鉴机制，架构与它不同（它含 num_loops/lora_adapters）"),
    (os.path.join(UP, "record_2026-03-17_NaiveBaseline_train_gpt.py"),
     "openai/parameter-golf @ main : records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
     "朴素评测记录，用于三方对照"),
    (os.path.join(UP, "LICENSE_upstream.txt"),
     "openai/parameter-golf @ main : LICENSE",
     "MIT License (c) 2026 OpenAI —— 允许把 baseline 复制进本交付包"),
]


def sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def lines(p: str) -> int:
    try:
        return len(open(p, encoding="utf-8").read().splitlines())
    except UnicodeDecodeError:
        return -1


def main() -> int:
    out = []
    out.append("base_sha256.txt — 基线、产物与证据文件的 sha256 登记（v4）")
    out.append(f"由 pg_lab/hash_registry.py 生成；复跑：python pg_lab/hash_registry.py")
    out.append("=" * 78)
    out.append("")

    out.append("[1] 官方基线（已复制进交付包，MIT 许可；本目录随包交付，不依赖 _upstream/）")
    for path, src, note in FILES:
        if not os.path.exists(path):
            out.append(f"  !! 缺失：{path}")
            continue
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        out.append(f"  {rel}")
        out.append(f"    sha256 = {sha256(path)}")
        out.append(f"    lines  = {lines(path)}   bytes = {os.path.getsize(path)}")
        out.append(f"    来源   = {src}")
        out.append(f"    说明   = {note}")
    out.append("")

    sub = os.path.join(ROOT, "卢怡然_C2G_train_gpt.py")
    patch = os.path.join(EV, "submission_script.patch")
    out.append("[2] 提交脚本与补丁")
    out.append("  卢怡然_C2G_train_gpt.py")
    out.append(f"    sha256 = {sha256(sub)}")
    out.append(f"    lines  = {lines(sub)}   bytes = {os.path.getsize(sub)}")
    out.append("  evidence/submission_script.patch")
    out.append(f"    sha256 = {sha256(patch)}")
    out.append(f"    lines  = {lines(patch)}   bytes = {os.path.getsize(patch)}")
    out.append("    diff 头（fromfile/tofile 均为包内路径，可直接 apply）：")
    with open(patch, encoding="utf-8") as fh:
        head = [next(fh) for _ in range(3)]
    for h in head:
        out.append(f"      {h.rstrip()}")
    out.append("")

    # [3] 补丁可 apply 性自检：把包内基线复制出来，apply 后比对
    out.append("[3] 补丁可 apply 性自检（在包内路径上真跑一次）")
    import tempfile
    tmp_root = tempfile.mkdtemp(prefix="c2g_patch_")
    dst_dir = os.path.join(tmp_root, "evidence", "upstream")
    os.makedirs(dst_dir, exist_ok=True)
    dst_base = os.path.join(dst_dir, "root_train_gpt.py")
    with open(dst_base, "wb") as fh:
        fh.write(open(os.path.join(UP, "root_train_gpt.py"), "rb").read())
    dst_patch = os.path.join(tmp_root, "evidence", "submission_script.patch")
    os.makedirs(os.path.dirname(dst_patch), exist_ok=True)
    with open(dst_patch, "wb") as fh:
        fh.write(open(patch, "rb").read())

    applied_ok, applied_note = False, ""
    # 必须 -c core.autocrlf=false：Windows 默认 autocrlf=true 会把 apply 产物转成 CRLF，
    # 字节数与 sha256 都与提交脚本对不上（实测：66604 vs 65141 字节）。
    cmd = ["git", "-c", "core.autocrlf=false", "apply", "-p1", dst_patch]
    try:
        r = subprocess.run(cmd, cwd=tmp_root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        got = os.path.join(tmp_root, "卢怡然_C2G_train_gpt.py")
        if r.returncode == 0 and os.path.exists(got):
            same = sha256(got) == sha256(sub)
            applied_ok = same
            applied_note = (f"apply 成功，产物 {os.path.getsize(got)} 字节，"
                            f"sha256 {'与提交脚本一致' if same else '与提交脚本不一致'}")
        else:
            applied_note = (f"git apply 失败（returncode={r.returncode}）: "
                            f"{(r.stderr or r.stdout)[:200]}")
    except FileNotFoundError:
        applied_note = "本机无 git，跳过"
    out.append("    git -c core.autocrlf=false apply -p1 evidence/submission_script.patch")
    out.append(f"    结果：{'PASS' if applied_ok else '未通过'} —— {applied_note}")
    out.append("")

    out.append("[4] evidence/ 下其他证据文件的 sha256（防证据文件被换过）")
    skip = {"base_sha256.txt"}
    for name in sorted(os.listdir(EV)):
        p = os.path.join(EV, name)
        if os.path.isdir(p) or name in skip:
            continue
        out.append(f"  {name:<52} {sha256(p)}  ({os.path.getsize(p):>8} bytes)")
    out.append("")

    out.append("[5] 一句话结论")
    out.append("  提交脚本 = 包内官方基线（MIT，随包交付）+ 包内补丁；")
    out.append("  两者路径都在包内，拿到包即可独立复现，无需 _upstream/ 或外网。")

    text = "\n".join(out) + "\n"
    with open(os.path.join(EV, "base_sha256.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    return 0 if applied_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
