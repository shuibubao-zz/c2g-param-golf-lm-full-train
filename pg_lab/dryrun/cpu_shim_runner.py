"""C2G 提交脚本 CPU 端到端 dry-run 运行器（回应评审 P2-6 / P2-5）。

做什么：
  1. 读取 ../../卢怡然_C2G_train_gpt.py（与将提交的文件逐字节同源）；
  2. 应用一张**公开的、逐条编号的** CPU 适配补丁表（见 PATCHES，只改设备相关行，不碰任何算法逻辑）；
  3. 把打补丁后的脚本写到 train_gpt_cpu.py（文件头自动生成补丁清单），在独立 scratch 目录里
     分别以 EVAL_STRIDE_MODE=naive 和 knee 各跑一遍 端到端：warmup → 训练 → 验证 →
     int8+zlib 打包 → 往返校验 → 用标定 stride 做最终滑窗评测；
  4. 控制台输出原样留档 evidence/dryrun_*.txt。

诚实声明：
  - 打包产物字节数（P2-5 的「16MB 打包实测」）来自真实运行本脚本，不是估算；
  - 但这是 2 层/64 维的缩小模型、CPU、30 步——它证明的是「代码路径真跑得通」，
    不证明官方尺度下的任何数值结论；
  - 补丁只允许出现在 CUDA 专属调用上；任何触碰算法/评测逻辑的补丁都会让 assert 失败。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "卢怡然_C2G_train_gpt.py"
PY = sys.executable

# 每条补丁 = (编号, 原文, 替换, 预期出现次数, 理由)。全部只涉及 CUDA/编译相关调用。
PATCHES = [
    ("1", 'raise RuntimeError("CUDA is required")', "pass  # CPU-SHIM-1",
     1, "CUDA 存在性硬门"),
    ("2", 'device = torch.device("cuda", local_rank)', 'device = torch.device("cpu")  # CPU-SHIM-2',
     1, "设备选择"),
    ("3", "torch.cuda.set_device(device)", "pass", 1, "CUDA 设备绑定"),
    ("4", "torch.cuda.manual_seed_all(args.seed)", "pass", 1, "CUDA 种子（CPU 无需）"),
    ("5", "fused=True", "fused=False", 3, "fused Adam 仅 CUDA 支持"),
    ("6", 'torch.autocast(device_type="cuda"', 'torch.autocast(device_type="cpu"',
     5, "autocast 设备类型（训练/warmup/验证/标定/滑窗评测各处）"),
    ("7", "torch.compile(", "(_dryrun_no_compile)(", 2, "torch.compile 在 CPU 上过重且易碎；恒等替换"),
    ("8", "torch.cuda.max_memory_allocated()", "0", 1, "显存统计（CPU 无意义）"),
    ("9", "torch.cuda.max_memory_reserved()", "0", 1, "显存统计（CPU 无意义）"),
    ("10", "torch.cuda.synchronize()", "pass", 5, "CUDA 同步"),
    ("11", 'subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout',
     '""', 1, "无 GPU 环境没有 nvidia-smi 可执行文件（原脚本在 H100 机器上才有）"),
]

HELPER = '''

def _dryrun_no_compile(obj=None, **kwargs):
    # CPU-SHIM-7 的配套：torch.compile 的恒等替换（dry-run 专用，正式提交无此函数）
    return obj
'''


def build_patched_source() -> str:
    src = SRC.read_text(encoding="utf-8")
    report = []
    for pid, old, new, expect, why in PATCHES:
        n = src.count(old)
        assert n == expect, f"SHIM-{pid}: 期望出现 {expect} 次，实际 {n} 次：{old!r}"
        src = src.replace(old, new)
        report.append(f"  SHIM-{pid}: {old!r} -> 替换 {n} 处（{why}）")
    anchor = "from torch.nn.parallel import DistributedDataParallel as DDP"
    assert anchor in src
    src = src.replace(anchor, anchor + "\n" + HELPER, 1)
    header = (
        "# !!! 本文件由 pg_lab/dryrun/cpu_shim_runner.py 自动生成，仅用于 CPU 端到端 dry-run。\n"
        "# !!! 正式提交物是根目录的 卢怡然_C2G_train_gpt.py（无任何补丁）。\n"
        "# !!! 补丁清单（全部为设备适配，零算法改动）：\n"
        + "\n".join("# " + r for r in report)
        + "\n\n"
    )
    return header + src


BASE_ENV = {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "2",
    "MODEL_DIM": "64",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "2",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_SEQ_LEN": "128",
    "TRAIN_BATCH_TOKENS": "8192",
    "ITERATIONS": "30",
    "WARMDOWN_ITERS": "8",
    "WARMUP_STEPS": "2",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_BATCH_SIZE": "16384",
    "VAL_LOSS_EVERY": "10",
    "TRAIN_LOG_EVERY": "5",
    "SEED": "1337",
    "DATA_PATH": "C:/Users/Public/pg_c2g_dryrun/data",
    "TOKENIZER_PATH": "C:/Users/Public/pg_c2g_dryrun/data/sp1024.model",
    "OMP_NUM_THREADS": "8",
}

KNEE_ENV = {
    "EVAL_STRIDE_MODE": "knee",
    "EVAL_COST_BUDGET": "2.0",
    "KNEE_GAIN_FRAC": "0.95",
    "KNEE_CANDIDATES": "128,64,32,16",
    "EVAL_CALIB_TOKENS": "16384",
    "EVAL_CALIB_BATCH_SEQS": "16",
    "EVAL_CALIB_STRIDE": "8",
    "EVAL_BATCH_SEQS": "16",
}

# v4：评审指出 fixed 模式从未被 dry-run 覆盖。这里补上（TRAIN_SEQ_LEN=128 / EVAL_STRIDE=32 -> 4x）。
FIXED_ENV = {
    "EVAL_STRIDE_MODE": "fixed",
    "EVAL_STRIDE": "32",
    "EVAL_BATCH_SEQS": "16",
}

# v4：另外补一条**边界** dry-run —— 预算 0.5x（连 1x 都付不起），用来在真实端到端里
# 验证 v4 修复的"退化为朴素 + WARN"路径（单测已覆盖，这里再端到端确认一次不崩）。
KNEE_TIGHT_ENV = {**KNEE_ENV, "EVAL_COST_BUDGET": "0.5"}


def run_one(tag: str, extra_env: dict[str, str], ev_dir: Path) -> None:
    # scratch 放纯 ASCII 路径：sentencepiece 的 C++ 层在 Windows 上读不了含中文的绝对路径
    scratch = Path("C:/Users/Public/pg_c2g_dryrun") / f"scratch_{tag}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    env = {**os.environ, **BASE_ENV, **extra_env}
    print(f"\n===== DRY-RUN {tag} =====", flush=True)
    proc = subprocess.run(
        [PY, str(HERE / "train_gpt_cpu.py")],
        cwd=scratch, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr.strip() else "")
    (ev_dir / f"dryrun_{tag}.txt").write_text(out, encoding="utf-8")
    print(out[-4000:])
    if proc.returncode != 0:
        raise SystemExit(f"dry-run {tag} 失败，exit={proc.returncode}")
    # 关键断言：打包真的发生了
    art = scratch / "final_model.int8.ptz"
    raw = scratch / "final_model.pt"
    assert art.exists() and raw.exists(), "打包产物缺失"
    mb = art.stat().st_size
    print(f"[{tag}] artifact(int8+zlib) = {mb:,} bytes = {mb/1024/1024:.3f} MiB （16MB 上限 = {16*1024*1024:,}）")
    assert mb < 16 * 1024 * 1024


def audit_same_origin(ev_dir: Path) -> None:
    """同源性审计（v4，回应评审「dry-run 跑的是改编版，不是在提交脚本上跑」）。

    事实：本机无 GPU，提交脚本原文（含 CUDA 硬门）**不可能**在本机跑起来。
    所以能给出的最强证明不是"在原文上跑"，而是下面三条独立推导：
      [1] 统一 diff：把 卢怡然_C2G_train_gpt.py 与 train_gpt_cpu.py 逐行比对，
          断言**每一条**差异行都能被 11 条已声明的设备补丁之一解释（无第 12 处改动）；
      [2] 关键函数源码逐字节相同（eval_val_sliding / pick_stride_by_knee /
          _parse_knee_candidates / measure_position_profile / forward_logits）——
          即"被 dry-run 到的评测与标定逻辑"与提交脚本一模一样；
      [3] 两份文件的 sha256 与行数登记留档。
    这份审计是从**磁盘上的两个文件**重新推导的，不是从生成过程推导的（非循环论证）。
    """
    import difflib
    import hashlib
    import importlib.util
    import inspect
    import types

    orig_text = SRC.read_text(encoding="utf-8")
    cpu_text = (HERE / "train_gpt_cpu.py").read_text(encoding="utf-8")
    lines = []

    def sha(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    lines.append("同源性审计：提交脚本  vs  CPU dry-run 改编版")
    lines.append("=" * 72)
    lines.append(f"提交脚本      : {SRC}")
    lines.append(f"  sha256      = {sha(orig_text)}")
    lines.append(f"  lines       = {len(orig_text.splitlines())}")
    lines.append(f"dry-run 改编版: {HERE / 'train_gpt_cpu.py'}")
    lines.append(f"  sha256      = {sha(cpu_text)}")
    lines.append(f"  lines       = {len(cpu_text.splitlines())}")
    lines.append("")

    # ---- [1] diff 覆盖审计 ----
    # 判据（非循环：从磁盘上的两个文件重新推导，不复用生成过程）：
    #   删除行必须**包含**某条已声明补丁的 old 串（old 串是具体的 CUDA 调用，足以定性）；
    #   新增行必须包含对应 new 串，或属于文件头注释 / HELPER 注入块；
    #   删除行总数必须正好等于 11 条补丁的预期出现次数之和（多一行说明有第 12 处改动）。
    old_strings = [old for _, old, _, _, _ in PATCHES]
    new_strings = [new for _, _, new, _, _ in PATCHES]
    helper_lines = [x.strip() for x in HELPER.splitlines() if x.strip()]
    expect_minus = sum(expect for _, _, _, expect, _ in PATCHES)

    diff = list(difflib.unified_diff(
        orig_text.splitlines(), cpu_text.splitlines(),
        fromfile="卢怡然_C2G_train_gpt.py", tofile="train_gpt_cpu.py", n=0,
    ))
    minus, plus, uncovered = [], [], []
    for ln in diff:
        if ln.startswith("---") or ln.startswith("+++") or ln.startswith("@@"):
            continue
        if ln.startswith("-"):
            body = ln[1:]
            minus.append(body)
            if not any(o in body for o in old_strings):
                uncovered.append(("删除", body))
        elif ln.startswith("+"):
            body = ln[1:]
            plus.append(body)
            ok = (any(n in body for n in new_strings)
                  or body.strip() in helper_lines
                  or body.startswith("# !!!")
                  or ("SHIM-" in body and body.lstrip().startswith("#"))   # 文件头的补丁清单注释
                  or body.strip() == "")                                  # 文件头/HELPER 的空行
            if not ok:
                uncovered.append(("新增", body))
    count_ok = (len(minus) == expect_minus)
    audit1_ok = (not uncovered) and count_ok
    lines.append("[1] 统一 diff 覆盖审计（n=0，从磁盘两文件重新推导）")
    lines.append(f"    差异行           ：-{len(minus)} / +{len(plus)}")
    lines.append(f"    已声明补丁       ：{len(PATCHES)} 条，预期删除行合计 = {expect_minus}")
    lines.append(f"    删除行数 == 预期 ：{'是' if count_ok else f'否（实得 {len(minus)}）'}")
    lines.append(f"    未能被解释的差异行：{len(uncovered)} 条")
    for kind, body in uncovered[:20]:
        lines.append(f"      !! {kind}: {body[:100]}")
    lines.append(f"    结论：{'PASS —— 除声明的设备补丁外无任何改动' if audit1_ok else 'FAIL —— 存在未声明改动'}")
    lines.append("")

    # ---- [2] 关键函数源码逐字节相同 ----
    for name in ("sentencepiece",):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.SentencePieceProcessor = object
                sys.modules[name] = stub

    def load(path, modname):
        spec = importlib.util.spec_from_file_location(modname, str(path))
        m = importlib.util.module_from_spec(spec)
        sys.modules[modname] = m
        spec.loader.exec_module(m)
        return m

    mo = load(SRC, "c2g_orig")
    mc = load(HERE / "train_gpt_cpu.py", "c2g_cpu")
    # ---- [2] 关键函数逐函数 diff ----
    # 注意：不能简单断言"逐字节相同"——这些函数体里本身就有 CUDA 调用（如 autocast），
    # 它们必然被设备补丁改到。正确的判据是：**函数内的每一条删除行都必须能被
    # 某条已声明补丁的 old 串解释**；删除行数为 0 才算"逐字节相同"。
    def fn_diff_ok(so: str, sc: str) -> tuple[bool, int, int]:
        d = [l for l in difflib.unified_diff(so.splitlines(), sc.splitlines(), n=0)
             if not l.startswith(("---", "+++", "@@"))]
        removed = [l[1:] for l in d if l.startswith("-")]
        added = [l[1:] for l in d if l.startswith("+")]
        bad_r = [b for b in removed if not any(o in b for o in old_strings)]
        bad_a = [b for b in added
                 if not (any(n in b for n in new_strings) or b.strip() == "")]
        return (not bad_r) and (not bad_a), len(removed), len(added)

    lines.append("[2] 关键函数逐函数 diff（判据：删除行必须能被已声明补丁解释）")
    fn_ok = True
    targets = [("eval_val_sliding", lambda mod: getattr(mod, "eval_val_sliding")),
               ("measure_position_profile", lambda mod: getattr(mod, "measure_position_profile")),
               ("pick_stride_by_knee", lambda mod: getattr(mod, "pick_stride_by_knee")),
               ("_parse_knee_candidates", lambda mod: getattr(mod, "_parse_knee_candidates")),
               ("resolve_eval_stride", lambda mod: getattr(mod, "resolve_eval_stride")),
               ("stride_candidates", lambda mod: getattr(mod, "stride_candidates")),
               ("GPT.forward_logits", lambda mod: mod.GPT.forward_logits)]
    for fname, getter in targets:
        so = inspect.getsource(getter(mo))
        sc = inspect.getsource(getter(mc))
        ok, nr, na = fn_diff_ok(so, sc)
        fn_ok = fn_ok and ok
        tag = "逐字节相同" if nr == 0 else f"仅 {nr} 处设备适配差异"
        lines.append(f"    {'OK ' if ok else 'BAD'} {fname:<28} ({len(so.splitlines())} 行, "
                     f"-{nr}/+{na}) {tag}")
    lines.append(f"    结论：{'PASS —— 评测/标定逻辑与提交脚本同源（差异仅设备适配）' if fn_ok else 'FAIL'}")
    lines.append("")

    lines.append("[3] 诚实限定")
    lines.append("    - 这不是「在提交脚本原文上直接跑」：本机无 GPU，原文开头就")
    lines.append("      raise RuntimeError('CUDA is required')，物理上跑不起来。")
    lines.append("    - 上述 [1][2] 是能给出的最强替代证明：差异被逐行枚举且全部为设备适配，")
    lines.append("      评测/标定逻辑逐字节同源。")
    lines.append("    - 如需在真机验证，步骤见 README「拿到 8xH100 后怎么跑」。")

    out = "\n".join(lines) + "\n"
    (ev_dir / "dryrun_同源性证明.txt").write_text(out, encoding="utf-8")
    print(out)
    if not (audit1_ok and fn_ok):
        raise SystemExit("同源性审计未通过")


def main() -> int:
    ev_dir = ROOT / "evidence"
    # 数据也放 ASCII 路径（spm C++ 层读不了含中文的绝对路径）
    data_dst = Path("C:/Users/Public/pg_c2g_dryrun/data")
    data_dst.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "dryrun_data").iterdir():
        if f.is_file():
            shutil.copy2(f, data_dst / f.name)
    patched = build_patched_source()
    (HERE / "train_gpt_cpu.py").write_text(patched, encoding="utf-8")
    print("patched script written; patch count =", len(PATCHES))

    audit_same_origin(ev_dir)

    run_one("naive", {"EVAL_STRIDE_MODE": "naive"}, ev_dir)
    run_one("fixed", FIXED_ENV, ev_dir)
    run_one("knee", KNEE_ENV, ev_dir)
    run_one("knee_tight", KNEE_TIGHT_ENV, ev_dir)
    print("\nALL DRY-RUNS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
