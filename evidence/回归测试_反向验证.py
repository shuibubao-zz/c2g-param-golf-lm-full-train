"""回归测试的反向验证（v4）：证明新加的 4 条用例**真的能抓到**旧 bug。

为什么需要这个文件：
  一条永远不会失败的测试等于没有测试。这里把修复前的旧实现原样抄回来，
  用**完全相同**的断言去跑，必须看到 FAIL。如果它们也 PASS，说明新用例是恒真空转。

用法：
  python evidence/回归测试_反向验证.py
期望输出：4 条 "旧实现 -> 被抓到(FAIL)" + 0 条 "旧实现也 PASS（测试用例无效！）"
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT = os.path.join(ROOT, "卢怡然_C2G_train_gpt.py")


def load_submission():
    import types
    for name in ("sentencepiece",):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.SentencePieceProcessor = object
                sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("c2g_submit", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["c2g_submit"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 修复前的旧实现
def old_pick_stride_by_knee(prof, window, cost_budget, gain_frac, candidates):
    """v3 的旧实现：else 分支返回 rows[-1]（最贵候选），且没有 budget_ok 标志。"""
    p = prof.detach().to(torch.float64)
    ln2 = math.log(2.0)
    rows = []
    naive = float(p.mean()) / ln2
    for s in candidates:
        steady = float(p[window - s:].mean()) / ln2
        rows.append({"stride": s, "cost": window / s, "gain": naive - steady, "bpb": steady})
    best_gain = max(r["gain"] for r in rows)
    target = gain_frac * best_gain
    affordable = [r for r in rows if r["cost"] <= cost_budget + 1e-9]
    eligible = [r for r in affordable if r["gain"] >= target]
    if eligible:
        chosen = max(eligible, key=lambda r: r["stride"])
    elif affordable:
        chosen = max(affordable, key=lambda r: r["gain"])
    else:
        chosen = rows[-1]          # <-- 旧 bug：stride 最小 = 代价最大，且突破预算
    return chosen["stride"], {"chosen": chosen, "naive_bpb": naive,
                              "best_gain": best_gain, "all": rows}


def old_parse_knee_candidates(raw, window, stride_min):
    """v3 的旧实现：只过滤 > window，0/负数放行；全非法 -> 空列表。"""
    return sorted({int(s) for s in raw.split(",") if int(s) <= window}, reverse=True)


def make_profile(window: int, kind: str) -> torch.Tensor:
    p = torch.arange(window, dtype=torch.float64)
    if kind == "flat":
        return torch.ones(window, dtype=torch.float64) * 2.0
    if kind == "fast_saturate":
        return 2.0 - 0.5 * torch.clamp(p / 128.0, max=1.0)
    if kind == "slow":
        return 2.0 - 0.5 * (p / (window - 1.0))
    raise ValueError(kind)


def main() -> int:
    m = load_submission()
    W = 1024
    cands = m.stride_candidates(W, 64)
    prof = make_profile(W, "fast_saturate")

    checks = []

    # --- 用例子 1：预算 <1x 应退化为朴素 ---
    def new_behavior_1():
        s, info = m.pick_stride_by_knee(prof, W, 0.5, 0.95, cands)
        assert info["budget_ok"] is False
        assert s == W

    def old_behavior_1():
        s, info = old_pick_stride_by_knee(prof, W, 0.5, 0.95, cands)
        assert info.get("budget_ok") is False      # 旧实现连这个字段都没有 -> KeyError/False
        assert s == W

    checks.append(("预算 <1x 退化为朴素", new_behavior_1, old_behavior_1))

    # --- 用例子 2：不变式扫描，代价永不突破预算 ---
    def scan(pick):
        for w in (256, 512, 1024):
            cs = m.stride_candidates(w, 32)
            for kind in ("flat", "fast_saturate", "slow"):
                pr = make_profile(w, kind)
                for budget in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0):
                    for gf in (0.0, 0.5, 0.95, 1.0):
                        s, _ = pick(pr, w, budget, gf, cs)
                        assert w / s <= max(budget, 1.0) + 1e-9, (
                            f"W={w} kind={kind} budget={budget} gf={gf} -> stride={s} cost={w/s}")

    checks.append(("代价永不突破预算（72 组合扫描）",
                   lambda: scan(m.pick_stride_by_knee),
                   lambda: scan(old_pick_stride_by_knee)))

    # --- 用例子 3：非正数/超窗被过滤 ---
    def new_behavior_3():
        assert m._parse_knee_candidates("128,0,-5,64", W, 64) == [128, 64]
        assert m._parse_knee_candidates("4096,2048,128", W, 64) == [128]

    def old_behavior_3():
        assert old_parse_knee_candidates("128,0,-5,64", W, 64) == [128, 64]
        assert old_parse_knee_candidates("4096,2048,128", W, 64) == [128]

    checks.append(("非正数/超窗被过滤", new_behavior_3, old_behavior_3))

    # --- 用例子 4：全非法回退自动枚举（旧版会得到空列表 -> 选择器 max() 崩溃） ---
    def new_behavior_4():
        got = m._parse_knee_candidates("0,-1,9999", W, 64)
        assert got == m.stride_candidates(W, 64) and len(got) > 0
        s, _ = m.pick_stride_by_knee(prof, W, 4.0, 0.95, got)
        assert 1 <= s <= W

    def old_behavior_4():
        got = old_parse_knee_candidates("0,-1,9999", W, 64)
        assert len(got) > 0                       # 旧实现得到 [-1, 0]... 其实非空
        # 真正致命的是下一步：0 作为 stride -> 除零；这里显式复现
        s, _ = old_pick_stride_by_knee(prof, W, 4.0, 0.95, got)
        assert 1 <= s <= W

    checks.append(("全非法 -> 回退自动枚举（不崩溃）", new_behavior_4, old_behavior_4))

    print("反向验证：新实现必须 PASS，旧实现必须 FAIL")
    print("-" * 72)
    bad = 0
    for name, new_fn, old_fn in checks:
        # 新实现
        try:
            new_fn()
            new_ok = True
            new_msg = ""
        except AssertionError as e:
            new_ok = False
            new_msg = str(e)[:100]
        except Exception as e:                      # noqa: BLE001
            new_ok = False
            new_msg = f"{type(e).__name__}: {str(e)[:100]}"
        # 旧实现
        try:
            old_fn()
            old_ok = True
            old_msg = ""
        except AssertionError as e:
            old_ok = False
            old_msg = str(e)[:100]
        except Exception as e:                      # noqa: BLE001
            old_ok = False
            old_msg = f"{type(e).__name__}: {str(e)[:100]}"

        verdict = "OK" if (new_ok and not old_ok) else "!! 测试用例无效 !!"
        if not (new_ok and not old_ok):
            bad += 1
        print(f"[{verdict}] {name}")
        print(f"        新实现: {'PASS' if new_ok else 'FAIL ' + new_msg}")
        print(f"        旧实现: {'PASS（未能抓到 bug）' if old_ok else 'FAIL -> ' + old_msg}")

    print("-" * 72)
    print(f"结论：{len(checks) - bad}/{len(checks)} 条用例能真正区分新旧实现")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
