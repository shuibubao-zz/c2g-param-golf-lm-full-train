"""W=1024 真实配置对照分析（回应评审 P2-4，填补 AAR §三 缺口）。

问题：原实验训练窗口 512，只测了「评测窗口 1024（训练窗口之外）」是负结果；
「训练窗口 1024 + 评测窗口 1024」——官方的真实配置——没有测。
本脚本对 runs_w1024/ 的 3 个 seed（与 runs/ 完全相同的 token 预算，唯一变量是训练窗口）：
  1. 测 ρ(p)（W=1024）→ 解析重建整个协议族（STRIDES_1024）；
  2. 直接评测金标准交叉验证（naive + 3 档 stride，W=1024）；
  3. 汇总 3 seed 的 ΔBPB（相对朴素）与配对标准差，找拐点。

输出：analysis_w1024/analysis_seed*.json + analysis_w1024/summary_w1024.json + 控制台表。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import evalproto as EP
from analyze import (LN2, STRIDES_1024, direct_eval_naive, direct_eval_sliding,
                     load_model, measure_profile)
from train import load_bin

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis_w1024"
SEEDS = [1337, 42, 2026]
RUNS = HERE / "runs_w1024"


def analyze_one(seed: int) -> dict:
    torch.set_num_threads(5)
    model = load_model(str(RUNS / f"ckpt_seed{seed}.pt"))
    data = load_bin(str(HERE / "data" / "val.bin"))
    W = 1024
    prof, meta = measure_profile(model, data, W, 64, 32)
    rows = EP.sweep_protocols(prof, W, STRIDES_1024, data.numel())
    naive_bpb = float(prof.mean()) / LN2

    # 金标准直接评测（W=1024）
    checks = []
    for S in (512, 256, 128):
        pred = EP.protocol_bpb(prof, W, S, data.numel())
        actual = direct_eval_sliding(model, data, W, S, 32)
        checks.append({"stride": S, "reconstructed_bpb": pred["bpb"], "direct_bpb": actual,
                       "abs_diff": abs(pred["bpb"] - actual)})
        print(f"  [seed {seed}] 交叉验证 W=1024 S={S}: ρ重建={pred['bpb']:.6f} 直接={actual:.6f} "
              f"Δ={abs(pred['bpb']-actual):.2e}", flush=True)
    naive_direct = direct_eval_naive(model, data, W, 32)
    checks.append({"stride": 1024, "kind": "pure_naive_partition",
                   "reconstructed_bpb": naive_bpb, "direct_bpb": naive_direct,
                   "abs_diff": abs(naive_bpb - naive_direct)})

    res = {"seed": seed, "window": W, "val_bytes": int(data.numel()),
           "naive_bpb": naive_bpb, "meta": meta,
           "protocols": rows, "profile_head": prof[:16].tolist(), "direct_check": checks}
    OUT.mkdir(exist_ok=True)
    p = OUT / f"analysis_seed{seed}.json"
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  [seed {seed}] -> {p}", flush=True)
    return res


def main() -> int:
    import multiprocessing as mp
    with mp.Pool(3) as pool:
        results = pool.map(analyze_one, SEEDS)

    # 汇总：每个 stride 的 ΔBPB（相对各自 naive），配对均值/标准差
    summary = {"seeds": SEEDS, "strides": [], "naive_bpb_mean": float(np.mean([r["naive_bpb"] for r in results])),
               "max_direct_check_diff": max(c["abs_diff"] for r in results for c in r["direct_check"])}
    strides = sorted({row["stride"] for r in results for row in r["protocols"]}, reverse=True)
    for S in strides:
        deltas, costs = [], []
        for r in results:
            row = next(x for x in r["protocols"] if x["stride"] == S)
            deltas.append(row["bpb"] - r["naive_bpb"])
            costs.append(row["cost_multiplier"])
        summary["strides"].append({
            "stride": S, "cost_mult": round(1024 / S, 2),
            "delta_bpb_mean": float(np.mean(deltas)),
            "delta_bpb_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "all_deltas": [round(d, 6) for d in deltas],
        })
    summary["strides"].sort(key=lambda x: x["cost_mult"])
    (OUT / "summary_w1024.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== W=1024 训练 + W=1024 评测（真实配置对照）=====")
    print(f"naive BPB 均值 = {summary['naive_bpb_mean']:.5f}  "
          f"ρ重建 vs 直接评测最大误差 = {summary['max_direct_check_diff']:.2e} BPB")
    print(f"{'代价':>6} {'stride':>7} {'ΔBPB 均值':>12} {'配对std':>10}  各seed")
    for s in summary["strides"]:
        print(f"{s['cost_mult']:>5}x {s['stride']:>7} {s['delta_bpb_mean']:>12.6f} "
              f"{s['delta_bpb_std']:>10.6f}  {s['all_deltas']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
