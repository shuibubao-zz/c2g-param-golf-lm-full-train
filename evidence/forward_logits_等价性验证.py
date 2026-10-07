"""forward_logits 等价性与因果性验证（v4，回应评审五问之「forward_logits 是否真产出逐位置 logits」）。

要证明三件事：
  [1] 形状：forward_logits(x) -> (bsz, seq_len, vocab)，未池化；
  [2] 数值：用它重算的 cross_entropy 均值与官方 forward() 的损失**逐位相同**
           （tie_embeddings True / False 各验一次）；
  [3] 因果：改 x[0, j] 只影响位置 j 及其之后，位置 < j 必须完全不变
           （这是滑窗评测按位置切分 s = max(wlen-stride, 0) 能成立的前提）。

为什么必须先随机化参数（本文件踩到的坑，一并留档）：
  baseline 的 CausalSelfAttention 有 `self.proj._zero_init = True`，lm_head 也被初始化为全零。
  在**未训练的初始点**上：
    - tie=False 时 logits 恒等于 0（lm_head 全零），改任何输入 token 输出都不动 -> 检验假阴性；
    - tie=True 时因为 proj 零初始化，位置之间没有耦合，改 x[0,j] 只有位置 j 自己变
      -> 「逐位置」检验平凡成立，测等于没测。
  所以这里对全部参数做一次非零随机初始化，再测因果性，检验才有意义。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT = os.path.join(ROOT, "卢怡然_C2G_train_gpt.py")

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


def make_model(tie: bool, randomized: bool) -> torch.nn.Module:
    torch.manual_seed(0)
    model = m.GPT(
        vocab_size=256, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=tie, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=1024.0, qk_gain_init=0.5,
    )
    if randomized:
        # 跳出 zero-init 退化点：给所有参数一个小的非零初始化
        torch.manual_seed(1)
        for p in model.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.02)
            elif p.dim() == 1:
                torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def main() -> int:
    print("forward_logits 等价性与因果性验证")
    print("=" * 74)
    bsz, seq, vocab = 2, 17, 256
    ok = True

    # ---------- [0] 先记录 zero-init 退化现象（这是本检验为何要随机化的原因） ----------
    print("[0] 对照：未训练初始点上的退化现象")
    m0 = make_model(tie=False, randomized=False)
    x = torch.randint(0, vocab, (bsz, seq))
    l1 = m0.forward_logits(x)
    x2 = x.clone()
    x2[0, 8] = (x2[0, 8] + 1) % vocab
    l2 = m0.forward_logits(x2)
    print(f"    tie=False 未随机化：logits 全零 = {bool((l1 == 0).all())}，"
          f"改输入后最大变化 = {float((l1 - l2).abs().max()):.3e}  <- 检验会假阴性")
    m1 = make_model(tie=True, randomized=False)
    l1 = m1.forward_logits(x)
    l2 = m1.forward_logits(x2)
    d = (l1 - l2).abs().sum(-1)[0]
    print(f"    tie=True  未随机化：改 x[0,8] 后只有位置 8 变（{float(d[8]):.3f}），"
          f"位置 9..16 最大变化 {float(d[9:].max()):.3e}  <- 因 proj 零初始化，无因果耦合")
    print()

    # ---------- [1][2][3] 正式检验 ----------
    for tie in (True, False):
        model = make_model(tie, randomized=True)
        torch.manual_seed(7)
        x = torch.randint(0, vocab, (bsz, seq))
        y = torch.randint(0, vocab, (bsz, seq))

        logits = model.forward_logits(x)
        with torch.no_grad():
            loss_official = model.forward(x, y)
        loss_recomputed = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="mean"
        )
        diff = float((loss_official.detach() - loss_recomputed.detach()).abs().max())

        shape_ok = tuple(logits.shape) == (bsz, seq, vocab)

        # 因果性：改 x[0, j] -> 位置 < j 必须完全不变；位置 j 必须变
        causal_ok = True
        detail = []
        for j in (0, 5, 8, 16):
            xx = x.clone()
            xx[0, j] = (xx[0, j] + 1) % vocab
            dd = (logits - model.forward_logits(xx)).abs().sum(-1).detach()[0]
            before = float(dd[:j].max()) if j > 0 else 0.0
            at = float(dd[j])
            causal_ok = causal_ok and (before < 1e-5) and (at > 1e-3)
            detail.append(f"j={j}: 前{max(j,1)}位最大Δ={before:.2e}, 本位Δ={at:.3f}")

        case_ok = shape_ok and (diff == 0.0) and causal_ok
        ok = ok and case_ok
        print(f"tie_embeddings={tie}（参数已随机化）")
        print(f"  [1] 形状 {tuple(logits.shape)} == ({bsz},{seq},{vocab}) -> {'OK' if shape_ok else 'BAD'}")
        print(f"  [2] 与官方 forward() 损失差 |Δ|max = {diff:.3e} -> {'OK' if diff == 0.0 else 'BAD'}"
              f"  ({float(loss_official.detach()):.8f} vs {float(loss_recomputed.detach()):.8f})")
        print(f"  [3] 因果性 -> {'OK' if causal_ok else 'BAD'}")
        for s in detail:
            print(f"        {s}")

    print("=" * 74)
    print(f"结论：{'PASS —— 逐位置、因果、与官方 forward 数值完全一致' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
