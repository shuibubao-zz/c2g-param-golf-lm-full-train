"""
build_submission.py — 由官方 baseline train_gpt.py **程序化生成**我们的提交脚本。

为什么不手写：手写容易与上游静默漂移，也无法向评审证明"改了哪几行"。
这里把改动写成精确的锚点插入，产出：
    - ../卢怡然_C2G_train_gpt.py      最终提交脚本
    - ../../C2G/evidence/submission_script.patch  统一 diff（可复核）
"""
from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# 基线有两份：_upstream/ 里的原始克隆（不进交付包，见 .gitignore）和
# evidence/upstream/ 里的包内副本（v4 起随包交付，让证据链自包含）。
# MIT 许可允许再分发（见 evidence/upstream/LICENSE_upstream.txt）。
PACKED_BASE = os.path.join(ROOT, "evidence", "upstream", "root_train_gpt.py")
UPSTREAM_BASE = os.path.join(ROOT, "_upstream", "parameter-golf-main", "train_gpt.py")
OUT = os.path.join(ROOT, "卢怡然_C2G_train_gpt.py")
PATCH = os.path.join(ROOT, "evidence", "submission_script.patch")
# 补丁头里的路径必须是**包内**路径，否则评审拿到包也无法 apply（v4 洞 2）。
PATCH_FROM = "evidence/upstream/root_train_gpt.py"
PATCH_TO = "卢怡然_C2G_train_gpt.py"

# ---------------------------------------------------------------- 插入 1：超参数
HP_ANCHOR = '    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))\n'
HP_ADD = HP_ANCHOR + '''
    # ---- 卢怡然 C2G：评测步长（eval stride）的运行时选择 -------------------
    # 朴素评测把验证集切成互不重叠的长度 = train_seq_len 的窗口，窗口内每个位置都计分，
    # 于是位置 0 是"零上下文"预测。滑窗评测（ stride < seq_len）让每个 token 至少带
    # window - stride 个上下文再被计分 —— 机制来自
    #   openai/parameter-golf  records/track_10min_16mb/2026-03-19_SlidingWindowEval（Matthew Li）
    # 我改的是：**stride 不再写死成 64**。64 要付 seq_len/64 = 16 倍评测算力，
    # 而 rho(p)（第 p 个位置的期望损失）对上下文的边际收益强递减，
    # 大部分收益在远短于此的上下文处就已经拿到（实测见 pg_lab/analysis/）。
    # 因此这里按"评测算力预算"现场标定出一个最省的 stride。
    eval_stride_mode = os.environ.get("EVAL_STRIDE_MODE", "knee")   # naive | fixed | knee
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))            # mode=fixed 时使用
    eval_cost_budget = float(os.environ.get("EVAL_COST_BUDGET", 4.0))   # 愿意付的前向 token 倍数上限
    knee_gain_frac = float(os.environ.get("KNEE_GAIN_FRAC", 0.95))      # 要求拿到最优收益的这一比例
    knee_candidates = os.environ.get("KNEE_CANDIDATES", "")             # 为空则自动枚举；非正数/超窗被过滤，全非法回退自动枚举（v4）
    eval_calib_tokens = int(os.environ.get("EVAL_CALIB_TOKENS", 1_048_576))
    eval_calib_batch_seqs = int(os.environ.get("EVAL_CALIB_BATCH_SEQS", 256))
    eval_calib_stride = int(os.environ.get("EVAL_CALIB_STRIDE", 64))    # rho 的测量步长（需整除候选 stride）
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 1024))
'''

# ---------------------------------------------------------------- 插入 2：GPT.forward_logits
FL_ANCHOR = "    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:\n"
FL_ADD = '''    def forward_logits(self, input_ids: Tensor) -> Tensor:
        # 与 forward 等价，但返回逐位置 logits 而非均值损失 —— 滑窗评测需要逐位置 NLL。
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # vocab 大小取自词嵌入行数（tied/untied 均恒等于 vocab_size）；
        # v3 修复：原写 self.vocab_size，但 GPT 未存该属性，真实实例上一调用即 AttributeError（dry-run 发现）。
        return logits.view(*input_ids.shape, self.tok_emb.weight.shape[0])

''' + FL_ANCHOR

# ---------------------------------------------------------------- 插入 3：核心评测函数
CORE_ANCHOR = """# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------"""
CORE_ADD = '''# -----------------------------
# 卢怡然 C2G：eval stride 的运行时拐点标定
# -----------------------------
# 方法论一句话：先在一段校准 token 上测出「逐位置期望损失曲线」 rho(p)，
# 再在给定的评测算力预算内，选出**最省**且仍能拿到 knee_gain_frac 比例最优收益的 stride。
#
# 为什么敢用训练分布的 token 做标定：
#   stride 的选择只依赖 rho 的**形状**（随上下文下降的相对速度），不依赖它的**水平**。
#   形状从 train 迁移到 val 远比水平稳健。这一点在本地实验里被直接检验过：
#   用 train / val 两份数据分别标定，选出的 stride 一致，算出的 val BPB 也一致
#   （见 pg_lab/analysis 与 evidence/）。因此默认不用验证集调这个参数，规避
#   CHALLENGE.md "不要用验证集调超参" 的陷阱。


def measure_position_profile(
    base_model: nn.Module,
    calib_tokens: Tensor,
    window: int,
    stride_min: int,
    batch_seqs: int,
    device: torch.device,
) -> Tensor:
    """rho[p] = 窗口第 p 个位置的期望 NLL（nats）。用 overlapping 窗口降低方差。"""
    n = calib_tokens.numel()
    if n < window + 1:
        raise ValueError(f"校准片段太短: {n} < window+1")
    starts = list(range(0, n - window, stride_min))
    if not starts:
        starts = [0]
    prof = torch.zeros(window, dtype=torch.float64, device=device)
    cnt = torch.zeros((), dtype=torch.float64, device=device)

    prev = base_model.training
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(starts), batch_seqs):
            ws = starts[bi: bi + batch_seqs]
            x = torch.stack([calib_tokens[s: s + window] for s in ws]).to(device=device, dtype=torch.int64)
            y = torch.stack([calib_tokens[s + 1: s + window + 1] for s in ws]).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none"
            ).reshape(len(ws), window)
            prof += nll.to(torch.float64).sum(dim=0)
            cnt += nll.shape[0]

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(prof, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
    base_model.train(prev)
    return prof / cnt.clamp(min=1)


def stride_candidates(window: int, stride_min: int) -> list[int]:
    """候选 stride：window 的约数里 >= stride_min 且 <= window 的那些，降序（由贵到便宜）。"""
    cands = [s for s in range(stride_min, window + 1) if window % s == 0]
    cands = sorted(set(cands), reverse=True)
    return cands if cands else [window]


def _parse_knee_candidates(raw: str, window: int, stride_min: int) -> list[int]:
    """解析 KNEE_CANDIDATES（逗号分隔）为合法候选列表。

    规则（v4 边界修复）：
      - 非整数字面量 -> int() 直接抛 ValueError（参数写错应当 fail fast，不静默吞掉）；
      - 非正数（0/负数）与超过 window 的值一律**过滤**——0 会在 cost=window/s 处除零，
        负数会产生负代价这种无意义行；
      - 过滤后为空（全部非法）-> **回退自动枚举**，绝不允许空候选往下传：
        旧版这里会把空列表交给 pick_stride_by_knee，max() 直接 ValueError 崩溃。
    """
    req: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        s = int(part)
        if 1 <= s <= window:
            req.add(s)
    if not req:
        return stride_candidates(window, stride_min)
    return sorted(req, reverse=True)


def pick_stride_by_knee(
    prof: Tensor, window: int, cost_budget: float, gain_frac: float, candidates: list[int]
) -> tuple[int, dict]:
    """在算力预算内选最省（stride 最大）且收益 >= gain_frac * 最优收益的 stride。

    rho 已知时协议的稳态 BPB 可解析算出（无需真的跑一遍）：
        BPB_steady(S) = mean(rho[window-S : window]) / ln2
    代价倍数 C(S) = window / S。由此遍历候选即可。
    预算语义（v4 澄清）：预算是**软上限**——评测必须完成，因此连 1x（朴素）都
    付不起的预算不会让程序崩溃，也不会反向突破去选更贵的候选，而是退化为朴素
    评测（1x）并在 info["budget_ok"]=False 中显式暴露，由调用方打 WARN 日志。
    """
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
    budget_ok = True
    if eligible:
        chosen = max(eligible, key=lambda r: r["stride"])   # 最省
    elif affordable:
        chosen = max(affordable, key=lambda r: r["gain"])   # 预算内拿最多
    else:
        # v4 修复：旧版这里返回 rows[-1]（stride 最小 = **最贵**的候选，如 16x），
        # 而注释写着"退化为朴素"——行为与注释相反，且正好突破预算这个硬约束。
        # 现在真正退化为朴素评测（stride=window，cost=1x），并显式暴露预算被忽略。
        budget_ok = False
        chosen = {"stride": window, "cost": 1.0, "gain": 0.0, "bpb": naive}
    return chosen["stride"], {"chosen": chosen, "naive_bpb": naive,
                             "best_gain": best_gain, "all": rows, "budget_ok": budget_ok}


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    """滑窗评测。机制与 Matthew Li 的 record 实现一致（窗口 0 全计分，其余计分尾部 stride 个）。

    改了什么：
      - stride 由调用方给定（来自拐点标定），而不是硬编码
      - batch_seqs 可按显存调节
      - **真实累计 forward token 数**并在结束时打一行 sliding_eval_cost 日志（v4）：
        amortized_cost = forward_tokens / scored_tokens，其中 1x = 官方朴素评测
        （朴素评测下每个验证 token 恰好前向一次）——EVAL_COST_BUDGET 的口径就是它。
        旧版 docstring 声称有这个审计、实际累计的是计分 token —— 言行一致才算数。
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    forward_tokens = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)
            forward_tokens += float(bsz * seq_len)   # 审计：forward 的真实 token 数（含 padding）

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            # Progress (rank 0 only)
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct = done / len(my_windows) * 100
                running_bpb = 0.0
                if token_count.item() > 0:
                    rl = (loss_sum / token_count).item()
                    running_bpb = rl / math.log(2.0) * (token_count.item() / byte_count.item())
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows "
                      f"running_bpb={running_bpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(forward_tokens, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    if rank == 0:
        fwd = forward_tokens.item()
        scored = token_count.item()
        print(f"sliding_eval_cost forward_tokens={fwd:.0f} scored_tokens={scored:.0f} "
              f"amortized_cost={fwd / max(scored, 1.0):.3f}x   # 1x = 官方朴素评测", flush=True)
    base_model.train()          # 与官方 baseline 逐字一致（无条件切回 train）
    return val_loss, bits_per_token * tokens_per_byte


def resolve_eval_stride(args: Hyperparameters, base_model: nn.Module, calib_tokens: Tensor,
                        device: torch.device, log0) -> tuple[int, dict]:
    """按 EVAL_STRIDE_MODE 决定最终 stride。返回 (stride, info)。"""
    window = args.train_seq_len
    if args.eval_stride_mode == "naive":
        return window, {"mode": "naive"}
    if args.eval_stride_mode == "fixed":
        return min(max(args.eval_stride, 1), window), {"mode": "fixed"}

    assert args.eval_stride_mode == "knee", args.eval_stride_mode
    calib = calib_tokens[: args.eval_calib_tokens]
    prof = measure_position_profile(
        base_model, calib, window, args.eval_calib_stride,
        args.eval_calib_batch_seqs, device,
    )
    # v4 边界修复：走统一的解析函数（非整数 fail fast / 非正数与超窗过滤 / 全非法回退自动枚举）。
    cands = _parse_knee_candidates(args.knee_candidates, window, args.eval_calib_stride)
    stride, info = pick_stride_by_knee(prof, window, args.eval_cost_budget,
                                       args.knee_gain_frac, cands)
    info["mode"] = "knee"
    info["candidates"] = cands
    if not info.get("budget_ok", True):
        log0(f"WARN knee_calib budget={args.eval_cost_budget}x 连 1x（朴素评测）都付不起，"
             f"已退化为朴素评测 stride={stride}；评测必须完成，预算在此被视为软上限。"
             f"请调大 EVAL_COST_BUDGET 或调小 KNEE_CANDIDATES 的代价倍数。")
    # rho 的形状诊断：每 1/8 窗口位置的平均 NLL，方便人工肉眼检查收益递减
    step = max(1, window // 8)
    diag = []
    for i in range(0, window, step):
        diag.append((i, round(float(prof[i:i + step].mean()), 5)))
    info["rho_shape"] = diag
    log0(f"knee_calib mode=knee budget={args.eval_cost_budget} gain_frac={args.knee_gain_frac} "
         f"chosen_stride={stride} cost={window / stride:.2f}x "
         f"predicted_gain_bpb={info['chosen']['gain']:.6f} "
         f"max_gain_bpb={info['best_gain']:.6f}")
    log0(f"knee_calib rho_shape={diag}")
    return stride, info


''' + CORE_ANCHOR

# ---------------------------------------------------------------- 替换 1：最终 eval 调用
FINAL_OLD = """    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()"""
FINAL_NEW = """    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    # ---- 卢怡然 C2G：用标定出的 stride 做最终评测（而非朴素的互不重叠窗口）----
    _calib = _load_calib_tokens(args)
    _stride, _info = resolve_eval_stride(args, model, _calib, device, log0)
    if _stride >= args.train_seq_len:
        # 退化路径：与官方 baseline 完全一致
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
    else:
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            _stride, args.eval_batch_seqs,
        )
    torch.cuda.synchronize()"""

# ---------------------------------------------------------------- 插入 4：校准数据读取
LOAD_ANCHOR = """# -----------------------------
# TRAINING
# -----------------------------"""
LOAD_ADD = '''def _load_calib_tokens(args: Hyperparameters) -> Tensor:
    """用于标定 stride 的 token。

    刻意取自**训练分布**而非验证集：stride 的选择只依赖 rho 的形状而不依赖水平，
    且可以避免用 Leaderboard 的验证集调超参（CHALLENGE.md 明确禁止）。
    """
    files = [Path(p) for p in sorted(glob.glob(args.train_files))]
    if not files:
        raise FileNotFoundError(f"No train files found for pattern: {args.train_files}")
    need = int(args.eval_calib_tokens) + args.train_seq_len + 2
    acc: list[Tensor] = []
    got = 0
    for f in files:
        if got >= need:
            break
        t = load_data_shard(f)
        acc.append(t[: min(t.numel(), need - got)])
        got += min(t.numel(), need - got)
    tok = torch.cat(acc).contiguous().to(torch.int64)
    return tok


''' + LOAD_ANCHOR


def apply_once(src: str, anchor: str, add: str, name: str) -> str:
    if src.count(anchor) != 1:
        raise SystemExit(f"锚点 {name} 出现 {src.count(anchor)} 次（应为 1）")
    return src.replace(anchor, add, 1)


def _resolve_base() -> str:
    """决定用哪份基线：两者都在时必须逐字节一致（防包内副本静默漂移）。"""
    have_packed = os.path.exists(PACKED_BASE)
    have_up = os.path.exists(UPSTREAM_BASE)
    if not have_packed and not have_up:
        raise SystemExit("找不到基线：既无 evidence/upstream/root_train_gpt.py 也无 _upstream/ 克隆")
    if have_packed and have_up:
        a = open(PACKED_BASE, "rb").read()
        b = open(UPSTREAM_BASE, "rb").read()
        if a != b:
            raise SystemExit(
                "包内基线与 _upstream 克隆不一致："
                f"{PACKED_BASE} sha256={hashlib.sha256(a).hexdigest()[:16]} vs "
                f"{UPSTREAM_BASE} sha256={hashlib.sha256(b).hexdigest()[:16]}。"
                "请重新 copy 后再生成，避免两份基线静默漂移。"
            )
    return UPSTREAM_BASE if have_up else PACKED_BASE


def build() -> None:
    BASE = _resolve_base()
    src = open(BASE, encoding="utf-8").read()
    out = src
    out = apply_once(out, HP_ANCHOR, HP_ADD, "hyperparameters")
    out = apply_once(out, FL_ANCHOR, FL_ADD, "forward_logits")
    out = apply_once(out, CORE_ANCHOR, CORE_ADD, "core eval block")
    out = apply_once(out, LOAD_ANCHOR, LOAD_ADD, "calib loader")
    if out.count(FINAL_OLD) != 1:
        raise SystemExit(f"最终 eval 代码块出现 {out.count(FINAL_OLD)} 次（应为 1）")
    out = out.replace(FINAL_OLD, FINAL_NEW, 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)
    os.makedirs(os.path.dirname(PATCH), exist_ok=True)
    # git 风格头：git apply 只有在 old/new 名字不同且带 `diff --git` 时才肯创建新文件。
    # 行尾提醒：必须 -c core.autocrlf=false，否则 Windows 默认配置会把产出转成 CRLF，
    # 字节数与 sha256 都对不上（C4A 里踩过的同一个坑，这里一并防住；包内还有 .gitattributes）。
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="a/" + PATCH_FROM, tofile="b/" + PATCH_TO, n=3,
    )
    with open(PATCH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# 本 diff 的两条路径都在交付包内，可直接 apply：\n")
        fh.write(f"#   git -c core.autocrlf=false apply -p1 evidence/submission_script.patch\n")
        fh.write(f"# 说明：必须关掉 autocrlf，否则 Windows 默认配置会把产出转成 CRLF，")
        fh.write(f"字节数与 sha256 都对不上。\n")
        fh.write(f"diff --git a/{PATCH_FROM} b/{PATCH_TO}\n")
        fh.writelines(diff)

    def sha(p: str) -> str:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    def sha16(p: str) -> str:
        return sha(p)[:16]

    print("基准（上游官方 root baseline）:", BASE)
    print("  sha256     =", sha(BASE))
    print("  sha256[:16]=", sha16(BASE), "行数 =", len(src.splitlines()))
    print("包内基线副本:", PACKED_BASE)
    if os.path.exists(PACKED_BASE):
        print("  sha256     =", sha(PACKED_BASE))
    print("产出:", OUT)
    print("  sha256     =", sha(OUT))
    print("  sha256[:16]=", sha16(OUT), "行数 =", len(out.splitlines()))
    print("diff:", PATCH)
    print("  sha256     =", sha(PATCH))
    print("  sha256[:16]=", sha16(PATCH), "行数 =", len(open(PATCH, encoding="utf-8").read().splitlines()))
    # 滑窗机制出处（1366 行记录版）也一并报哈希，供 base_sha256.txt 留档
    sw = os.path.join(ROOT, "evidence", "upstream", "record_2026-03-19_SlidingWindowEval_train_gpt.py")
    if os.path.exists(sw):
        print("滑窗机制出处:", sw)
        print("  sha256     =", sha(sw))
        print("  行数 =", len(open(sw, encoding="utf-8").read().splitlines()))
    nb = os.path.join(ROOT, "evidence", "upstream", "record_2026-03-17_NaiveBaseline_train_gpt.py")
    if os.path.exists(nb):
        print("朴素基线记录:", nb)
        print("  sha256     =", sha(nb))
        print("  行数 =", len(open(nb, encoding="utf-8").read().splitlines()))


if __name__ == "__main__":
    build()
