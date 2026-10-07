# 卢怡然 C2G —— 参数高尔夫（Parameter Golf）

> 赛道：OpenAI Parameter Golf `track_10min_16mb`
> 交付日期：2026-10-04 ｜ 最近加固：2026-10-07（v4）｜ 截止：2026-12-31

---

## ⚠️ 先看这一节：本包**没有**官方尺度的成绩

| 项 | 状态 |
|---|---|
| 8×H100 上的 BPB | **无。从未跑过。** |
| 官方榜单排名 | **未上榜** |
| 提交到官方 repo | 未做 |

**原因**：交付者本机**无 GPU**（`nvidia-smi` 不存在）、**无外网**（FineWeb / Gutenberg 均不可达）。
这不是托词，是实测结果。所有 BPB 相关的 `submission.json` 字段一律填 `null` 并标 `status: not_run`。

### `status: not_run` 与 `bpb: null` 的确切语义（v4 补，回应评审洞 1）

评审指出「`submission.json` 中 `bpb` 字段全为 `null`，但未明确说明 `status: not_run` 语义」。
这里把话说死：

**`status` 是三态，不是一个模糊标签：**

| 取值 | 含义 | 此时成绩字段应该 |
|---|---|---|
| `not_run` | **从未在该硬件上执行过本脚本**（本次交付即此状态） | 全部 `null` |
| `submitted` | 跑过了，有真实日志，但尚未被第三方复核 | 填真实数字 + 日志路径 |
| `verified` | 跑过且被第三方复核通过 | 填真实数字 + 复核证据 |

**`not_run` ≠ "省略" ≠ "跑过但成绩不好"。** 后两种都必须填 `submitted` 并给出真实数字——
把难看的真数字写成 `null`，和把假数字写成真数字，是同一种欺骗的两个方向。

**每一个 `null` 都必须是"尚未观测"，不能是"漏填"。** 因此 `submission.json` 里新增了
`null_field_semantics`（逐字段交代它会在哪一步、由哪条日志产生）和
`how_to_fill_after_getting_gpu`（拿到算力后的 6 步执行顺序）。**没有任何一个 `null`
允许用估算值、引用值或他人成绩回填。**

**本包提供的是**：一个**真跑通过**的提交脚本（单元测试 12/12 + CPU 端到端 dry-run，
naive / fixed / knee / knee_tight 四条路径完整跑通训练→验证→int8 打包→往返校验）
+ 一套在 CPU 上真跑过的、n=3+3 seed 的对照实验（含官方真实配置 W=1024）
+ 完整的方法学与诚实声明。

---

## 一、方案一句话

**不改训练、不改架构、不改量化。只改评测：把 `eval stride` 从写死的常数 64，
改成训练结束后用一小段校准 token 现场标定的量。**

```
官方 baseline  = 窗口 W = 步长 S = train_seq_len   →  位置 0 的 token 是零上下文预测
滑窗评测      = W = train_seq_len，S < W           →  每个 token 至少带 W−S 个上下文
代价倍数      = W / S                              →  官方 stride=64 / W=1024 即 16×
```

朴素评测是 `S = W` 的一个特例，所以整个协议族只有两个自由度，可以画出一根「代价—收益」曲线。

**要解决的问题**：历史方案付了 16× 的评测算力；如果 ρ(p)（逐位置期望损失）对上下文的
边际收益是强递减的，那么 16× 里有一大部分可能买的是一条几乎平坦的尾巴。

---

## 二、目录

### 交付文档（CHALLENGE.md 必交项）

| 文件 | 说明 |
|---|---|
| **`卢怡然_C2G_方案草案.md`** | Level 1 算力申请用，回答 4 个门槛问题（1979 字） |
| `卢怡然_C2G_方案设计.md` | 目标 Level、技术选型、实验矩阵、风险判据 |
| **`卢怡然_C2G_train_gpt.py`** | 最终提交脚本（**1463 行 / 65,141 字节**，由官方 baseline 程序化生成；v4 新增边界修复与代价审计后 +46 行，sha256 `19f493edbac875ac…`） |
| `卢怡然_C2G_submission.json` | 提交元数据（8×H100 字段为 null，标注 `not_run`；**v4 新增 `status_note` / `null_field_semantics` / `how_to_fill_after_getting_gpu`**） |
| `卢怡然_C2G_logs/` | 3 seed 训练日志（`trainlog_seed{1337,42,2026}.txt/.json` + 语料 manifest） |
| `卢怡然_C2G_ablation.md` | 消融实验报告（协议族 + 拐点 + 负结果 + 交叉验证） |
| `卢怡然_C2G_leaderboard.md` | 与 baseline / SOTA 的对比（含官方 30 条真实记录） |
| **`卢怡然_C2G_AI日志.md`** | AI 使用全过程 + 反向举证 |
| `卢怡然_C2G_拿来说明.md` | 拿了什么 / 改了什么 / 没拿什么 |
| **`卢怡然_C2G_AAR.md`** | 复盘：3 个失败 + 1 个明确标注的未做实验 |

### 代码与证据

| 路径 | 说明 |
|---|---|
| `pg_lab/` | **CPU 可跑的实验台**（详细见 `pg_lab/README.md`） |
| `pg_lab/analyze_w1024.py` | **W=1024 真实配置对照**（训练窗口=评测窗口=1024，3 seed）v3 新增 |
| `pg_lab/dryrun/` | **提交脚本 CPU 端到端 dry-run**（tiny spm+数据构建 + 11 条公开 CPU 适配补丁），v3 新增；**v4 扩到 4 个模式**（naive / fixed / knee / knee_tight） |
| `pg_lab/selftest.py` | 对提交脚本决策函数的单元测试，**12/12 通过**（v3 为 7/7，v4 补 5 条边界回归） |
| `pg_lab/build_submission.py` | 由官方 baseline 程序化生成提交脚本 + 统一 diff（v4 起基线优先取包内副本，并断言两份一致） |
| `pg_lab/hash_registry.py` | **v4 新增**：一次算全 `evidence/base_sha256.txt`，含「补丁能否在包内 apply」的真跑自检 |
| `evidence/upstream/` | **v4 新增**：官方 baseline 与滑窗机制出处的**包内副本**（MIT 许可，合法入库）。评审拿到包即可 apply 补丁、核对哈希，无需外网 |
| `evidence/` | 原始输出留档：分析输出、交叉验证、官方榜单表、哈希、diff。其中 **`evidence/审计说明_空文件.md`** 专门解释自检器报告的 2 个「空文件」（均为上游自带或写入时序问题，非交付物缺失），评审如见空文件告警请先读它 |
| `_upstream/` | 官方 repo 解压副本（**不进自己的 GitHub 仓库**，已在 `.gitignore`；包内副本在 `evidence/upstream/`） |
| `.gitattributes` | **v4 新增**：`* -text` 关闭 Git 行尾自动转换。不关的话 Windows 默认 `autocrlf=true` 会让 apply 产物变成 CRLF、sha256 对不上（实测 66604 vs 65141 字节） |

---

## 三、核心实测结果（CPU 小尺度，n=3 seed）

> ⚠️ **坐标系不同**：494K 参数、字节级 vocab=256、4MB 本地语料。
> **绝对 BPB 与官方不可比**，只有 Δ 有参考价值。

### ρ(p) 的形状 —— 本方案的立论

| 位置区间 | 0–7 | 8–31 | 32–63 | 64–127 | 128–255 | 256–383 | 384–495 | 496–511 |
|---|---|---|---|---|---|---|---|---|
| BPB | 2.8512 | 2.1302 | 2.0295 | 1.9899 | 1.9794 | 1.9773 | **1.9770** | 1.9799 |

**前 8 个位置就掉掉 82% 的可降幅度，到 64–127 已拿下 98.5%。**
最末尾（496–511）反而回升 0.0029 —— 这解释了为什么极小 stride 会变差。

### 协议族消融

| 代价倍数 | 1.00× | 1.14× | 1.33× | 1.60× | **2.00×** | 2.67× | 4.00× | 8.00× | **16.00×** |
|---|---|---|---|---|---|---|---|---|---|
| ΔBPB | 0 | −0.02375 | −0.02544 | −0.02596 | **−0.02614** | −0.02618 | −0.02610 | −0.02545 | **−0.02460** |

**2× 代价拿到 99.8% 的最大收益。16×（官方 stride=64 的等价代价）比 2× 差 0.00154 BPB**
（配对标准差 0.00050，3 个 seed 符号一致）。

### 拐点

| 想要 50% 收益 | 想要 90% | 想要 95% | 想要 99% | 想要 100% |
|---|---|---|---|---|
| 1.14× | 1.14× | 1.33× | 1.60× | 2.67× |

### 负结果：窗口拉到训练长度之外（W=1024，训练窗口 512）

整条曲线都是灾难，ΔBPB 从 **+0.166 恶化到 +0.467**，越努力越糟。
**这直接否掉了"省下的算力拿去买更长评测窗口"的想法**（除非同时改训练或加 RoPE 缩放）。

### 真实配置对照：训练窗口 1024 + 评测窗口 1024（v3 新增，已补上旧缺口）

| 代价倍数 | 1.00× | 1.33× | **2.00×** | 2.67× | 4.00× | 8.00× | **16.00×** |
|---|---|---|---|---|---|---|---|
| ΔBPB | 0 | −0.01774 | **−0.01855** | −0.01831 | −0.01761 | −0.01602 | **−0.01365** |

- **最优档就是 2×**（stride=512），朴素基线 BPB 均值 2.07601
- **2× 比 16× 好 0.004894 BPB**（配对 std 0.000674，3/3 seed 符号一致）
- 1.33× 已拿到 95.6% 的最大收益
- 与 W=512 那组比：**结论不变，效应更强**（W=512 时 2× vs 16× 只差 0.00154）

**这组实验的意义**：官方真实配置下，"16× 买贵了"不是小窗口的假象。
token 预算与 W=512 组完全相同（1024×8 = 512×16），唯一变量是训练窗口。
详见 `pg_lab/analysis_w1024/W1024真实配置对照.md`、原始输出 `evidence/w1024_config_check.txt`。

### 交叉验证：用训练分布标定、在验证分布生效？

| 检验 | 结果 |
|---|---|
| ρ 的**水平**差（val − train） | **+0.884 nats/byte**（巨大） |
| 五档预算下两种标定选出的 stride | **全部一致**（256） |
| 两种标定算出的验证集 BPB 差 | **0.00e+00** |

水平差了 0.88 nats，决策却完全一致 ——
**"stride 只依赖形状不依赖水平"这个论证成立**，所以敢用训练分布标定（规避验证集调参）。

### 方法学自检

ρ 重建 vs 逐字节金标准直接评测：**最大误差 2.09×10⁻³ BPB**（约为效应量的 8%）。

---

## 四、复现

```bash
# 环境：Python 3.13 + torch(cpu) + numpy，离线可跑
cd pg_lab

python build_corpus.py --out data                                    # ~10s，构建确定性语料
python train.py --seed 1337 --steps 1200 --lr 5e-3 --threads 14 --out runs
python train.py --seed 42    --steps 1200 --lr 5e-3 --threads 14 --out runs
python train.py --seed 2026  --steps 1200 --lr 5e-3 --threads 14 --out runs

python analyze.py       --ckpt runs/ckpt_seed1337.pt --windows 512 1024
python crossvalidate.py --ckpt runs/ckpt_seed1337.pt --window 512
python report.py        --dir analysis
python selftest.py                                                   # 12/12

# 边界回归用例的反向验证：证明新增的 4 条用例真的能抓到旧 bug（不是恒真空转）
python ../evidence/回归测试_反向验证.py                                # 4/4 能区分新旧实现

# forward_logits 的逐位置 / 因果 / 与官方 forward 数值等价
python ../evidence/forward_logits_等价性验证.py                        # PASS

# W=1024 真实配置对照（v3 补的那一组，token 预算与上面完全一致）
python train.py --seed 1337 --steps 1200 --seq-len 1024 --batch-seqs 8 --lr 5e-3 --threads 5 --out runs_w1024
python train.py --seed 42    --steps 1200 --seq-len 1024 --batch-seqs 8 --lr 5e-3 --threads 5 --out runs_w1024
python train.py --seed 2026  --steps 1200 --seq-len 1024 --batch-seqs 8 --lr 5e-3 --threads 5 --out runs_w1024
python analyze_w1024.py                                              # 3 seed 并行，约 7 分钟

# 提交脚本的 CPU 端到端 dry-run（v4 起 4 个模式：naive / fixed / knee / knee_tight）
cd dryrun && python build_dryrun_data.py && python cpu_shim_runner.py
# 依赖：pip install torch numpy sentencepiece

# 重新生成提交脚本与 diff
python build_submission.py

# 重新登记所有哈希（含「补丁能否在包内 apply」的真跑自检）
python hash_registry.py
```

单个 seed 训练约 8 分钟（16 线程 CPU）；W=1024 组因 batch 减半保持 token 预算相同，耗时相当。
dry-run 约 4 分钟/模式（2 层/64 维，30 步，CPU），**4 个模式约 10 分钟**。

### 拿到包以后怎么核对证据链（v4 新写，3 条命令）

```bash
# 1. 基线 + 补丁 -> 提交脚本，应与包内文件字节一致
git -c core.autocrlf=false apply -p1 evidence/submission_script.patch
sha256sum 卢怡然_C2G_train_gpt.py       # 应 = base_sha256.txt [2] 里的 19f493edbac875ac…

# 2. 重算所有哈希（含上面这条 apply 的自动自检）
python pg_lab/hash_registry.py

# 3. 单测 + 反向验证
python pg_lab/selftest.py               # 12/12
python evidence/回归测试_反向验证.py     # 4/4 能区分新旧实现
```

第 1 条**必须**带 `-c core.autocrlf=false`：Windows 版 Git 出厂默认 `autocrlf=true`，
不关的话 apply 产物会被转成 CRLF（实测 66,604 字节 vs 正确的 65,141 字节），
sha256 对不上——看起来就像证据造假。包内 `.gitattributes` 已经用 `* -text` 从仓库层面
挡住了这个坑，但手工 apply 时仍建议显式关掉。

---

## 五、拿到 H100 后怎么跑

```bash
# H1 复现 baseline（必须通过才继续）
EVAL_STRIDE_MODE=naive torchrun --standalone --nproc_per_node=8 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
    MAX_WALLCLOCK_SECONDS=600 卢怡然_C2G_train_gpt.py

# H2 同一权重扫协议族（不重训，最便宜的一步）
EVAL_STRIDE_MODE=fixed EVAL_STRIDE=256 ...

# H3 拐点标定
EVAL_STRIDE_MODE=knee EVAL_COST_BUDGET=2.0 KNEE_GAIN_FRAC=0.95 ...
```

判据见 `卢怡然_C2G_方案草案.md` §问题 4。**baseline 复现失败就停，不往下走。**

---

## 六、诚实清单（未做的事）

- [ ] **未在 8×H100 上跑过任何一次**（无 GPU）
- [ ] **未复现官方 baseline BPB**（无算力）
- [ ] **未提交到官方榜单**
- [x] **「训练窗口 1024 + 评测窗口 1024」的真实配置（v3 已补）** —— 3 seed 真跑，
      拐点落在 **2×**，2× 比 16× 好 **0.004894** BPB（配对 std 0.000674，3/3 同号）。
      见 `pg_lab/analysis_w1024/W1024真实配置对照.md`
- [x] **16MB artifact 打包验证（v3 已实测，v4 四模式复测）** —— 真实跑通 int8+zlib 打包并往返校验：
      缩小模型 artifact = **212,061 字节 = 0.202 MiB**（上限 16,777,216），4 个模式数值一致。
      ⚠️ 这是 2 层/64 维缩小模型、CPU、30 步的实测，证明的是**打包代码路径真的跑通且远低于上限**，
      不是官方尺度模型的最终体积。见 `evidence/dryrun_{naive,fixed,knee,knee_tight}.txt`
- [ ] 未做 Level 2 以上的任何组合优化
- [ ] **未在真机上核对 `amortized_cost` 的预测值与实测值** —— CPU dry-run 上
      预测 4.0x / 实测 3.992x（fixed 模式），但真机的 padding 与尾窗结构不同，偏差量级未知
- [ ] **本轮未重跑提交前体检** —— `evidence/submit_guard_audit_v3.txt` 是之前由 C2G 专用
      体检工具产出的。v4 试过用 C4A 那套 `c4_evaluator.py` 扫一遍，但它用的是 C4 的技能
      提交 rubric，把 `_upstream/` 里的 33 份 record 识别成"50 位作者"，结论无意义，已弃用

**这份清单里没有任何一项是靠文档篇幅掩盖过去的。**

---

## 七、评审反馈响应（2026-10-06，v2）

平台评分 **82/100** 及修改建议收到后，本轮处理如下：

| 评审项 | 处理 | 位置 |
|---|---|---|
| P1：AAR「成绩占 35 分」与 rubric 不一致 | 已改为 **25 分**并标注出处（`rubric.json` → `scoreAchievement.maxPoints=25`） | `卢怡然_C2G_AAR.md` §〇 |
| P1：`github_id`（luyiran）与仓库所有者不统一 | 已改为 **shuibubao--zz**（本机 git 全局配置的账号名；各挑战仓库建立时所有者将与此一致） | `卢怡然_C2G_submission.json` |
| P1：README 未交叉引用空文件说明 | 已在目录表 `evidence/` 行加入显式指引 | 本文件 §二 |
| P1：AI 日志补强 | 已补 v2 轮（评审响应轮）的触发、决策与验证记录 | `卢怡然_C2G_AI日志.md` §迭代记录 |
| **P0：真机复现 baseline 1.2244 ± 0.005** | **未做，也无法在本机做**——无 GPU（`nvidia-smi` 不存在）。判据与执行步骤已在 §五写死，拿到算力即可执行 | 本文件 §五、§六 |

**诚实预期**：P0（成绩达成维度，25 分）不做，总分上限受它硬性约束；本轮修的是其余四个维度里所有可修复的一致性问题。

---

## 八、评审反馈响应（2026-10-06，v3｜第三方独立评审 cogseed）

评审给出 P0/P1/P2 共 6 条。逐条核实后处理如下（**先核实再动手**，因为评审自己的判断也可能有误）：

| 评审项 | 我的核实与处理 | 位置 |
|---|---|---|
| **P1-1**：`scoreAchievement.maxPoints=25` 引用无法在提交包内验证 | **评审观察正确（文件确实不在包内），但引用本身成立** —— C2G 挑战资料包内有专属 `rubric.json`，`maxPoints` 就是 25，实测 SHA-256 `0ffaedb175d51751…` 与资料包 README 自带清单逐字一致。已把原件复制进包 | `evidence/官方评分标准_rubric_C2G.json`、`evidence/评分标准出处核验.md` |
| **P1-2**：AAR §〇「成绩占 25 分」与 §六「成绩达成（25）」措辞不统一 | **正确**，已统一为「成绩达成（25 分）」，与 rubric `label`+`maxPoints` 对齐 | `卢怡然_C2G_AAR.md` §〇、§六 |
| **P1-3**：`github_id` 全链一致性 | 已核查：唯一带该字段的提交卡就是 C2G 这份，值 `shuibubao--zz` = git 全局 `user.name`；C1/C2/C2A/C4 交付形态不含提交卡，无需回填 | `evidence/github_id一致性核验.md` |
| **P2-4**：补「训练窗口 1024 + 评测窗口 1024」真实配置 | **已补**，3 seed 真跑（token 预算与 W=512 组完全相同）。拐点 **2×**，2× 比 16× 好 **0.004894** BPB | `pg_lab/analysis_w1024/W1024真实配置对照.md`、`evidence/w1024_config_check.txt` |
| **P2-5**：16MB artifact 打包实测 | **已实测**（不是理论估算）：int8+zlib 真实打包 + 往返校验，212,061 字节 = 0.202 MiB | `evidence/dryrun_{naive,knee}.txt` |
| **P2-6**：提交脚本小规模端到端 dry-run | **已做**，且**真抓出一个 bug**：`forward_logits` 引用了不存在的 `self.vocab_size`，单测覆盖不到（只测决策函数），真实实例一调用即 `AttributeError`。已修为 `self.tok_emb.weight.shape[0]`，重新生成脚本后 naive + knee 两条路径全流程跑通 | `pg_lab/dryrun/`、`卢怡然_C2G_train_gpt.py`（1417 行） |
| **P0**：真机复现 baseline 1.2244 ± 0.005 | **仍未做，本机无法做**：无 GPU（`nvidia-smi` 不存在）、无 torch-cuda。判据与步骤写死在 §五，拿到算力即执行 | §五、§六 |

**本轮最值钱的一条**：不是补了什么文档，而是 dry-run 抓出的那个 bug——
7 项单元测试全部通过、语法检查通过，脚本却**根本跑不起来**。
"单测通过"从来不等于"能跑"；这次是第一次真把它跑起来，代价是发现了自己的错。

---

## 九、评审反馈响应（2026-10-07，v4｜平台 82/100 + 独立复核 ~85/100）

评审给了三个洞 + 五个必须回答的问题。**先逐条核实，再动手**——评审的观察对不对，
要自己跑一遍才知道。

### 三个洞

| 洞 | 评审说的 | 我的核实 | 修复 |
|---|---|---|---|
| **洞 1** | `bpb` 全为 `null` 但未说明 `status: not_run` 语义 | **成立**。v3 只写了"无 GPU"，没定义 `not_run` 到底排除了哪些情形 | `submission.json` 新增 `status_note`（三态定义 + "难看的真数字也必须填"）+ `null_field_semantics`（逐字段交代产生时机）+ `how_to_fill_after_getting_gpu`（6 步）；README 顶部同步 |
| **洞 2** | 证据链断口：补丁引用的 `upstream/…train_gpt.py` 不在包内；`base_sha256.txt` 陈旧（停留在 1415 行版本） | **成立，且比评审说的更严重**：`_upstream/` 被 gitignore，评审拿到的包里连基线都没有，补丁根本没法 apply | 官方基线在 MIT 许可下复制进 `evidence/upstream/`；`.gitignore` 加 `!evidence/upstream/`；补丁头改成包内路径 + git 风格头；`base_sha256.txt` 改由 `pg_lab/hash_registry.py` 生成（含 apply 真跑自检、evidence 全量哈希） |
| **洞 3** | dry-run 跑的是改编版 `train_gpt_cpu.py`，不是在提交脚本上跑；且未覆盖 `fixed` 模式 | **前半成立但无法做得更好**：本机无 GPU，原文开头 `raise RuntimeError("CUDA is required")`，物理上跑不起来。**后半完全成立**：`fixed` 确实一次都没跑过 | 3→4 模式（补 `fixed`，另加边界档 `knee_tight`）；新增**同源性审计**：从磁盘两文件重新推导 diff，断言每条差异行都能被 11 条已声明补丁解释（实测 -22/+43，0 条未解释），并逐函数比对关键函数源码 |

### 五个问题（逐条回答，全部有可执行复现，见 `evidence/代码自查_评审五问.md`）

| 问题 | 结论 |
|---|---|
| 滑窗边界 `s = 0 if ws == 0 else max(wlen - stride, 0)` 是否正确 | **与官方 2026-03-19 记录逐字相同**。逐行 diff：官方 95 行 / 我的 103 行，差异 -13/+21，其中 20 条是注释/变量名/排版，**唯一实质新增是 forward token 审计** |
| stride 候选枚举 | 自动枚举取 `window` 的约数（保窗口对齐）+ 含 `window` 自身（保证退化路径在候选里）。**v4 修了两个真 bug**（见下） |
| `EVAL_COST_BUDGET` 口径 | `forward_tokens / scored_tokens`，1x = 官方朴素评测。**v4 补了运行时审计**：旧版 docstring 声称审计 forward token，实际累计的是计分 token（恒等于 1x，等于没审计）。现在真跑会打印 `amortized_cost`，fixed 模式实测 **3.992x vs 预测 4.0x** |
| `forward_logits` 是否真产出逐位置 logits | **是**。形状 `(bsz,seq,vocab)`；与官方 `forward()` 的损失差 `0.000e+00`；因果性（改 `x[0,j]` 不影响 `<j`）实测通过 |
| 训练/验证/量化/打包路径未改 | **未改**。补丁 5 个块、+349/-12 行，其中 4 块是纯新增（删除 0 行），唯一有删除的是最终评测调用那一段 |

### v4 自己抓出的三个真 bug（评审没提，是我读代码时撞出来的）

1. **预算分支与注释相反**：`pick_stride_by_knee` 的 else 分支（预算连 1x 都付不起）返回 `rows[-1]`——
   `rows` 是降序，`rows[-1]` 是 **stride 最小 = 代价最大** 的候选。注释写着"退化为朴素"，
   行为却正好突破预算这个硬约束。实测 `EVAL_COST_BUDGET=0.25` 时选出 `cost=8.0`（预算的 32 倍）。
2. **`KNEE_CANDIDATES` 含 0 会除零崩溃**：旧过滤只写 `if int(s) <= window`，0 和负数都放行。
   实测 `"0,-1,9999"` → `ZeroDivisionError`。
3. **审计日志言行不一**：见上表第 3 行。

**并且验证了这些修复不是恒真空转**：`evidence/回归测试_反向验证.py` 把旧实现抄回来，
用**完全相同**的断言跑一遍，必须 FAIL。结果 4/4 能区分新旧实现，其中两条给出了具体失败
数值（`cost=8.0` 突破预算、`ZeroDivisionError`）。

**一条我主动改回去的偏离**：v3 里我把 `eval_val_sliding` 的收尾写成 `base_model.train(prev)`
（保存/恢复 train 状态），而官方（root baseline 与 2026-03-19 记录两者）都是无条件的
`model.train()`。这更"正确"，但属于没必要的行为偏离，v4 已改回，缩小了攻击面。

### 本轮最值钱的一条（我的选择）

不是补的三个洞，而是 **`knee_tight` 那个边界档**：`EVAL_COST_BUDGET=0.5`（连 1x 都付不起）时，
程序打 WARN 后退化为朴素评测，其 `val_bpb:3.08445300` 与 `naive` 模式**逐位完全相同**。
这一条同时证明了三件事：预算不足的退化路径走的是官方朴素评测（不是随便挑个候选）、
修复 1 真的生效、`EVAL_STRIDE_MODE=naive` 与官方基线等价。**是对比出来的，不是声称出来的。**

### 仍未做

- **P0：真机复现 baseline 1.2244 ± 0.005** —— 无 GPU，本机无法做。步骤写死在 §五。
- 包内基线是本机已有上游克隆的副本，不能保证等于 GitHub 此刻最新版（上游比赛期间持续更新）。
  提交脚本对应的就是 sha256 `c6cae5e7…` 这一版，已在 `evidence/基线核对说明.md` 写明。
