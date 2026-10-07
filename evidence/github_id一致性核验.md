# github_id 全链一致性核验（回应评审 P1-3）

> 日期：2026-10-06 ｜ 交付人：卢怡然

## 核验方法与结果

| 检查项 | 命令/位置 | 结果 |
|---|---|---|
| C2G 提交卡 | `卢怡然_C2G_submission.json` | `"author": "卢怡然"`，`"github_id": "shuibubao--zz"` |
| 本机 git 全局配置 | `git config --global user.name` | `shuibubao--zz` |
| C2G 仓库局部配置 | `git -C C2G config user.name` | `shuibubao--zz`（继承全局，无冲突覆盖） |
| C1（翻译管线） | 全目录 grep `github_id|shuibubao` | 无 github_id 字段——C1 交付物为语料/术语表/脚本，无提交卡，**无需回填** |
| C2（AI for Math 论文） | 同上 | 无 github_id 字段——交付物为 paper.tex/pdf 等，无提交卡，**无需回填** |
| C2A（ΔE-Bench 提案） | 同上 | 无 github_id 字段，**无需回填** |
| C4（challenge-submit-guard 技能）、C4A（评审技能，进行中） | 同上 | 无 github_id 字段，**无需回填** |

## 结论

- 目前**唯一**带 `github_id` 字段的提交卡就是 C2G 这份，其值为 `shuibubao--zz`，与 git 全局 `user.name` 一致。
- 其余挑战（C1/C2/C2A/C4）的交付形态不含提交卡，不存在需要统一的所有者字段。
- 后续新挑战若出现提交卡，将直接沿用 `shuibubao--zz`，保证全链一致。
