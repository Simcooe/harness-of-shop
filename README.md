# ShopSimulator Harness：不训练模型，只改 harness

在 ShopSimulator（淘宝商品库的购物 agent 环境）上，用通用 agent 框架 [pi](https://github.com/earendil-works/pi-mono) 做底座，把购物场景的增强做成**可插拔插件**：加载时生效，不加载时退化为原版 pi。评测不只看最终是否买对，也看执行过程（需求逐条核对、购买前核验、虚假完成声明、首错定位、LLM 过程评审、成本）。

在单轮任务上，插件修好了接口反馈问题；在需求信息不全的多轮任务上，插件的提问策略让 agent "该问就问"。

## 结构

```
pi-shop/adapter.ts     环境适配层（始终加载）：10 个商店工具 → 环境 HTTP 接口；每回合只放行一个动作；
                       模型只看到白名单渲染的页面，gold/奖励/进度只写进日志；
                       多轮模式额外提供 ask_shopper（由模拟买家回答，每题最多 3 次，不占环境步数）
pi-shop/plugin.ts      场景插件 v1（可选加载），组件由 SHOP_PLUGIN 开关：
                         guard            合法性校验：动作在当前页无效时拦下并说明原因，不消耗环境步数
                         nudge            停止追问：没买就停下时追问
                         ask_hint         提问提示（多轮）：商品页上有用户的话确定不了的规格时，提示可以问
                         confirm_gate     购买前确认（多轮）：已选规格没有依据、也没问过用户时，拦一次购买
                         dialogue_ledger  对话台账（多轮，可选）
pi-shop/plugin_v0.ts   第一版插件（6 个组件），保留用于复现消融实验
prompts/               base.md 基线系统提示（所有组共用）；shopper.md 模拟买家提示
runner/run.py          批量运行：reset → 启动干净的 pi 进程 → release；--multiturn/--ask 跑多轮
scripts/               题目导出、模糊需求构造（make_multiturn.py）、轨迹查看、人工抽查清单
eval/                  过程评测（见下文）
data/                  test_ids.json（Final-200 Clean）、dev_ids.json（100 题，与测试集不重叠）、
                       rubrics/（冻结的需求约束）、multiturn/（模糊需求）；private/ 下是 gold，agent 侧不读
docs/                  harness_design.md 设计方案与实施记录；narrative.md 秋招叙事
```

## 运行

环境用 [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 的 ShopSimulator Environment v2.1，克隆在 `third_party/`（该仓库无 LICENSE，不随本项目分发）。本地部署时的改动：`termination.exact_repeat_limit` 和 `no_progress_limit` 设为 999（防循环交给 harness），`max_steps` 保持 35；服务只绑定 127.0.0.1。

```bash
SHOPSIM_ENV_SLOTS=16 bash third_party/shopping-grpo-longhorizon/scripts/start_environment.sh
```

单轮（原版 / 插件）：

```bash
python3 runner/run.py --run-name test_off --tasks data/test_ids.json --workers 8 --plugin off
```

```bash
python3 runner/run.py --run-name test_v1 --tasks data/test_ids.json --workers 8 --plugin all
```

多轮（B 模糊需求不能问 / C 能问 / D 能问 + 提问策略）：

```bash
python3 runner/run.py --run-name mt_B --multiturn data/multiturn/test.jsonl --plugin all --workers 5
```

```bash
python3 runner/run.py --run-name mt_C --multiturn data/multiturn/test.jsonl --ask --plugin guard,nudge --workers 5
```

```bash
python3 runner/run.py --run-name mt_D --multiturn data/multiturn/test.jsonl --ask --plugin all --workers 5
```

评测与比较：

```bash
python3 eval/evaluate.py runs/test_v1 --judge
```

```bash
python3 eval/compare.py runs/test_off runs/test_v1
```

```bash
python3 eval/recovery.py --multiturn data/multiturn/test.jsonl --A runs/test_v1 --B runs/mt_B --C runs/mt_C --D runs/mt_D
```

```bash
python3 scripts/show_traj.py runs/mt_D/task_1627 --eval
```

模型通过 `.pi-clean/models.json` 配置（独立于全局 `~/.pi`），密钥从 `.env` 的 `MODEL_PROXY_API_KEY` 读取。

## 评测（eval/evaluate.py）

只读日志、冻结的需求约束和离线 gold，不调用环境，不用环境奖励。能用日志判定的用规则，需要语义理解的才用 LLM，LLM 的结论都要引用原文并经程序核对。报告分面板，不合成总分。

**准备（每题一次，冻结）**
- 需求约束（`eval/rubric.py`）：DeepSeek V4 Flash 只读 query，拆成原子约束（品类、品牌、型号、功能、规格、预算、服务），标 hard/soft，附 query 原文（程序校验必须是原文子串）。预算由 LLM 写成规范形式后用规则解析。
- gold 自检（`eval/checker.py --gold-selfcheck`）：用同一套核对流程核对 gold 商品。core 口径通过：开发集 92/100，测试集 187/200；未通过的题多是标注矛盾（如需求要米白、gold 是杏色），报告里单列。

**七个面板**
1. **结果**：严格成功（买到 gold 且 gold 规格全中）；可接受购买：用商品库和购买记录做证据卡，逐条核对约束——预算用规则（"以内/不超过"是硬上限，"左右/上下/出头"上浮 20%），数值类规格由 LLM 报数、程序按 20% 容差判定，其余由 LLM 判定且必须引用证据卡原文。core 口径：品类、品牌、型号、规格、预算必须满足，功能类不被判违反。
2. **过程正确性**：每条 hard 约束在购买前是否出现在 agent 看过的页面上；自述核对（假装已买、虚假核验、声称价格与成交价不符）；成功分为"过程完备"和"幸运成功"。
3. **失败诊断**：对不可接受的失败做首错定位：检索失败、漏看、误拒、规格错选、规格漏选、无效动作、假完成、提前放弃、步数耗尽。
4. **过程质量**：DeepSeek V4 Pro 按 yyhdbl 博客的 5 个维度（搜索策略、候选利用、证据核验、决策质量、终止效率）打 0–2 分并引用步骤编号，看不到 gold 和奖励；同时回答 4 个可以用日志核对的客观问题，用来校准评审。
5. **harness 干预**：各类拦截和提醒的次数、模型是否原样重复被拦动作。
6. **效率与成本**：环境步数、浪费的步数、LLM 回合、token、耗时。
7. **多轮提问**（`eval/asking.py`）：提问率；每个问题问到了哪些约束（LLM 判定），分为有效（问到被删的约束）、冗余（问已知信息）、不必要（问用户没要求的方面）；被删约束的覆盖率；答案利用率（问到的约束最后满足没有）；不可接受的失败归因为"该问没问"或"问了没用"；模拟买家每次回答是否针对问题、忠于需求、有没有泄露没问到的需求。

## 结果（DeepSeek V4 Flash 非思考模式，temperature 0，每个配置跑一次）

### 单轮：Final-200 测试集

| 配置 | 严格成功 | 可接受购买 | 有购买 | 没买却声称已买 | 无效动作占比 | 平均环境步数 | 平均 prompt token |
|---|---|---|---|---|---|---|---|
| 原版 pi | 51% | 52% | 57% | 29% | 32% | 10.7 | 73K |
| 插件 v1（合法性校验 + 停止追问） | 89% | 90% | 98% | 0% | 0% | 9.1 | 60K |

- 配对检验：+38 个百分点（95% CI [+31, +45]），78 题变好、2 题变差，McNemar p = 5e-21。
- 不可接受的失败从 87 题降到 9 题。原版的首错主要是无效动作（30）、假完成（28）和检索失败（18）；v1 剩下的是步数耗尽（5）和规格选错或漏选（3）。
- LLM 评审 5 个维度的平均分从 1.13–1.60 升到 1.77–1.90；评审对 4 个客观问题的回答与日志一致率 85%–100%（原版）、100%（v1）。
- 原版的主要失败：环境对无效动作静默不响应但照样计步，模型在详情子页调用 buy_now 无效后误以为已购买，29% 的题以"购买已完成"结束却没有任何购买。

### 多轮：信息缺失时能否"该问就问"（Final-200，信息缺失组 164 题）

从需求里删掉规格类约束（颜色、尺码、容量、数量、套餐等），agent 只拿到删减后的模糊需求；模拟买家（qwen3.7-plus）只知道完整需求，不知道 gold。

| 组 | 需求 | 能否提问 | harness | 严格成功 | 可接受购买 | 平均提问 | 决策质量（评审） |
|---|---|---|---|---|---|---|---|
| A | 完整 | 否 | v1 | 92% | 90% | 0 | — |
| B | 模糊 | 否 | v1 | 52% | 53% | 0 | 1.59 |
| C | 模糊 | 能 | v1 | 75% | 76% | 0.43 | 1.69 |
| D | 模糊 | 能 | v1 + 提问提示 + 购买前确认 | 80% | 84% | 0.73 | 1.81 |

- **恢复率 = (D − B) / (A − B)**：可接受购买口径，模型自己问（C）恢复 63%，加提问策略（D）恢复 83%；严格成功口径 57% → 71%。开发集上分别是 59% → 89%。
- 配对检验（可接受购买）：B → C 40 题变好、2 题变差（p = 4e-10）；C → D 17 题变好、5 题变差（p = 0.017）；D 和 A 之间仍有显著差距（11 比 1，p = 0.006）。
- 提问行为：信息缺失组提问率 C 38% → D 70%，被删约束被问到 34% → 58%，问到之后照着选的比例都是 99%；"该问没问"导致的失败 27 题 → 10 题。
- 代价：对照组（35 题，需求本来就完整）的提问率从 11% 升到 40%，多是"这个规格可以吗"一类的确认，可接受购买率没有下降（86% → 89%）。
- 模拟买家质量：213 次回答 100% 针对问题，99.5% 忠于需求，0 次泄露没问到的需求。
- 边界：D 剩下的 10 题"该问没问"里，5 题是被删的信息决定"买哪件商品"而不是"选哪个规格"（商品只有一个规格，或模糊需求已经唯一确定了规格），提示没有触发。当前策略只覆盖规格层面的信息缺失。

### 开发集：第一版插件的组件消融（100 题，plugin_v0.ts）

| 配置 | 严格成功 | 有购买 | 没买却声称已买 | 无效动作占比 | 平均环境步数 | 平均 prompt token |
|---|---|---|---|---|---|---|
| 原版 pi（两次运行） | 59 / 55 | 65 / 64 | 25 / 28 | 34% / 36% | 12.6 / 12.8 | 89K / 91K |
| 插件全开（两次运行） | 86 / 88 | 98 / 99 | 0 / 1 | 0% / 0% | 10.2 / 9.7 | 104K / 100K |
| 只开 guard | 86 | 96 | 0 | 0% | 9.8 | 69K |
| 只开 finish | 80 | 93 | 4 | 38% | 15.2 | 103K |
| 只开 ledger | 75 | 86 | 11 | 34% | 14.1 | 116K |
| 只开 monitor | 66 | 77 | 15 | 30% | 13.0 | 97K |
| 只开 audit | 55 | 61 | 25 | 41% | 14.3 | 107K |
| 只开 contract | 54 | 63 | 28 | 36% | 14.1 | 113K |
| 去掉 guard | 86 | 95 | 0 | 31% | 15.3 | 146K |
| 去掉其他任一组件 | 87–92 | 94–99 | 0–2 | 0% | 9.6–10.3 | 82K–104K |

- guard 单独就达到全开水平，而且成本最低；finish 和 ledger 从别的路径缓解同一个问题，所以"逐个去掉"时看不出 guard 的作用，必须配合"只开一个"实验。audit、contract、monitor 没有可测的单独效果。
- 据此精简出 v1（只保留合法性校验和停止追问）：开发集严格成功 86%，与全开持平（5 题变好、5 题变差，p = 1），prompt token 从 104K 降到 74K。

## 已知局限

- 每个配置只跑了一次。模型服务在 temperature 0 下也不完全确定：开发集两次相同的原版运行有 22 题成败翻转；多轮对照组里 A、B 两组输入完全相同，严格成功也差了 9 个点（35 题）。小差异要看配对检验，不能只看百分比。
- 搜索结果直接展示每个商品的属性标签（`key_attributes`），gold 商品的标签多半能在题面里找到，绝对成功率可能高于论文设置；各组看到的一样，对比是公平的。
- 多轮只删了规格类信息，模拟买家是 LLM，回答风格比真实用户规整；没有做用户画像。
- 自动评测由人工抽查过 20 条单轮轨迹（docs/spotcheck_dev.md）；多轮新增的指标只经过开发集上的人工排查，还没做系统的人工校准。
