# 交接文档（2026-10-12 晚，第二版）

给接手的会话：先读完第 0、1、2 节再动手。项目根目录 `/Users/ywwl/long-horizon-harness`。设计细节和每一轮的实施记录在 `docs/harness_design.md`（第七节是本轮的论文多轮复现和新组件）。

## 0. 工作方式（用户明确要求过，必须遵守）

- **用中文回复，大白话，短。** 先说结论，再说细节。不要用自己起的代号、缩写和统计术语（p 值、置信区间要说成"差距明显"或"100 题上还看不出来"）。用户多次看不懂又长又密的汇报，直接回过"什么意思？"。写汇报时目标是自然写法的一半长。
- **只回答问到的问题。** 用户问某个具体来源（比如"ShopSimulator 这篇论文"）时，只讲那个来源、点明是哪篇，不要混进 yyhdbl 的文档或我们自己的设置，也不要顺带抛新方案。用户为此发过火。
- **先给方案、等确认再动手。** 每个新阶段先列出要做什么、哪些地方需要用户拍板，并附上推荐选项，等用户回复后再开工。用户要在面试里讲清楚每个设计决策，所以设计由用户拍板。实施中如果必须偏离已确认的设计，在汇报里单独说明改了什么、为什么。
- **实验不用太严谨，设计要站得住。** 用户原话："不用太严谨，只要设计是 ok 的就行了。" 每个配置跑一次即可，不做重复运行和区间；重点是设计合理、讲得清楚。
- **要有故事、有想象空间。** 这是秋招作品（投 agent 岗）。纯防守型的实验（基线阶梯、换模型、人工抽查）被用户认为"没有故事讲"；现有的插件组件被认为"没有想象空间"，像是针对失败一个个打补丁。新的工作要有清楚的设计思想，能推广到别的场景。
- **不擅自安装任何东西。**
- **测试集（Final-200）只在用户同意后跑**，不用测试集调参，调整只在开发集上做。
- **安全：**
  - 模型代理的地址和密钥只在 `.env`（权限 600），用 `MODEL_PROXY_API_KEY` 引用，绝不打印、不写进任何文件或提交。所有实验结束后提醒用户轮换密钥。
  - agent 绝不能看到 gold。runner 会去掉 reset 返回的 `goal_options`、`user_persona`、`reason_key`、`_trace_target`。pi 的内置工具全部禁用，agent 没有文件访问能力。`data/private/` 只给评测用。
  - 例外：论文多轮模式下，按官方做法，模拟买家（不是 agent）知道标准答案的规格选项，写在每题的 `runs/<run>/task_<id>/shopper.json` 里，agent 读不到。
  - 作者仓库和官方仓库都没有 LICENSE：代码和提示原文只放在 gitignore 的 `third_party/` 里，通过 HTTP 调用或运行时读取，不复制进项目目录，也不公开。
  - 环境服务只绑定 127.0.0.1。

## 1. 项目是什么、故事线怎么走到这里

**目标**：大厂秋招作品，投 agent 岗。ShopSimulator（阿里，淘宝商品库的购物 agent 环境）原本是后训练场景：论文用 SFT+RL，作者 yyhdbl 用 SFT+GRPO 把 2B 模型训到 62%。本项目反过来做：**不改模型权重，只改 harness**，同时评测执行过程，而不只看最终有没有买对。

**做法**：底座是通用 agent 框架 pi，购物增强做成可插拔插件，插件不加载时就是"原版 pi + 环境适配层"。环境用 yyhdbl 的 ShopSimulator Env v2.1，不用环境奖励。agent 模型是 DeepSeek V4 Flash，非思考模式，temperature 0。

**故事线的演变（本次会话）**：
1. 第一遍：单轮测试集 51% → 89%。但提升主要来自"拦下点错的动作"，用户担心被说成"修环境 bug"，没有故事。
2. 主线改为"会问问题的导购 agent"。ShopSimulator 论文本身就把多轮澄清当核心难点：多轮场景最强的 Claude-4-Sonnet 38%、GPT-5 29%，SFT+RL 训练后的 Qwen3-8B 约 35%，论文自己说这是没解决的挑战。论文除了提示词（"信息不全先问、下单前确认"）和训练，没有专门的提问方法。yyhdbl 直接跳过了多轮（只做单轮、禁止追问）。
3. 我们之前自己搭的多轮设置（删规格信息、最多问 3 次）和论文不同，分数不能和论文比。于是**照官方代码复现了论文的多轮规则**（只在开发集上跑过）。
4. 在论文设置下加了"收尾"和"规格选择题"两个组件，开发集严格成功：原版 43% → 最终组合 81%。
5. 用户觉得现在的设计"没有想象空间"。上个会话建议的下一步是**把任务状态显式化**（第 2.2 节），用户还没拍板，就让新会话接手。

## 2. 当前停在哪、下一步做什么

### 2.1 本轮已完成

- **论文多轮规则复现**（实现细节和与官方的差异见 `docs/harness_design.md` 第七节）：
  - 买家用官方配置里的提示（所有属性确认前一律拒绝下单），拼上完整需求和标准答案规格选项；开场先问买家"请提供您的购物需求。"，买家的回复（中位数 18 个字）就是 agent 收到的第一句话，每题冻结在 `data/multiturn_paper/`。
  - 提问不限次数，提问和操作合计 40 个回合；每条返回附"轮次: N"。
  - agent 直接回复的文字会被适配层转给买家，买家的回答作为下一条消息发回（官方流程只有下单或回合用完才结束）。
  - 买家模型用 qwen3.7-plus（官方 qwen3-235b 代理上没有）。
- **新组件**（都不在 "all" 里，"插件全开"仍可和之前比较）：
  - `closing` 收尾：剩 ≤12 回合起提醒剩余回合；剩 ≤6 回合或用户说了再见时，要求立即下单并附上看过的商品清单；这时不再做购买前确认和提问提示。提示里必须引用官方任务规则（"对话结束时用户仍不满意，就买最接近需求的那一件，不需要用户同意"），否则用户说"不买"后模型会拒绝下单。
  - `choice` 规格选择题：提问提示里把还没定的规格选项原样列出，让 agent 请用户直接挑。
- **开发集结果**见第 5 节。收尾是主要贡献（64% → 79%）；规格选择题作用小。
- 用户被告知过一个设计上的张力：收尾等于在买家拒绝后仍然下单。论文打分要求最后必须买，但论文的错误分析也把"买家拒绝后还下单"算作错误；真实产品里收尾应该是"推荐最合适的一款让用户决定"。

### 2.2 推荐的下一步：显式维护任务状态（第三版 harness，用户还没拍板）

**接手后第一件事：用大白话、简短地把这个方案和下面的拍板点给用户看，等用户确认再做。** 用户的原话是"让另一个会话执行后面的"，但第 0 节的规则仍然适用。

**动机（给用户讲的版本）**：现在的几个组件都是"看到一种失败、打一个补丁"，背后是同一个原因：agent 心里没有一份清楚的任务状态。用户要什么、哪些确认了、哪些还没问，看过的商品各满足哪几条，还剩多少回合，这些都埋在越来越长的对话里（论文设置下每题平均 18 万到 31 万个 prompt token，大部分在重复读历史），模型自己记不清，只好靠插件在各个时刻提醒。

**设计草图**：
- **需求表**：每条需求（品类、品牌、预算、颜色、尺码、功能……）标成"已确认 / 还不知道"，附用户原话。每次买家回答后用一个小模型从对话里抽取更新，只依据用户说过的话。
- **商品表**：对看过的每件商品，逐条记"满足 / 不满足 / 页面上没写"，只记页面上真实出现的事实（每次打开商品详情页后更新）。
- **基于两张表做决策**：缺哪条问哪条；商品对表满足了才买；下单前对照表格核验，不靠 agent 自述；回合不够时按表选最好的那件。
- **上下文瘦身**：agent 每一步只看系统提示、两张表和最近几步，不再带全部历史。
- 现有组件都能放进这个框架：提问提示 = 问表里缺的需求；购买前确认 = 对表核验；收尾 = 回合不够时按表选最好的；规格选择题 = 问缺的规格时把选项列出来。

**故事和想象空间**：
- 从"一堆补丁"变成"一个架构"，能讲清设计思想。
- 和 reference 里阿里的 LongHorizon-Harness（arXiv 2608.01964）一脉相承：它的核心是把任务状态放到执行之外维护、只用独立核验过的事实更新（Manage-Execute-Audit 循环：管理器维护状态并定下一个子任务，新上下文的执行器执行，只读的核验器核实环境状态）。但它只管环境里的状态；这里扩展到"用户需求"这种要靠对话才能拿到的状态，是它没做的。
- 能推广：
  - **个性化**：ShopSimulator 的个性化场景可以接进来，画像信息先记成"推测"，关键的再问用户，正好对着论文的两类个性化错误（没用上画像 55.8%、过度解读画像 35.7%）。
  - **多件商品、换领域**（订机票、订酒店）：都是往表里加东西。
- 新的可量化收益：token 大幅下降，提问更少更准；成功率不一定比 81% 高多少（已接近单轮"需求说全"时的 86%）。

**需要用户拍板的点（附推荐）**：
1. **表由谁维护**：每次买家回答或打开商品后调一次小模型（V4 Flash，非思考）抽取（推荐，通用）；还是用规则匹配（便宜，但不通用）。
2. **上下文**：只给"系统提示 + 两张表 + 最近 K 步"（推荐，K 先取 6 左右，不调）；还是保留全部历史、另外附表（改动小，但省不了 token）。
3. **决策由谁做**：仍由 agent 决定，表作为输入，插件在关键时刻提醒（推荐，改动适中）；还是像 LongHorizon-Harness 那样由管理器定下一个子任务、执行器只做子任务（更彻底，工作量大）。
4. **个性化要不要放进来**：开发集 16 题、测试集 38 题有改写需求，可以做；但 reset 对每题都返回 `user_persona`，可能是占位，要先核实（见第 8 节）。推荐先不放，第三版跑通后再加。
5. **对比组**：开发集上和 v2（`all,closing,choice`）比成功率、token、提问次数。

**实现提示**：
- pi 的 `context` 事件可以在每次调用模型前替换消息列表（`pi.on("context", e => ({ messages: [...] }))`，见 pi 的 `core/extensions/runner.js` 的 `emitContext`），用它实现"系统提示 + 两张表 + 最近 K 步"。注意保留工具调用和工具结果的配对，不要切断在一对中间。
- 页面状态通过适配层发出的 `shop:state` 事件拿到，买家问答通过 `shop:ask` 事件拿到（转发的文字也会触发 `shop:ask`）。
- pi 的事件总线是 Node EventEmitter，同步调用。插件可以在 `shop:relayed` 事件里往 `notes` 数组追加内容，适配层会附在发回 agent 的买家回答后面（收尾就是这么做的）。
- 在 TS 里调模型代理可以参考 `pi-shop/adapter.ts` 的 `shopperReply`（`${MODEL_PROXY_BASE_URL}/v1/chat/completions`，DeepSeek 关思考用 `thinking: {type: "disabled"}`）。抽取只能用 agent 能看到的信息（对话和页面），不能用 gold。
- 建议做成新插件组件或新插件文件，原来的 `plugin.ts` 组件保持可复现。
- 评测：`eval/evaluate.py` 已经报告 token、提问次数、开场没说的需求被问到的比例；可以加一个面板，把需求表和用户真实需求（`data/rubrics/`）对一下，看抽取准不准（评测侧可以看 gold 和完整需求）。

**备选方向 B（一句话给用户）**：harness 自动优化，让一个 agent 读评测器给出的失败轨迹和首错定位，自动提出新的 harness 规则并在开发集上验证，保留有提升的。更吸引眼球，但结果不确定，适合在面试里当"下一步"来讲。

### 2.3 测试集（论文多轮设置，已跑完）

用户同意后，四组各跑一次并带 LLM 评审：原版 50%、论文提示词 58%、插件全开 64%、v2 84%（严格成功），结果见第 5 节和 `docs/harness_design.md` 第七节。评审发现 v2 的"决策质量"没有提高：收尾让 agent 在用户拒绝后仍然下单，评审认为这违背用户意愿。第三版如果要做，可以把这一点当作要改进的地方（比如收尾时先把最接近的商品推荐给用户、说明差在哪里，而不是直接下单）。

第三版做完后，测试集只需要再跑第三版这一组，和这四组配对比较。

## 3. 代码结构

```
pi-shop/adapter.ts     环境适配层（所有组都加载）
                       - 10 个商店工具 → POST http://127.0.0.1:5700/api/shop_agent
                       - 每个 LLM 回合只放行一个动作（pi 默认会并行执行同一条消息里的多个工具调用）
                       - 页面按白名单渲染；gold/奖励/进度只写进日志
                       - SHOP_ASK=1 时注册 ask_shopper，读 SHOP_SHOPPER_FILE 里的模拟买家配置（system、model、
                         max_tokens、可选的预置对话 history），调 qwen3.7-plus（temperature 0，enable_thinking=false）
                       - 我们自己的多轮：每题最多 3 次，不占环境步数
                       - 论文多轮（SHOP_MAX_QUESTIONS=0）：提问不限次数、计入回合上限；每条返回附"轮次: N"；
                         agent 直接回复的文字会转给买家（ask_log 里 via=text），回答连同插件追加的提示作为下一条消息发回
                       - 通过 pi.events 发出 shop:state / shop:done / shop:ask / shop:relayed 给插件
pi-shop/plugin.ts      插件 v1，SHOP_PLUGIN 选组件（"all" = guard,nudge,ask_hint,confirm_gate）
                       - guard 合法性校验；nudge 没买就停下时追问（最多 2 次；论文模式下文字会转给买家，基本用不上）
                       - ask_hint 提问提示；confirm_gate 购买前确认（每件商品拦一次）；dialogue_ledger 可选
                       - closing 收尾、choice 规格选择题：论文多轮设置下加的，"all" 不含
                       - 多轮组件只在 SHOP_ASK=1 时生效；SHOP_MAX_QUESTIONS=0 时不提示"还能问几次"
pi-shop/plugin_v0.ts   第一版插件（guard,ledger,monitor,audit,finish,contract），原样保留以复现消融
prompts/base.md        基线系统提示（所有组共用，只有一句话）
prompts/shopper.md     我们自己多轮设置的模拟买家提示（已冻结）
runner/run.py          批量运行：reset → 写 reset_public.json → 启动干净的 pi 进程 → release → done.json
                       参数：--tasks/--task-ids/--limit、--run-name（必填）、--model、--plugin off|all|组件列表、
                       --plugin-file、--multiturn <jsonl>、--ask、--shopper-model、--max-questions、--workers 等。
                       --paper-openings <jsonl> 开启论文多轮模式（自动 ask、不限提问次数、40 回合上限、官方买家提示、
                       买家回答最多 512 token）。run_config.json 里记 protocol = single / ours / paper。
                       已有 done.json 的题会跳过（中断后重跑同名即可续跑）。环境槽位满时 reset 会等待重试。
runner/paper_protocol.py   论文多轮规则：按官方 shopper.reset 拼买家系统提示（官方买家提示 + 完整需求 + 标准答案规格选项）
scripts/export_tasks.py    导出 data/test_ids.json（作者 Final-200）、data/dev_ids.json、data/private/gold.jsonl
scripts/make_multiturn.py  构造模糊需求（删掉规格类约束，规则 + LLM 校验）→ data/multiturn/<split>.jsonl
scripts/make_paper_openings.py  论文多轮的买家开场白（每题生成一次并冻结）→ data/multiturn_paper/<split>_openings.jsonl
scripts/show_traj.py       看一条轨迹：python3 scripts/show_traj.py runs/<run>/task_<id> [--full] [--eval]
scripts/make_spotcheck.py  生成单轮人工抽查清单（docs/spotcheck_dev.md + docs/traces/）
scripts/run_ablation.sh / run_addone.sh   v0 消融（已指向 plugin_v0.ts）；run_test.sh 为第二遍重复实验准备
eval/                  评测，见第 6 节
data/rubrics/          300 题冻结的需求约束（开发集 100 + 测试集 200）
data/multiturn/        dev.jsonl（100 题：信息缺失 83 + 对照 17）、test.jsonl（199 题：164 + 35；1 题校验没过，在 test_rejected.jsonl）
data/multiturn_paper/  论文多轮的开场白：dev_openings.jsonl（100 题）、test_openings.jsonl（200 题）
data/cache/            products_min.json、gold_selfcheck_{dev,test}.jsonl、shopper_probe_dev.jsonl、llm/（LLM 调用缓存）
third_party/shopsim_official/  从官方仓库取的提示（没有许可证，gitignore）：
                       shopper_paper.md（官方配置 multi_eval/configs/standard/qwen3_8b.yaml 里的买家提示 + 任务模板）、
                       agent_paper.md（官方 agent 提示改写成工具版，"论文提示词"对比组用）
third_party/shopsim_hf/       官方 HF 数据（商品库、画像数据）
docs/                  harness_design.md（设计方案和实施记录）、narrative.md（秋招叙事、简历写法、面试问答，还没按新主线改写）、
                       spotcheck_dev.md + traces/（单轮人工抽查，用户已确认）
reference/             论文 shopsimulator.pdf、LongHorizon-Harness.pdf、作者博客 yyhdbl-shopping-agent-rl.html
PLAN.md                最早的计划，已过时；以 README、docs/ 和本文件为准
```

官方 multi_eval 的源码（agent.py、shopper.py、env.py、配置）只下载到 `/tmp/shopsim_official/` 读过，没有放进项目；需要时用 curl 从 raw.githubusercontent.com 取（GitHub API 和 WebFetch 在这里不通）。

## 4. 怎么跑

**环境服务**：在 app 的终端标签 "shopsim env" 里运行（16 个槽位，端口 5700）。检查是否在跑（返回 200 即可）：

```bash
curl -s -m 5 -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:5700/api/shop_agent -H 'Content-Type: application/json' -d '{"action":"noop"}'
```

没在跑就启动：

```bash
SHOPSIM_ENV_SLOTS=16 bash third_party/shopping-grpo-longhorizon/scripts/start_environment.sh
```

对 `third_party/` 的本地改动：环境配置里的 `exact_repeat_limit`、`no_progress_limit` 设为 999（原文件备份为 `environment.original.json`），`max_steps` 保持 35；`pack_api.py` 的 SERVER_HOST 从 `SHOPSIM_HOST` 读，默认 127.0.0.1。

**运行**：同时跑的几组 workers 加起来不要超过 16（论文多轮每组 5 个，三组并行约 40 分钟跑完 100 题）。

单轮、我们自己的多轮：

```bash
python3 runner/run.py --run-name test200_v1 --tasks data/test_ids.json --workers 8 --plugin all
```

```bash
python3 runner/run.py --run-name test_mt_D --multiturn data/multiturn/test.jsonl --ask --plugin all --workers 5
```

我们自己的多轮四组：A 直接用单轮 v1 的结果；B `--multiturn --plugin all`（不加 `--ask`）；C `--multiturn --ask --plugin guard,nudge`；D `--multiturn --ask --plugin all`。

论文多轮：开场白已为开发集和测试集生成；新的题目列表用这个命令生成：

```bash
python3 scripts/make_paper_openings.py --tasks data/test_ids.json --out data/multiturn_paper/test_openings.jsonl
```

然后各组（把 `<split>` 换成 dev 或 test）：
- 原版：`--paper-openings data/multiturn_paper/<split>_openings.jsonl --plugin off`
- 论文提示词：同上，加 `--system-prompt-file third_party/shopsim_official/agent_paper.md`
- 插件全开：`--plugin all`
- v2：`--plugin all,closing,choice`

```bash
python3 runner/run.py --run-name dev_paper_v2 --paper-openings data/multiturn_paper/dev_openings.jsonl --plugin all,closing,choice --workers 5
```

**评测**：`--judge` 会调 V4 Pro 做过程评审，有缓存，可以多个评测并行跑。开发集一般不带评审。

```bash
python3 eval/evaluate.py runs/dev_paper_v2 --workers 8
```

```bash
python3 eval/compare.py runs/dev_paper_all runs/dev_paper_v2
```

```bash
python3 eval/recovery.py --multiturn data/multiturn/test.jsonl --A runs/test200_v1 --B runs/test_mt_B --C runs/test_mt_C --D runs/test_mt_D
```

每次评测都会写 `runs/<run>/eval/tasks.jsonl`（逐题记录）和 `summary.txt`（面板）。

## 5. 已有结果

### 单轮，测试集 200 题

| 配置 | 严格成功 | 可接受购买（core） | 有购买 | 没买却声称已买 | 无效动作占比 | 平均步数 | 平均 prompt token |
|---|---|---|---|---|---|---|---|
| 原版 pi | 51% | 52% | 57% | 29% | 32% | 10.7 | 73K |
| v1 | 89% | 90% | 98% | 0% | 0% | 9.1 | 60K |

- +38 个点，78 题变好、2 题变差，McNemar p = 5e-21。
- v1 的"无效动作 0%"是因为被拦下了：模型在 153 题里做了 243 次无效尝试，按尝试算约 12%（原版 32%）。"没买就停下时追问"只触发 4 次，自动修正参数 0 次，所以 v1 的提升几乎全来自合法性校验。
- 开发集：v1 严格成功 86%、core 90%；v0 消融见 README（只开 guard 就达到全开水平）。

### 我们自己的多轮设置，测试集信息缺失组 164 题

| 组 | 严格成功 | 可接受购买 | 平均提问次数 |
|---|---|---|---|
| A 完整需求 | 92% | 90% | 0 |
| B 模糊需求，不能问 | 52% | 53% | 0 |
| C 模糊需求，能问 | 75% | 76% | 0.43 |
| D 能问 + 提问策略 | 80% | 84% | 0.73 |

恢复率（可接受购买口径）C 63%、D 83%；C → D p = 0.017。对照组提问率 11% → 40%，成功率没降。模拟买家 213 次回答 0 次泄露。这套设置和论文不同，分数不能和论文比。

### 论文多轮设置，开发集 100 题（每组一次，不带评审）

| 组 | `--plugin` | 严格成功 | 可接受购买 | 有购买 | gold 规格全选对 | 平均 prompt token |
|---|---|---|---|---|---|---|
| 原版 | off | 43% | 47% | 57% | 52% | 309K |
| 论文提示词 | off + agent_paper.md | 54% | 54% | 62% | 75% | 270K |
| 插件全开 | all | 64% | 65% | 70% | 69% | 245K |
| + 规格选择题 | all,choice | 69% | 73% | 82% | 73% | 199K |
| + 收尾 | all,closing | 79% | 85% | 100% | 79% | 197K |
| v2 | all,closing,choice | 81% | 86% | 100% | 81% | 182K |

- 原版 → 插件全开 +21（p = 0.0003）；插件全开 → 加收尾 +15（p = 0.0015）；加规格选择题单独 +5（p = 0.42）、叠在收尾上 +2（p = 0.82）。
- 原版 → v2 +38（42 题变好、4 题变差，p = 5e-9）；论文提示词 → v2 +27（p = 4e-5）。
- 没加收尾时，没下单的题几乎都是 40 个回合用完：互相说再见（买家拒绝后双方一直说再见），或核实僵局（买家要确认页面上没有的信息，agent 一直找）。论文提示词里写了"买家说再见就直接买"，这一组仍有 38 题没买。
- v2 剩下 19 题严格失败：10 题买到符合需求的替代品，6 题买错商品，2 题商品对规格错，1 题标注矛盾。
- 开场白平均只说出 21% 的需求约束。
- 和论文数字只能当参考：论文多轮最强 38%、训练后约 35%，我们的原版已经 43%，但模型、题目子集、买家模型都不同。

### 论文多轮设置，测试集 200 题（每组一次，带评审）

| 组 | 严格成功 | 可接受购买 | 有购买 | 决策质量均分 | 平均 prompt token |
|---|---|---|---|---|---|
| 原版 | 50% | 50% | 54% | 1.62 | 300K |
| 论文提示词 | 58% | 59% | 65% | 1.53 | 230K |
| 插件全开 | 64% | 68% | 78% | 1.69 | 207K |
| v2 | 84% | 89% | 100% | 1.62 | 185K |

- 原版 → v2 +34（76 题变好、8 题变差，p = 5e-15）；论文提示词 → v2 +25.5（p = 2e-9）；插件全开 → v2 +20（p = 1e-8）。
- 原版 → 插件全开 +14（p = 0.002）；原版 → 论文提示词 +8.5（p = 0.06）；论文提示词 → 插件全开 +5.5（p = 0.27）。
- v2 决策质量 0 分的 28 题里 27 题触发过"现在必须下单"（16 题其实买对了），评审理由都是"用户拒绝后仍下单"。

### runs 目录索引

| 运行 | 内容 | 是否带评审 |
|---|---|---|
| dev100_off / dev100_off_r2 | 开发集原版，两次 | 第一次带 |
| dev100_on / dev100_on_r2 | 开发集 v0 全开，两次 | 第一次带 |
| dev100_only_* / dev100_no_* | v0 消融 | 否 |
| dev100_v1 | 开发集 v1 | 否 |
| dev_mt_B/C/D | 开发集，我们自己的多轮 | 否 |
| test200_off / test200_v1 | 测试集单轮 | 是 |
| test_mt_B/C/D | 测试集，我们自己的多轮 | 是 |
| dev_paper_off / dev_paper_prompt / dev_paper_all | 开发集，论文多轮：原版 / 论文提示词 / 插件全开 | 否 |
| dev_paper_choice / dev_paper_closing / dev_paper_v2 | 开发集，论文多轮：全开 + 规格选择题 / 全开 + 收尾 / 全开 + 两者 | 否 |
| test_paper_off / test_paper_prompt / test_paper_all / test_paper_v2 | 测试集，论文多轮：原版 / 论文提示词 / 插件全开 / v2 | 是 |
| smoke* | 冒烟测试，可忽略 | 否 |

## 6. 评测器（单轮部分已由用户抽查确认并锁定）

- **需求约束**（`eval/rubric.py`）：V4 Flash 只读 query，拆成原子约束（category/brand/model/function/spec/budget/service），标 hard/soft，附原文片段，冻结在 `data/rubrics/`。
- **gold 自检**（`eval/checker.py --gold-selfcheck --split dev|test`）：core 口径开发集 92/100、测试集 187/200 通过；没通过的题记为"标注矛盾"，单列。
- **核对**（`eval/checker.py`）：用商品库和购买记录做证据卡，逐条核对约束。预算"以内"是硬上限，"左右/出头"上浮 20%；数值类按 20% 容差；其余由 LLM 判定并引用证据卡原文。**主口径是 core**（用户定的）：品类、品牌、型号、规格、预算必须满足，功能类不被判违反即可。
- **过程**（`eval/process.py`）：购买前核验、自述核对（假装已买、虚假核验、价格不符）、首错定位、插件干预统计。
- **评审**（`eval/judge.py`）：V4 Pro，作者博客的 5 个维度，0–2 分，引用步骤；外加 4 个客观问题和日志对照来校准。多轮时告诉评审完整需求和这一组能不能提问；论文模式有单独的说明（`PAPER_NOTE`）。
- **多轮提问**（`eval/asking.py`，第 7 个面板）：问题问到了哪些约束，分有效、冗余、不必要；该问没问、问了没用；答案利用率。
  - 论文模式没有"删掉了哪些约束"的记录：hidden 取开场白里没说出来的约束（LLM 判断，有缓存）。
  - 论文模式不做买家回答检查，因为买家知道标准答案的规格选项，只看需求原文会误判"不忠实"。
  - agent 直接回复的文字也算一次提问（包括"再见"之类），所以原版和论文提示词组的"不必要提问"偏高。
- **多轮指标还没做系统的人工校准**（成功率用的是已锁定的单轮核对器，没问题；提问类指标没校准）。
- **失败原因诊断**（`eval/diagnose.py`，`evaluate.py --diagnose`，2026-10-13 在 task-state 分支加的）：首错定位只说得出错在哪一步，这里回答为什么错。原因按事实由规则判定，不让模型直接挑（最初版本让 V4 Pro 读轨迹挑原因，抽查 22 道错 3 道，都在"没问需求 / 没回头 / 搜索不对"的分界上，输入稍一改就有 7/93 道变了）。
  - 用到的事实：每条需求买的那件和正确商品各自满不满足（评测器核对 + 标准答案自检 `data/cache/gold_selfcheck_*.jsonl`）；每条需求用户第几回合说过（V4 Flash 判断，必须引用原话，原话在对话里找不到就不算）；正确商品什么时候出现过、harness 拦过哪些购买（日志）；买对了商品但规格不对时，按用户的话和预算能不能确定选哪个（V4 Pro 只回答选哪个）；正确商品没搜到时，单轮实验的运行用什么词搜到过它，以及这些词按用户当时说过的话想不想得到（V4 Pro 逐词判断）。
  - 判定顺序：系统拦下了正确的购买 → 动作无效 → 没买成 → 买对了商品时看规格（单选项没点 = 规格没点；能推出来 = 规格没点 / 规格选错；推不出来 = 没问清需求）→ 正确商品满足的买的也都满足 = 替代品也满足需求 → 单轮实验也从没搜到 = 正确商品难搜到 → 用户说过、正确商品不满足而买的满足 = 标准答案有问题 → 用户说过某条正确商品更好的条件之后正确商品又出现 = 没回头（买的那件明确不满足的条件优先，这些全是用户没说过的就是没问清需求）→ 之后没再出现：对照的搜索词想得到 = 搜索词不对，否则没问清需求。"回合用完被迫下单""追查页面上没有的信息"作为附加标签。
  - 准确性：设计时看过两批，23 道全部对上，27 道里 22 道对上、5 道有争议，但都参与了定规则；没参与设计的 dev_paper_closing 先盲标 15 道再对答案，14 道一致（不一致的那道是页面本身说不清该选哪个规格）；提示词改几个字重跑 84 道，原因全部不变。抽查清单 `docs/diagnosis_check.md`（`scripts/make_diagnosis_check.py` 生成，附 Claude 的判断，**等用户确认**）。
- **拦截审计**（`process.buy_blocks`，第 5 个面板）：harness 每拦下一次购买，看被拦时选的是不是正确商品和全对的规格（放行会不会严格成功），以及这些题最后有没有买对。
- **状态卡的规格标注**（`state_check.card_specs`，第 8 个面板）：状态卡给正确商品标的"用户要"规格和正确规格对不对得上。

## 7. 用户拍板过的决策

- 底座用原版 pi，购物插件可插拔；复用 yyhdbl 的环境，屏蔽奖励。
- 评测：环境设置 A（属性标签直接显示在页面上），评审用作者的 5 个维度加 V4 Pro，主口径 core。单轮抽查通过后评测器锁定。
- 第一遍 harness 设计（`docs/harness_design.md` 第一到六节，全部按推荐）：单轮 v1 只保留合法性校验和停止追问；没有内容的详情子页保持现状；模糊需求只删规格类约束；模拟买家用 qwen3.7-plus；每题最多问 3 次、不占环境步数；提问策略用 ask_hint + confirm_gate。
- 本次会话：
  - 不采用"基线阶梯 / 换模型 / 多轮抽查"作为主线（没有故事）。
  - 方向是"从 harness 优化"，主线是会问问题的导购 agent。
  - 在我们的运行脚本里照官方代码复现论文的多轮规则（"可以，那你做吧"）。
  - 先做收尾和规格选择题（"那你先优化 harness 吧"）；实验不用太严谨。
  - "论文提示词"对比组是会话自己加的，汇报过，用户没有反对。
  - 测试集上论文多轮四组已跑完并评测（"那后续就先跑下测试集和测评吧"）。
- 尚未拍板：第三版 harness 的方向和第 2.2 节的拍板点；测试集什么时候跑；个性化要不要做；空详情子页要不要拦（之前定的是保持现状，论文设置下回合是瓶颈，约五分之一的环境步花在空子页上，可以重新提给用户）。

## 8. 坑和注意事项

- **环境的特点**（细节见 README）：
  - 无效动作会被静默忽略，但照样扣一步。
  - 详情子页没有信息：Description 就是标题，Features 和 Reviews 为空。论文设置下买家常要求确认页面上没有的信息，这是核实僵局的来源。
  - 两个以上规格项都影响价格时，环境不显示规格价。
  - 环境自带的判定有假阴性：把"左右"当成硬上限。
  - 环境步数上限 35；论文设置的 40 个回合里包括提问，很少触及 35。
- **模型不完全确定**：temperature 0 下，两次相同的运行也有约 20% 的题成败翻转，看配对比较。模型有时会用英文回复。
- **画像数据**：之前以为公开数据没有个性化场景要用的改写需求，其实有：`third_party/shopsim_hf/fine_items_*_persona.jsonl` 每条的 `instructions[]` 里有 `instruction_simple`，环境 reset 也会返回。开发集 16 题、测试集 38 题的 `instruction_simple` 和原需求不同。但 reset 对所有 300 题都返回非空的 `user_persona`，可能是占位，用之前先核实。
- **Python 是 3.9**：不能写 `str | None` 这类类型注解；f-string 的花括号里不能有反斜杠。
- **zsh**：glob 要加引号（如 `--include='*.py'`）；`for x in "a b"` 不会按空格拆开；`set -- $p` 也不会拆，写成函数传参。Bash 工具里 `cd` 会改变后续命令的工作目录，尽量用绝对路径。
- **pi**：
  - tool_call 钩子按加载顺序串联（先适配层、后插件），第一个拦截的生效；同一事件的多个处理函数也按加载顺序依次执行。
  - 用了 `--system-prompt` 时，pi 会把 cwd 追加进系统提示，所以 runner 统一用 `runs/.cwd` 作为 cwd。
  - usage 里的 `input` 不含缓存命中；prompt token = input + cacheRead。
  - 模型只回复文字、不调工具时，pi 会结束这一轮。论文模式靠适配层在 `agent_end` 里转发给买家并续上对话。
  - `pi.events` 是同步的 Node EventEmitter；`context` 事件可以替换发给模型的消息列表。
- **换模型**：`.pi-clean/models.json` 现在只配了 `deepseek-v4-flash`。新模型在 yowant provider 下加条目，用 `--model yowant/<id>` 指定。关思考的参数各家不同：DeepSeek `thinking: {type: disabled}`，Qwen `enable_thinking: false`，GPT 系列要先试。代理上可选 `gpt-6-astra`、`gpt-6-luna`、`gpt-6-sol`、`gpt-6.1-sol`、`qwen3.8-flash`、`qwen3.8-max`；`deepseek-v4-pro` 是评审模型，不能当 agent；多轮换模型要避开 Qwen 家族（模拟买家是 qwen3.7-plus）。
- **LLM 缓存**（`data/cache/llm/`）是原子写入，可以多个评测并行跑。
- **auto 模式**的安全分类器偶尔不可用，Bash 会被拒；等一下重试即可。
- 项目还不是 git 仓库。`.gitignore` 里已有 `.env`、`third_party/`、`runs/`、`data/private/`；公开前还要考虑 `data/cache/`、含 query 原文的 `data/rubrics/`、`data/multiturn/`、`data/multiturn_paper/`（数据集许可不明）。
- `docs/narrative.md`（简历和面试问答）还是旧主线，没按"会问问题的导购 agent + 论文多轮设置"改写；README 也还没写论文多轮的结果。

## 9. 不再是主线的旧方案（需要时参考）

1. **基线阶梯（最简反馈）**：插件新组件，例如 `feedback`。tool_call 时记下执行前的页面，tool_result 时和执行后比较（page_type、商品 asin、已选规格、搜索词和页码），没变化就追加一句"动作没有生效，页面没有变化。" 不和 guard 同时开。在测试集跑 `--plugin feedback`，和 test200_off、test200_v1 组成三级阶梯。
2. **换模型**：按第 8 节加模型配置，每个候选先跑 10 题开发集摸底，选一强一弱；测试集上每个模型跑原版和 v1，带评审，用 compare.py 配对比较。
3. **多轮人工抽查**：参照 `scripts/make_spotcheck.py`，从多轮运行的评测结果按类别抽约 15 条（有效提问、该问没问、问了没用、不必要提问、模拟买家的回答），每条用大白话写评测器的结论，附相关几步轨迹和是/否问题，输出 `docs/spotcheck_multiturn.md`。
4. **搜索前先问清需求 / 按候选商品的差异提问**：用户只说了品类时，在第一次搜索前提示先问清预算、品牌、款式和必须满足的条件；搜到结果后，候选商品在某个用户没说过的方面差别很大时，提示去问这一项。如果做第三版的需求表，这条自然包含在里面。
