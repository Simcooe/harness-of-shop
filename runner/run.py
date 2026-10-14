#!/usr/bin/env python3
"""批量运行：每道题 reset 环境 → 启动一个干净的 pi 进程 → release 环境槽位。

agent 侧只拿到需求原文；gold、画像等字段不进入 pi 进程。每题的日志写在 runs/<run>/task_<id>/ 下。
多轮模式（--multiturn）：agent 拿到的是删掉规格信息的模糊需求；加 --ask 时提供 ask_shopper，
完整需求只写进模拟买家读的文件（agent 没有文件工具）。
论文多轮模式（--paper-openings）：照 ShopSimulator 官方 multi_eval 的规则。agent 拿到的是买家的开场白
（scripts/make_paper_openings.py 冻结），买家知道完整需求和标准答案的规格选项，需求没问全时拒绝购买；
提问不限次数，但和商店动作一起计入 40 个回合的上限。
任务状态层（--state）：在插件之后加载 pi-shop/task_state.ts，维护需求表和商品表并按表提问、核对、收尾；
--state-window K 只给模型最近 K 步原文（0 表示保留全部历史）。
分工模式（--roles，第四版）：主 agent 是导购（prompts/roles_sales.md），不能直接操作商店；买手由 pi-shop/roles.ts
在同一个进程里另开 pi 会话（prompts/roles_buyer.md），会话记录在 task_<id>/buyer_sessions/。只用于论文多轮设置，不加载插件和任务状态层。
画像设置（--persona，配合按 --persona 生成的开场白）：agent 的系统提示末尾附上用户画像（prompts/persona_agent.md），
买家只知道简版需求和规格选项（runner/paper_protocol.py）。画像另存在 task_<id>/persona.json。
画像分级（--profile-tiers，只和 --persona --roles 一起用）：第四版之上的改进，系统按来源把画像整理进需求单（pi-shop/roles.ts），
导购的系统提示在画像段之前加上 prompts/roles_profile.md。
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from paper_protocol import (
    OPENING_REQUEST,
    PERSONA_AGENT_BLOCK,
    PERSONA_SHOPPER_RULE,
    SHOPPER_PROMPT as PAPER_SHOPPER_PROMPT,
    persona_agent_block,
    persona_profile,
    persona_shopper_system,
    shopper_system,
    short_query,
)

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_RESET_KEYS = {"goal_options", "user_persona", "reason_key", "_trace_target"}
DEFAULT_PLUGIN = ROOT / "pi-shop" / "plugin.ts"
ADAPTER_PATH = ROOT / "pi-shop" / "adapter.ts"
STATE_PATH = ROOT / "pi-shop" / "task_state.ts"
ROLES_PATH = ROOT / "pi-shop" / "roles.ts"
NEUTRAL_CWD = ROOT / "runs" / ".cwd"
SHOPPER_PROMPT = ROOT / "prompts" / "shopper.md"
BASE_PROMPT = ROOT / "prompts" / "base.md"
SALES_PROMPT = ROOT / "prompts" / "roles_sales.md"
BUYER_PROMPT = ROOT / "prompts" / "roles_buyer.md"
PROFILE_PROMPT = ROOT / "prompts" / "roles_profile.md"

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def load_dotenv(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_call(url: str, payload: dict, timeout: float = 120) -> dict:
    req = Request(
        f"{url}/api/shop_agent",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    result = body.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"环境返回格式错误: {str(body)[:200]}")
    if result.get("error"):
        raise RuntimeError(f"环境错误: {result['error']}")
    return result


def load_multiturn(path: str) -> dict:
    if not path:
        return {}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["task_id"]: r for r in rows}


def load_task_ids(args, multiturn: dict) -> list:
    if args.task_ids:
        ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    elif multiturn:
        ids = list(multiturn)
    elif args.paper_openings:
        ids = [json.loads(line)["task_id"] for line in Path(args.paper_openings).read_text().splitlines() if line.strip()]
    else:
        path = Path(args.tasks)
        if path.suffix == ".jsonl":
            ids = [json.loads(line)["task_id"] for line in path.read_text().splitlines() if line.strip()]
        else:
            ids = json.loads(path.read_text())
    return ids[: args.limit] if args.limit else ids


def query_from_reset(reset: dict) -> str:
    text = str(reset.get("instruction") or "").strip()
    if text.lower().startswith("instruction:"):
        text = text[len("instruction:"):].strip()
    if not text:
        raise RuntimeError("reset 没有返回需求文本")
    return text


def run_task(task_id: int, args, run_dir: Path, base_env: dict, system_prompt: str, multiturn: dict, openings: dict) -> dict:
    task_dir = run_dir / f"task_{task_id}"
    done_file = task_dir / "done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())
    task_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record = {"task_id": task_id, "returncode": None, "error": None}
    for attempt in range(30):
        try:
            reset = env_call(args.env_url, {"action": "reset", "idx": task_id})
            break
        except RuntimeError as exc:
            # 环境槽位暂时占满时等一会儿再试
            if "available environment resource" not in str(exc) or attempt == 29:
                raise
            time.sleep(10)
    env_idx = reset["env_idx"]
    try:
        query = query_from_reset(reset)
        agent_query = query
        if multiturn:
            mt = multiturn[task_id]
            if mt["full_query"].strip() != query:
                raise RuntimeError("多轮数据的完整需求和环境给的需求不一致")
            agent_query = mt["vague_query"]
        elif openings:
            opening = openings[task_id]
            if opening["full_query"].strip() != query:
                raise RuntimeError("开场白数据的完整需求和环境给的需求不一致")
            agent_query = opening["opening"]
        task_prompt = system_prompt
        shopper_query = query
        if args.persona:
            profile = persona_profile(reset)
            shopper_query = short_query(reset)
            if not opening.get("persona") or str(opening.get("shopper_query", "")).strip() != shopper_query:
                raise RuntimeError("开场白不是按这道题的简版需求生成的")
            profile_how = f"\n\n{PROFILE_PROMPT.read_text(encoding='utf-8').strip()}" if args.profile_tiers else ""
            task_prompt = f"{system_prompt}{profile_how}\n\n{persona_agent_block(profile)}"
            (task_dir / "persona.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        public = {k: v for k, v in reset.items() if k not in PRIVATE_RESET_KEYS}
        public["query"] = query
        public["agent_query"] = agent_query
        if args.persona:
            public["shopper_query"] = shopper_query
        (task_dir / "reset_public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
        shopper_file = task_dir / "shopper.json"
        if args.ask:
            shopper = {"model": args.shopper_model, "max_tokens": args.shopper_max_tokens}
            if openings:
                # 官方做法：买家还知道标准答案的规格选项。agent 没有文件工具，读不到这个文件。
                goal_options = reset.get("goal_options")
                shopper["system"] = (persona_shopper_system(shopper_query, goal_options, args.shopper_prompt_file) if args.persona
                                     else shopper_system(query, goal_options, args.shopper_prompt_file))
                shopper["history"] = [{"question": OPENING_REQUEST, "answer": agent_query}]
            else:
                system = args.shopper_prompt_file.read_text(encoding="utf-8").strip()
                shopper["system"] = system.replace("{full_query}", query).replace("{initial_query}", agent_query)
            shopper_file.write_text(json.dumps(shopper, ensure_ascii=False, indent=2) + "\n")

        cmd = [
            args.pi_bin, "--mode", "json", "--offline",
            "--no-builtin-tools", "--no-extensions", "-e", str(ADAPTER_PATH),
        ]
        if args.plugin != "off":
            cmd += ["-e", str(args.plugin_file)]
        if args.state != "off":
            cmd += ["-e", str(STATE_PATH)]
        if args.roles:
            cmd += ["-e", str(ROLES_PATH)]
        cmd += [
            "--no-skills", "--no-prompt-templates", "--no-context-files",
            "--system-prompt", task_prompt,
            "--model", args.model, "--thinking", "off",
            "--session-dir", str(task_dir / "session"),
            agent_query,
        ]
        env = dict(
            base_env,
            SHOP_ENV_URL=args.env_url,
            SHOP_ENV_IDX=str(env_idx),
            SHOP_RUN_DIR=str(task_dir),
            SHOP_MAX_LLM_TURNS=str(args.max_llm_turns),
            SHOP_MAX_STEPS=str(args.max_steps),
            SHOP_INITIAL_STATE=json.dumps(reset.get("observation_state") or {}, ensure_ascii=False),
            SHOP_PLUGIN="" if args.plugin == "off" else args.plugin,
            SHOP_ASK="1" if args.ask else "0",
            SHOP_SHOPPER_FILE=str(shopper_file) if args.ask else "",
            SHOP_MAX_QUESTIONS=str(args.max_questions),
            SHOP_STATE="" if args.state == "off" else args.state,
            SHOP_STATE_WINDOW=str(args.state_window),
            SHOP_ROLES="1" if args.roles else "0",
            SHOP_BUYER_PROMPT_FILE=str(BUYER_PROMPT) if args.roles else "",
            SHOP_BUYER_MODEL=args.model,
            SHOP_PROFILE_TIERS="1" if args.profile_tiers else "0",
            SHOP_PERSONA_FILE=str(task_dir / "persona.json") if args.profile_tiers else "",
        )
        with open(task_dir / "events.jsonl", "w") as out, open(task_dir / "stderr.log", "w") as err:
            try:
                # 所有题共用同一个空目录作为 cwd：pi 会把 cwd 写进系统提示，这样每题的系统提示完全一致。
                proc = subprocess.run(
                    cmd, stdout=out, stderr=err, env=env, cwd=str(NEUTRAL_CWD), timeout=args.timeout,
                    stdin=subprocess.DEVNULL,
                )
                record["returncode"] = proc.returncode
            except subprocess.TimeoutExpired:
                record["error"] = "timeout"
    except Exception as exc:  # 记录下来继续跑其他题
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env_call(args.env_url, {"action": "release_one", "env_idx": env_idx})
        except Exception as exc:
            record["release_error"] = str(exc)

    record["seconds"] = round(time.time() - started, 1)
    record["terminal"] = (task_dir / "terminal.json").exists()
    done_file.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(ROOT / "data" / "dev_ids.json"))
    parser.add_argument("--task-ids", default="", help="逗号分隔，优先于 --tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="yowant/deepseek-v4-flash")
    parser.add_argument("--plugin", default="off", help="off、all，或逗号分隔的组件名（见插件文件头部说明）")
    parser.add_argument("--plugin-file", type=Path, default=DEFAULT_PLUGIN, help="插件文件；旧版组件在 pi-shop/plugin_v0.ts")
    parser.add_argument("--state", default="off", help="任务状态层（pi-shop/task_state.ts）：off、all，或逗号分隔的组件名；在插件之后加载")
    parser.add_argument("--state-window", type=int, default=6, help="任务状态层开启时只给模型最近几步原文；0 表示保留全部历史")
    parser.add_argument("--roles", action="store_true", help="第四版：前台导购 + 后台买手（pi-shop/roles.ts），只用于论文多轮设置")
    parser.add_argument("--multiturn", default="", help="多轮数据（data/multiturn/<split>.jsonl）：agent 拿到模糊需求")
    parser.add_argument("--paper-openings", default="", help="论文多轮模式：买家开场白（data/multiturn_paper/<split>_openings.jsonl），自动开启 ask")
    parser.add_argument("--persona", action="store_true", help="画像设置：agent 拿到用户画像，买家只知道简版需求；开场白须按 --persona 生成")
    parser.add_argument("--profile-tiers", action="store_true", help="画像分级：按来源把画像整理进第四版的需求单（需 --persona --roles）")
    parser.add_argument("--ask", action="store_true", help="提供 ask_shopper 工具（模拟买家回答）")
    parser.add_argument("--shopper-model", default="qwen3.7-plus")
    parser.add_argument("--shopper-prompt-file", type=Path, default=None, help="默认：论文模式用官方买家提示，否则 prompts/shopper.md")
    parser.add_argument("--shopper-max-tokens", type=int, default=None, help="默认：论文模式 512，否则 200")
    parser.add_argument("--max-questions", type=int, default=None, help="0 表示不单独限制提问次数；默认：论文模式 0，否则 3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--timeout", type=int, default=900, help="单题 pi 进程超时（秒）")
    parser.add_argument("--max-llm-turns", type=int, default=None, help="回合上限（提问也算一个回合）；默认：论文模式 40，否则 60")
    parser.add_argument("--max-steps", type=int, default=35, help="须与环境配置 termination.max_steps 一致，插件用它算剩余步数")
    parser.add_argument("--system-prompt-file", default=None, help="默认：分工模式用 prompts/roles_sales.md，否则 prompts/base.md")
    parser.add_argument("--pi-bin", default="pi")
    args = parser.parse_args()

    paper = bool(args.paper_openings)
    if paper and args.multiturn:
        sys.exit("--paper-openings 和 --multiturn 不能同时使用")
    if args.roles and (not paper or args.plugin != "off" or args.state != "off"):
        sys.exit("--roles 只用于论文多轮设置（--paper-openings），而且不和 --plugin、--state 一起用")
    if args.persona and not paper:
        sys.exit("--persona 只用于论文多轮设置（--paper-openings）")
    if args.profile_tiers and not (args.persona and args.roles):
        sys.exit("--profile-tiers 要和 --persona --roles 一起用")
    if args.system_prompt_file is None:
        args.system_prompt_file = str(SALES_PROMPT if args.roles else BASE_PROMPT)
    if paper:
        args.ask = True
    if args.max_questions is None:
        args.max_questions = 0 if paper else 3
    if args.max_llm_turns is None:
        args.max_llm_turns = 40 if paper else 60
    if args.shopper_prompt_file is None:
        args.shopper_prompt_file = PAPER_SHOPPER_PROMPT if paper else SHOPPER_PROMPT
    if args.shopper_max_tokens is None:
        args.shopper_max_tokens = 512 if paper else 200
    if args.plugin != "off" and not args.plugin_file.exists():
        sys.exit(f"找不到插件: {args.plugin_file}")
    secrets = load_dotenv(ROOT / ".env")
    if "MODEL_PROXY_API_KEY" not in secrets and "MODEL_PROXY_API_KEY" not in os.environ:
        sys.exit("缺少 MODEL_PROXY_API_KEY（.env 或环境变量）")

    base_env = dict(os.environ)
    base_env.update(secrets)
    base_env.update(
        PI_CODING_AGENT_DIR=str(ROOT / ".pi-clean"),
        PI_OFFLINE="1",
        PI_SKIP_VERSION_CHECK="1",
        PI_TELEMETRY="0",
    )
    system_prompt = Path(args.system_prompt_file).read_text().strip()
    multiturn = load_multiturn(args.multiturn)
    openings = load_multiturn(args.paper_openings)
    task_ids = load_task_ids(args, multiturn)

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "plugin": args.plugin,
        "plugin_file": str(args.plugin_file.relative_to(ROOT)) if args.plugin != "off" else None,
        "state": args.state,
        "state_window": args.state_window if args.state != "off" else None,
        "roles": args.roles,
        "buyer_prompt": BUYER_PROMPT.read_text().strip() if args.roles else None,
        "protocol": "paper" if paper else ("ours" if args.multiturn else "single"),
        "multiturn": args.multiturn or None,
        "paper_openings": args.paper_openings or None,
        "ask": args.ask,
        "shopper_model": args.shopper_model if args.ask else None,
        "shopper_prompt_file": str(Path(args.shopper_prompt_file).resolve().relative_to(ROOT)) if args.ask else None,
        "max_questions": args.max_questions if args.ask else None,
        "persona": args.persona,
        "persona_agent_block": PERSONA_AGENT_BLOCK.read_text(encoding="utf-8").strip() if args.persona else None,
        "persona_shopper_rule": PERSONA_SHOPPER_RULE.read_text(encoding="utf-8").strip() if args.persona else None,
        "profile_tiers": args.profile_tiers,
        "profile_prompt": PROFILE_PROMPT.read_text(encoding="utf-8").strip() if args.profile_tiers else None,
        "system_prompt": system_prompt,
        "max_llm_turns": args.max_llm_turns,
        "n_tasks": len(task_ids),
        "task_source": args.task_ids or args.multiturn or args.paper_openings or args.tasks,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    log(f"run={args.run_name} tasks={len(task_ids)} plugin={args.plugin} state={args.state} roles={args.roles} protocol={config['protocol']} ask={args.ask} "
        f"model={args.model} workers={args.workers}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t, args, run_dir, base_env, system_prompt, multiturn, openings): t for t in task_ids}
        for fut in as_completed(futures):
            task_id = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            results.append(rec)
            log(f"[{len(results)}/{len(task_ids)}] task {task_id}: terminal={rec.get('terminal')} "
                f"rc={rec.get('returncode')} err={rec.get('error')} {rec.get('seconds')}s")

    failed = [r for r in results if r.get("error") or r.get("returncode") not in (0, None)]
    log(f"完成 {len(results)} 题，进程/环境异常 {len(failed)} 题。日志目录: {run_dir}")


if __name__ == "__main__":
    main()
