/**
 * ShopSimulator 环境适配层：基线和插件实验都加载它，插件关掉时就是"原版 pi + 这层适配"。
 *
 * - 把商店动作注册成 pi 工具，转发到环境的 HTTP 接口（SHOP_ENV_URL / SHOP_ENV_IDX）。
 * - pi 默认并行执行同一条消息里的多个工具调用；商店是有状态的，所以每个 LLM 回合只放行一个商店动作。
 * - 模型只看到按白名单渲染的公开页面。环境返回的 gold、奖励、进度等字段只写进 SHOP_RUN_DIR 下的评测日志。
 * - 多轮模式（SHOP_ASK=1）额外提供 ask_shopper：由模拟买家回答。买家的系统提示由 runner 填好后写进 SHOP_SHOPPER_FILE。
 *   我们自己的多轮设置：买家只知道完整需求（不知道 gold），提问不消耗环境步数，每题最多 SHOP_MAX_QUESTIONS 次。
 *   论文多轮设置（SHOP_MAX_QUESTIONS=0）：买家文件里预置了开场对话，提问不限次数，但和商店动作一起计入回合上限
 *   （SHOP_MAX_LLM_TURNS）；照官方做法，每次返回都附上当前回合数。
 * - 分工模式（SHOP_ROLES=1，第四版 harness，见 pi-shop/roles.ts）：主 agent 是导购，不注册商店工具；买手和下单动作
 *   通过 "shop:api" 拿到的 act() 操作商店。回合只算对外的动作：每个商店动作（含被合法性校验拦下的）、每句对用户说的话；
 *   导购和买手内部的调用不算回合，另外统计调用次数和 token。
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const ENV_URL = (process.env.SHOP_ENV_URL ?? "http://127.0.0.1:5700").replace(/\/+$/, "");
const ENV_IDX = process.env.SHOP_ENV_IDX;
const RUN_DIR = process.env.SHOP_RUN_DIR;
const MAX_LLM_TURNS = Number(process.env.SHOP_MAX_LLM_TURNS ?? "60");
const CONTENT_CHAR_LIMIT = 6000;
const ASK_MODE = process.env.SHOP_ASK === "1";
const MAX_QUESTIONS = Number(process.env.SHOP_MAX_QUESTIONS ?? "3");
const PAPER_MODE = ASK_MODE && MAX_QUESTIONS <= 0;
const PROXY_URL = (process.env.MODEL_PROXY_BASE_URL ?? "").replace(/\/+$/, "");
const ROLES_MODE = process.env.SHOP_ROLES === "1";
// 分工模式下导购自己的模型调用不算回合，这里给它一个上限，防止一直在内部打转
const MAX_ROLE_LLM_CALLS = 100;

type Params = Record<string, string>;

export interface ShopTool {
	name: string;
	label: string;
	description: string;
	parameters: ReturnType<typeof Type.Object>;
	toAction: (params: Params) => string;
}

const NO_PARAMS = Type.Object({});

// 环境的商品页不渲染 Attributes 按钮（属性已直接显示在页面上），所以不提供 view_attributes。
export const SHOP_TOOLS: ShopTool[] = [
	{
		name: "search_products",
		label: "搜索商品",
		description: "在商店中按关键词搜索商品，返回搜索结果页。",
		parameters: Type.Object({ query: Type.String({ description: "搜索关键词" }) }),
		toAction: (p) => `search[${p.query}]`,
	},
	{
		name: "open_product",
		label: "打开商品",
		description: "打开当前搜索结果页中的一个商品，进入商品详情页。",
		parameters: Type.Object({ asin: Type.String({ description: "搜索结果中方括号内的商品ID" }) }),
		toAction: (p) => `click[${p.asin}]`,
	},
	{
		name: "select_option",
		label: "选择规格",
		description: "在商品详情页选择一个规格值（如颜色、尺码、套餐）。",
		parameters: Type.Object({ value: Type.String({ description: "“可选规格”中列出的某个值，需与页面上的文字一致" }) }),
		toAction: (p) => `click[${p.value}]`,
	},
	{
		name: "view_description",
		label: "查看描述",
		description: "查看当前商品的描述（Description）。",
		parameters: NO_PARAMS,
		toAction: () => "click[Description]",
	},
	{
		name: "view_features",
		label: "查看卖点",
		description: "查看当前商品的卖点（Features）。",
		parameters: NO_PARAMS,
		toAction: () => "click[Features]",
	},
	{
		name: "view_reviews",
		label: "查看评价",
		description: "查看当前商品的评价（Reviews）。",
		parameters: NO_PARAMS,
		toAction: () => "click[Reviews]",
	},
	{
		name: "next_page",
		label: "下一页",
		description: "搜索结果翻到下一页。",
		parameters: NO_PARAMS,
		toAction: () => "click[Next >]",
	},
	{
		name: "prev_page",
		label: "上一页",
		description: "返回上一页：搜索结果的上一页，或从商品详情/详情子页返回。",
		parameters: NO_PARAMS,
		toAction: () => "click[< Prev]",
	},
	{
		name: "back_to_search",
		label: "回到搜索",
		description: "回到搜索首页，以便重新搜索。",
		parameters: NO_PARAMS,
		toAction: () => "click[Back to Search]",
	},
	{
		name: "buy_now",
		label: "立即购买",
		description: "按当前已选规格购买当前商品。购买后任务结束，无法撤销。",
		parameters: NO_PARAMS,
		toAction: () => "click[Buy Now]",
	},
];

// 提问也算本回合的一个动作：先看到回答，再决定下一步。分工模式下导购的工具同样每个回合只放行一个。
const ROLE_TOOL_NAMES = ["find_products", "show_candidates", "buy"];
const SHOP_TOOL_NAMES = new Set([
	...(ROLES_MODE ? ROLE_TOOL_NAMES : SHOP_TOOLS.map((t) => t.name)),
	...(ASK_MODE ? ["ask_shopper"] : []),
]);

interface QA {
	question: string;
	answer: string;
}

interface ShopperConfig {
	system: string;
	model: string;
	max_tokens?: number;
	/** 预置在对话最前面的问答（论文设置里的开场：请提供您的购物需求 → 开场白），不算 agent 的提问。 */
	history?: QA[];
}

function loadShopper(): ShopperConfig | null {
	const file = process.env.SHOP_SHOPPER_FILE;
	if (!ASK_MODE || !file) return null;
	return JSON.parse(readFileSync(file, "utf-8")) as ShopperConfig;
}

async function shopperReply(cfg: ShopperConfig, history: QA[], question: string): Promise<{ answer: string; usage: unknown }> {
	const messages = [{ role: "system", content: cfg.system }];
	for (const h of [...(cfg.history ?? []), ...history]) messages.push({ role: "user", content: h.question }, { role: "assistant", content: h.answer });
	messages.push({ role: "user", content: question });
	let lastError: unknown;
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.MODEL_PROXY_API_KEY ?? ""}` },
				body: JSON.stringify({ model: cfg.model, messages, temperature: 0, max_tokens: cfg.max_tokens ?? 200, enable_thinking: false }),
				signal: AbortSignal.timeout(60_000),
			});
			if (!res.ok) throw new Error(`模拟买家 HTTP ${res.status}`);
			const body = (await res.json()) as any;
			const answer = String(body?.choices?.[0]?.message?.content ?? "").trim();
			if (!answer) throw new Error("模拟买家返回为空");
			return { answer, usage: body?.usage };
		} catch (err) {
			lastError = err;
			await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
		}
	}
	throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

function log(file: string, record: Record<string, unknown>): void {
	if (!RUN_DIR) return;
	mkdirSync(RUN_DIR, { recursive: true });
	appendFileSync(join(RUN_DIR, file), `${JSON.stringify({ t: new Date().toISOString(), ...record })}\n`);
}

async function envInteract(action: string): Promise<Record<string, any>> {
	if (!ENV_IDX) throw new Error("SHOP_ENV_IDX 未设置");
	const res = await fetch(`${ENV_URL}/api/shop_agent`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ action: "interact", env_idx: Number(ENV_IDX), response: action }),
	});
	if (!res.ok) throw new Error(`环境 HTTP ${res.status}`);
	const body = (await res.json()) as { result?: Record<string, any> };
	const result = body?.result;
	if (!result || typeof result !== "object") throw new Error("环境返回格式错误");
	if (result.error) throw new Error(`环境错误：${result.error}`);
	return result;
}

function fmtPrice(price: unknown): string {
	if (Array.isArray(price)) return price.map(String).join(" / ");
	if (price && typeof price === "object") return JSON.stringify(price);
	return price === undefined || price === null || price === "" ? "未知" : String(price);
}

function contentToText(content: unknown): string {
	let text: string;
	if (typeof content === "string") text = content;
	else if (Array.isArray(content)) text = content.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n");
	else if (content && typeof content === "object") text = JSON.stringify(content);
	else text = "";
	text = text.trim() || "（无内容）";
	return text.length > CONTENT_CHAR_LIMIT ? `${text.slice(0, CONTENT_CHAR_LIMIT)}\n（内容过长，已截断）` : text;
}

export function renderPage(state: any): string {
	if (!state || typeof state !== "object") return "（页面为空）";
	const out: string[] = [];
	const pageType = String(state.page_type ?? "unknown");

	if (pageType === "search_home") {
		out.push("【搜索首页】");
	} else if (pageType === "search_results") {
		out.push(`【搜索结果】搜索词：${state.query ?? ""}`);
		out.push(
			`第 ${state.page ?? 1}/${state.total_pages ?? 1} 页，共 ${state.total_results ?? 0} 个结果` +
				`（本页为第 ${state.rank_start ?? 0}–${state.rank_end ?? 0} 名）`,
		);
		for (const p of state.products ?? []) {
			out.push(`${p.rank ?? "-"}. [${p.asin}] ${p.title ?? ""}`);
			out.push(`   价格：${fmtPrice(p.price)}｜店铺：${p.brand ?? ""}｜类目：${p.category ?? ""}`);
			if (Array.isArray(p.key_attributes) && p.key_attributes.length) out.push(`   属性：${p.key_attributes.join("；")}`);
		}
	} else if (pageType === "product_detail" || pageType === "information_subpage") {
		const p = state.product ?? {};
		out.push(pageType === "product_detail" ? "【商品详情】" : `【商品详情 · ${state.subpage ?? "详情"}】`);
		out.push(`商品ID：${p.asin ?? ""}`);
		out.push(`标题：${p.title ?? ""}`);
		out.push(`店铺：${p.brand ?? ""}｜类目：${p.category ?? ""}`);
		out.push(`价格：${fmtPrice(p.price)}`);
		if (state.selected_price !== undefined && state.selected_price !== null) out.push(`当前规格价格：${state.selected_price}`);
		if (Array.isArray(p.key_attributes) && p.key_attributes.length) out.push(`属性：${p.key_attributes.join("；")}`);
		const axes = Object.entries((state.available_options ?? {}) as Record<string, string[]>);
		if (axes.length) {
			out.push("可选规格：");
			for (const [axis, values] of axes) out.push(`  ${axis}：${JSON.stringify(values)}`);
		} else {
			out.push("可选规格：无");
		}
		const selected = Object.entries((state.selected_options ?? {}) as Record<string, string>);
		out.push(`已选规格：${selected.length ? selected.map(([k, v]) => `${k}=${v}`).join("，") : "无"}`);
		if (pageType === "information_subpage") {
			out.push(`${state.subpage ?? "详情"}内容：`);
			out.push(contentToText(state.content));
		}
	} else if (pageType === "terminal") {
		out.push("【任务已结束】");
	} else {
		out.push(`【页面：${pageType}】`);
	}

	const actions: string[] = Array.isArray(state.actions) ? state.actions.map(String) : [];
	out.push(`可点击：${JSON.stringify(actions)}`);
	return out.join("\n");
}

export interface ActResult {
	kind: "ok" | "done" | "error";
	text: string;
	step: number;
	message?: string;
	/** 分工模式：回合用完或任务已经结束，这个动作没有执行 */
	refused?: boolean;
}

/** 分工模式下给 roles.ts 用的商店接口（通过 pi.events 的 "shop:api" 拿到）。 */
export interface ShopApi {
	maxTurns: number;
	turn: () => number;
	steps: () => number;
	isDone: () => boolean;
	withTurn: (text: string) => string;
	act: (name: string, params: Params, who: string) => Promise<ActResult>;
	countBlocked: () => number;
}

export default function (pi: ExtensionAPI) {
	let turn = 0;
	let llmCalls = 0;
	let shopCallsThisTurn = 0;
	let envSteps = 0;
	let done = false;
	// 官方 multi_eval 在每条返回后附上"轮次: N"（已做出的决策数）
	const withTurn = (text: string) => (PAPER_MODE ? `${text}\n轮次: ${turn}` : text);

	pi.on("turn_start", async () => {
		llmCalls += 1;
		// 分工模式下回合只算对外的动作，在商店动作和提问时累加
		if (!ROLES_MODE) turn += 1;
		shopCallsThisTurn = 0;
	});

	pi.on("tool_call", async (event) => {
		if (!SHOP_TOOL_NAMES.has(event.toolName)) return;
		const base = { turn, tool: event.toolName, args: event.input };
		if (done) {
			log("harness_events.jsonl", { type: "after_done", ...base });
			return { block: true, reason: "任务已经结束。", terminate: true };
		}
		if (ROLES_MODE ? turn >= MAX_LLM_TURNS || llmCalls > MAX_ROLE_LLM_CALLS : turn > MAX_LLM_TURNS) {
			log("harness_events.jsonl", { type: "turn_cap", ...base, ...(ROLES_MODE ? { llm_calls: llmCalls } : {}) });
			return { block: true, reason: "已达到回合上限，任务终止。", terminate: true };
		}
		shopCallsThisTurn += 1;
		if (shopCallsThisTurn > 1) {
			log("harness_events.jsonl", { type: "serial_block", ...base });
			// terminate: true 只在同批结果全部终止时生效：同批放行的那个动作若结束了任务，就不再多调一次模型。
			return {
				block: true,
				reason: "每个回合只能执行一个商店动作，这条调用没有执行。请先看上一个动作返回的页面，再决定下一步。",
				terminate: true,
			};
		}
		return undefined;
	});

	async function act(tool: ShopTool, params: Params, who?: string): Promise<ActResult> {
		const action = tool.toAction(params ?? {});
		const by = who ? { who } : {};
		let result: Record<string, any>;
		try {
			result = await envInteract(action);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			log("harness_events.jsonl", { type: "env_error", turn, tool: tool.name, args: params, action, message, ...by });
			done = true;
			return { kind: "error", text: "商店环境异常，任务中止。", step: envSteps, message };
		}
		envSteps += 1;
		log("env_steps.jsonl", { step: envSteps, turn, tool: tool.name, args: params, action, result, ...by });
		if (!result.done) pi.events.emit("shop:state", { step: envSteps, turn, state: result.observation_state });

		if (result.done) {
			done = true;
			pi.events.emit("shop:done", { step: envSteps, turn });
			if (RUN_DIR) {
				writeFileSync(
					join(RUN_DIR, "terminal.json"),
					`${JSON.stringify({ step: envSteps, turn, tool: tool.name, action, result }, null, 2)}\n`,
				);
			}
			return { kind: "done", text: "【任务已结束】", step: envSteps };
		}
		return { kind: "ok", text: withTurn(renderPage(result.observation_state)), step: envSteps };
	}

	if (ROLES_MODE) {
		const api: ShopApi = {
			maxTurns: MAX_LLM_TURNS,
			turn: () => turn,
			steps: () => envSteps,
			isDone: () => done,
			withTurn,
			async act(name: string, params: Params, who: string) {
				const tool = SHOP_TOOLS.find((t) => t.name === name);
				if (!tool) throw new Error(`未知的商店动作：${name}`);
				if (done) return { kind: "done", text: "任务已经结束。", step: envSteps, refused: true };
				if (turn >= MAX_LLM_TURNS) return { kind: "ok", text: "回合已经用完，不能再操作商店。", step: envSteps, refused: true };
				turn += 1;
				return act(tool, params, who);
			},
			// 被合法性校验拦下的商店动作也算一个回合：和 v2 一样，模型做了这个决定，只是没交给环境
			countBlocked() {
				turn += 1;
				return turn;
			},
		};
		pi.events.on("shop:api", (register: unknown) => {
			if (typeof register === "function") register(api);
		});
	} else {
		for (const tool of SHOP_TOOLS) {
			pi.registerTool({
				name: tool.name,
				label: tool.label,
				description: tool.description,
				parameters: tool.parameters,
				async execute(_toolCallId: string, params: Params) {
					const r = await act(tool, params);
					if (r.kind === "error") {
						return { content: [{ type: "text" as const, text: r.text }], details: { envError: r.message }, terminate: true };
					}
					if (r.kind === "done") {
						return { content: [{ type: "text" as const, text: r.text }], details: { step: r.step, done: true }, terminate: true };
					}
					return { content: [{ type: "text" as const, text: r.text }], details: { step: r.step } };
				},
			});
		}
	}

	const shopper = loadShopper();
	if (!shopper) return;
	const sh: ShopperConfig = shopper;
	const history: QA[] = [];

	async function askShopper(question: string, via: "tool" | "text"): Promise<string> {
		if (ROLES_MODE) turn += 1;
		let answer = "（用户暂时没有回应）";
		let usage: unknown = null;
		let error: string | undefined;
		try {
			({ answer, usage } = await shopperReply(sh, history, question));
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		}
		history.push({ question, answer });
		const remaining = PAPER_MODE ? null : MAX_QUESTIONS - history.length;
		log("ask_log.jsonl", { type: "ask", via, n: history.length, turn, step: envSteps, question, answer, model: sh.model, usage, error });
		pi.events.emit("shop:ask", { question, answer, remaining });
		return answer;
	}

	pi.registerTool({
		name: "ask_shopper",
		label: "询问用户",
		description: PAPER_MODE
			? "向用户提问，或请用户确认商品和规格。提问也计入回合数。"
			: `向用户提问，澄清需求里没说清楚的地方（如颜色、尺码、容量、数量、套餐）。不消耗商店步数，每个任务最多问 ${MAX_QUESTIONS} 次。`,
		parameters: Type.Object({ question: Type.String({ description: "要问用户的问题" }) }),
		async execute(_toolCallId: string, params: Params) {
			const question = String(params?.question ?? "").trim();
			if (!question) return { content: [{ type: "text" as const, text: "问题不能为空。" }], details: {} };
			if (!PAPER_MODE && history.length >= MAX_QUESTIONS) {
				log("ask_log.jsonl", { type: "refused", turn, step: envSteps, question });
				return {
					content: [{ type: "text" as const, text: `提问次数已用完（最多 ${MAX_QUESTIONS} 次），请根据已有信息继续。` }],
					details: { refused: true },
				};
			}
			const answer = await askShopper(question, "tool");
			const remaining = MAX_QUESTIONS - history.length;
			return {
				content: [{ type: "text" as const, text: PAPER_MODE ? withTurn(`用户回答：${answer}`) : `用户回答：${answer}\n（还能提问 ${remaining} 次）` }],
				details: { ask: history.length },
			};
		},
	});

	if (!PAPER_MODE) return;
	// 官方协议里 agent 每一步不是问买家就是操作商店，只有购买或回合用尽才结束。pi 里模型直接回复文字会结束这一轮，
	// 所以把这段文字当作对用户说的话转给买家，再把回答作为下一条消息发回去。
	pi.on("agent_end", async (event) => {
		if (done || turn >= MAX_LLM_TURNS) return;
		const last = [...(event.messages ?? [])].reverse().find((m: any) => m?.role === "assistant") as any;
		if (!last || last.stopReason !== "stop") return;
		const text = contentToPlain(last.content);
		if (!text) return;
		const answer = await askShopper(text, "text");
		// 插件在这个事件里同步追加要附在回答后面的提示（如收尾提醒）
		const notes: string[] = [];
		pi.events.emit("shop:relayed", { turn, notes });
		const tail = notes.length ? `\n——\n${notes.join("\n")}` : "";
		pi.sendUserMessage(withTurn(`用户回答：${answer}`) + tail, { deliverAs: "followUp" });
	});
}

function contentToPlain(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((c: any) => c?.type === "text")
		.map((c: any) => String(c.text ?? ""))
		.join("\n")
		.trim();
}
