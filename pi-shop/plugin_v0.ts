/**
 * ShopSimulator 场景插件：加载时增强 harness，不加载时就是原版 pi + 适配层。
 * 只用模型也能看到的公开信息（页面状态、用户需求），不读 gold、奖励或环境内部进度。
 *
 * 组件由 SHOP_PLUGIN 开关（"all" 或逗号分隔），便于逐个消融：
 *   guard    合法性守卫：动作在当前页面无效时直接拦下并说明原因，不消耗环境步数；
 *            文本只差空白或全/半角时改写成页面原文
 *   ledger   状态台账：每次返回页面时附上步数预算、已搜过的词、已看过的商品
 *   monitor  进展监控：连续多步没有获得新信息时提醒换策略
 *   audit    购买前审计：规格没选全或超过明确预算时拦一次购买
 *   finish   收尾保护：步数快用完时提醒尽快购买；没买就停下时追问一次
 *   contract 需求契约：系统提示里要求先把需求拆成清单，购买前逐条核对
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const COMPONENTS = ["guard", "ledger", "monitor", "audit", "finish", "contract"];
const requested = (process.env.SHOP_PLUGIN || "all").split(",").map((s) => s.trim()).filter(Boolean);
const enabled = new Set(requested.includes("all") ? COMPONENTS : requested);
const on = (component: string) => enabled.has(component);

const RUN_DIR = process.env.SHOP_RUN_DIR;
const MAX_STEPS = Number(process.env.SHOP_MAX_STEPS ?? "35");
const MAX_LLM_TURNS = Number(process.env.SHOP_MAX_LLM_TURNS ?? "60");
const STALL_LIMIT = 3;
const FINISH_THRESHOLD = 6;
const MAX_FINISH_NUDGES = 2;

const SHOP_TOOLS = new Set([
	"search_products",
	"open_product",
	"select_option",
	"view_description",
	"view_features",
	"view_reviews",
	"next_page",
	"prev_page",
	"back_to_search",
	"buy_now",
]);

const CONTRACT_PROMPT = `

购物方法：
1. 先把需求拆成清单：商品品类、必须满足的属性或功能、需要选的规格（颜色、尺码、容量、数量等）、预算。
2. 搜索用 2–4 个核心词（品类词加最关键的属性），不要把整句需求当搜索词；没找到就换词或翻页。
3. 打开候选商品后对照清单核对。需求里提到的规格要在“可选规格”中逐项选择，填写页面上的原文。
4. 购买前确认：关键属性满足，每个规格项都已选好，价格符合预算。`;

function log(record: Record<string, unknown>): void {
	if (!RUN_DIR) return;
	mkdirSync(RUN_DIR, { recursive: true });
	appendFileSync(join(RUN_DIR, "harness_events.jsonl"), `${JSON.stringify({ t: new Date().toISOString(), ...record })}\n`);
}

// 环境把动作参数转成小写后与页面可点击项做精确匹配。
const envKey = (s: unknown) => String(s ?? "").toLowerCase();
const looseKey = (s: unknown) => String(s ?? "").normalize("NFKC").replace(/\s+/g, "").toLowerCase();

function parseInitialState(): any {
	try {
		const state = JSON.parse(process.env.SHOP_INITIAL_STATE || "{}");
		if (state && typeof state === "object" && state.page_type) return state;
	} catch {
		// 用默认的搜索首页
	}
	return { page_type: "search_home", actions: ["search"] };
}

function pageLabel(state: any): string {
	switch (state?.page_type) {
		case "search_home":
			return "搜索首页";
		case "search_results":
			return `搜索结果第 ${state.page ?? 1} 页`;
		case "product_detail":
			return "商品详情页";
		case "information_subpage":
			return `商品的 ${state.subpage ?? "详情"} 子页`;
		default:
			return "当前页面";
	}
}

/** 需求里明确写出的预算上限（“以内/以下/不超过”等）；“左右”不算上限。 */
function explicitBudget(query: string): number | null {
	const patterns = [
		/(\d+(?:\.\d+)?)\s*(?:元|块|rmb)?\s*(?:以内|以下|之内|内)/i,
		/(?:不超过|不要超过|不高于|最多|上限)\s*(\d+(?:\.\d+)?)/,
	];
	for (const re of patterns) {
		const m = query.match(re);
		if (m) return Number(m[1]);
	}
	return null;
}

const short = (s: unknown, n = 18) => {
	const text = String(s ?? "");
	return text.length > n ? `${text.slice(0, n)}…` : text;
};

type Verdict = { block?: string; rewrite?: { field: string; value: string } };

export default function (pi: ExtensionAPI) {
	let cur: any = parseInitialState();
	let query = "";
	let stepsUsed = 0;
	let turns = 0;
	let done = false;
	let stall = 0;
	let finishNudges = 0;
	let lastSelection = "";
	const searches: { q: string; total: number; maxPage: number }[] = [];
	const seenResultPages = new Set<string>();
	const opened = new Map<string, { title: string; price: string; views: Set<string> }>();
	const auditBlocked = new Set<string>();

	pi.on("before_agent_start", async (event) => {
		if (!query) query = String(event.prompt ?? "");
		if (on("contract") && !event.systemPrompt.includes(CONTRACT_PROMPT.trim())) {
			return { systemPrompt: `${event.systemPrompt}${CONTRACT_PROMPT}` };
		}
		return undefined;
	});

	pi.on("turn_start", async () => {
		turns += 1;
	});

	pi.events.on("shop:done", () => {
		done = true;
	});

	pi.events.on("shop:state", (data: any) => {
		cur = data?.state ?? {};
		stepsUsed = Number(data?.step ?? stepsUsed);
		let progressed = false;
		if (cur.page_type === "search_results") {
			const q = String(cur.query ?? "");
			const key = `${looseKey(q)}#${cur.page ?? 1}`;
			if (!seenResultPages.has(key)) {
				seenResultPages.add(key);
				progressed = true;
			}
			const entry = searches.find((s) => looseKey(s.q) === looseKey(q));
			if (entry) entry.maxPage = Math.max(entry.maxPage, Number(cur.page ?? 1));
			else searches.push({ q, total: Number(cur.total_results ?? 0), maxPage: Number(cur.page ?? 1) });
		} else if (cur.page_type === "product_detail" || cur.page_type === "information_subpage") {
			const p = cur.product ?? {};
			const asin = String(p.asin ?? "");
			let entry = opened.get(asin);
			if (!entry) {
				entry = { title: String(p.title ?? ""), price: String(p.price ?? ""), views: new Set() };
				opened.set(asin, entry);
				progressed = true;
			}
			if (cur.page_type === "information_subpage" && !entry.views.has(String(cur.subpage))) {
				entry.views.add(String(cur.subpage));
				progressed = true;
			}
			const selection = `${asin}:${JSON.stringify(cur.selected_options ?? {})}`;
			if (selection !== lastSelection && Object.keys(cur.selected_options ?? {}).length) progressed = true;
			lastSelection = selection;
		}
		stall = progressed ? 0 : stall + 1;
	});

	function guard(tool: string, input: Record<string, string>): Verdict {
		const page = cur.page_type ?? "search_home";
		const clickables: string[] = Array.isArray(cur.actions) ? cur.actions.map(String) : [];
		const has = (text: string) => clickables.some((c) => envKey(c) === envKey(text));
		const repair = (text: string, pool: string[]) => {
			const matches = pool.filter((c) => looseKey(c) === looseKey(text));
			return matches.length === 1 ? matches[0] : null;
		};
		const where = pageLabel(cur);
		const backToProduct = "请先 prev_page 回到商品详情页。";

		switch (tool) {
			case "search_products":
				return String(input.query ?? "").trim() ? {} : { block: "搜索词不能为空。" };
			case "open_product": {
				if (page === "product_detail")
					return { block: `当前在${where}，只能在搜索结果页打开商品。请先 prev_page 回到搜索结果，或直接 search_products 重新搜索。` };
				if (page === "information_subpage") return { block: `当前在${where}，只能在搜索结果页打开商品。${backToProduct}` };
				if (page !== "search_results") return { block: `当前在${where}，请先搜索。` };
				if (has(input.asin)) return {};
				const asins = (cur.products ?? []).map((p: any) => String(p.asin));
				const fixed = repair(input.asin, asins);
				if (fixed) return { rewrite: { field: "asin", value: fixed } };
				return {
					block: `商品 ${input.asin} 不在当前这页搜索结果里（${where}）。本页可打开的商品ID：${asins.join("、")}。之前搜索结果里的商品需要重新搜到才能打开。`,
				};
			}
			case "select_option": {
				if (page === "information_subpage") return { block: `当前在${where}，不能选规格。${backToProduct}` };
				if (page !== "product_detail") return { block: `当前在${where}，只能在商品详情页选择规格。` };
				const values: string[] = Object.values(cur.available_options ?? {}).flat().map(String);
				if (!values.length) return { block: "这个商品没有可选规格，可以直接购买。" };
				if (has(input.value) && values.some((v) => envKey(v) === envKey(input.value))) return {};
				const fixed = repair(input.value, values);
				if (fixed && has(fixed)) return { rewrite: { field: "value", value: fixed } };
				return {
					block: `“${input.value}”不是这个商品的可选规格值。可选规格：${JSON.stringify(cur.available_options)}。请原样填写其中一个值。`,
				};
			}
			case "view_description":
			case "view_features":
			case "view_reviews": {
				const button = { view_description: "description", view_features: "features", view_reviews: "reviews" }[tool]!;
				if (page === "information_subpage") return { block: `当前已在${where}。要看其他详情，${backToProduct}` };
				if (page !== "product_detail") return { block: `当前在${where}，只能在商品详情页查看商品详情。` };
				return has(button) ? {} : { block: "这个商品没有该详情页。" };
			}
			case "next_page":
				if (page !== "search_results") return { block: `当前在${where}，只有搜索结果页可以翻页。` };
				return has("next >") ? {} : { block: "已经是最后一页搜索结果。" };
			case "prev_page":
				if (has("< prev")) return {};
				return {
					block:
						page === "search_results"
							? "已经是搜索结果第一页。要换关键词请直接调用 search_products。"
							: `当前在${where}，没有上一页。`,
				};
			case "back_to_search":
				return has("back to search") ? {} : { block: "当前已在搜索首页，可以直接调用 search_products。" };
			case "buy_now":
				if (page === "information_subpage") return { block: `当前在${where}，不能购买。${backToProduct}` };
				if (page !== "product_detail" || !has("buy now")) return { block: `当前在${where}，只能在商品详情页购买。` };
				return {};
			default:
				return {};
		}
	}

	function audit(): string | null {
		const asin = String(cur.product?.asin ?? "");
		const axes = Object.entries((cur.available_options ?? {}) as Record<string, string[]>).filter(([, v]) => v.length > 0);
		const selected = (cur.selected_options ?? {}) as Record<string, string>;
		const missing = axes.filter(([axis]) => !(axis in selected)).map(([axis]) => axis);
		if (missing.length && !auditBlocked.has(`${asin}|options`)) {
			auditBlocked.add(`${asin}|options`);
			return `还有规格没选：${missing.join("、")}。请为每一项选好最符合需求的值再购买；如果确定不需要选，再次调用 buy_now 即可。`;
		}
		const budget = explicitBudget(query);
		const price = Number(cur.selected_price);
		if (budget !== null && Number.isFinite(price) && price > budget && !auditBlocked.has(`${asin}|budget`)) {
			auditBlocked.add(`${asin}|budget`);
			return `当前规格价格 ${price} 元，超过需求里的预算上限 ${budget} 元。请换规格或换商品；如果确定要买，再次调用 buy_now。`;
		}
		return null;
	}

	pi.on("tool_call", async (event) => {
		if (!SHOP_TOOLS.has(event.toolName)) return undefined;
		const input = event.input as Record<string, string>;
		if (on("guard")) {
			const verdict = guard(event.toolName, input);
			if (verdict.block) {
				log({ type: "guard_block", turn: turns, tool: event.toolName, args: input, page: cur.page_type, reason: verdict.block });
				return { block: true, reason: verdict.block };
			}
			if (verdict.rewrite) {
				log({ type: "guard_rewrite", turn: turns, tool: event.toolName, from: input[verdict.rewrite.field], to: verdict.rewrite.value });
				input[verdict.rewrite.field] = verdict.rewrite.value;
			}
		}
		if (on("audit") && event.toolName === "buy_now") {
			const reason = audit();
			if (reason) {
				log({ type: "audit_block", turn: turns, asin: cur.product?.asin, selected: cur.selected_options, price: cur.selected_price, reason });
				return { block: true, reason };
			}
		}
		return undefined;
	});

	pi.on("tool_result", async (event) => {
		if (!SHOP_TOOLS.has(event.toolName) || event.isError || done || (event.details as any)?.done) return undefined;
		const notes: string[] = [];
		const remaining = MAX_STEPS - stepsUsed;
		if (on("ledger")) {
			notes.push(`【进度】已用 ${stepsUsed}/${MAX_STEPS} 步，剩余 ${remaining} 步`);
			if (searches.length) {
				const recent = searches.slice(-8).map((s) => `“${s.q}”(看到第${s.maxPage}页)`);
				notes.push(`【已搜索】${recent.join("；")}`);
			}
			if (opened.size) {
				const recent = [...opened.entries()].slice(-10).map(([asin, p]) => `${asin} ${short(p.title)} ${p.price}`);
				notes.push(`【已看过的商品】${recent.join("；")}`);
			}
		}
		if (on("monitor") && stall >= STALL_LIMIT) {
			notes.push(
				`【提醒】最近 ${stall} 步没有获得新信息。可以换一组更短的核心关键词重新搜索、翻到下一页，或在已看过的商品中选最符合的一个购买。`,
			);
			log({ type: "monitor_hint", turn: turns, step: stepsUsed, stall });
		}
		if (on("finish") && remaining <= FINISH_THRESHOLD) {
			notes.push(`【提醒】只剩 ${remaining} 步。任务要以购买结束：请从已看过的商品中选最符合需求的一个，打开它、选好全部规格后 buy_now。`);
			log({ type: "finish_hint", turn: turns, step: stepsUsed, remaining });
		}
		if (!notes.length) return undefined;
		return { content: [...event.content, { type: "text" as const, text: `\n——\n${notes.join("\n")}` }] };
	});

	pi.on("agent_end", async () => {
		if (!on("finish") || done || stepsUsed >= MAX_STEPS || turns >= MAX_LLM_TURNS || finishNudges >= MAX_FINISH_NUDGES) return;
		finishNudges += 1;
		log({ type: "finish_nudge", turn: turns, step: stepsUsed, n: finishNudges });
		pi.sendUserMessage(
			"任务还没有完成：需要购买一件商品才算完成。请继续，在已看过的商品中选最符合需求的一个（或继续搜索），选好规格后调用 buy_now。",
			{ deliverAs: "followUp" },
		);
	});
}
