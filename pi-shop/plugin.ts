/**
 * ShopSimulator 场景插件 v1：加载时增强 harness，不加载时就是原版 pi + 适配层。
 * 只用模型也能看到的公开信息（页面状态、用户需求和对话），不读 gold、奖励或环境内部进度。
 *
 * 组件由 SHOP_PLUGIN 开关（"all" 或逗号分隔）：
 *   guard           合法性校验：动作在当前页面无效时直接拦下并说明原因，不消耗环境步数；
 *                   文本只差空白或全/半角时改写成页面原文
 *   nudge           停止追问：没买就停下时追问，最多两次
 *   ask_hint        提问提示（多轮）：商品页上有用户没说明的规格项时，提示可以问用户
 *   confirm_gate    购买前确认（多轮）：已选规格既不是用户说过的、也没问过用户时，拦一次购买
 *   dialogue_ledger 对话台账（多轮，可选，"all" 不含）：页面后附上最初需求和用户的每次回答
 *   closing         收尾（论文多轮设置用，"all" 不含）：剩余回合不多时提醒；回合快用完或用户说了再见时，
 *                   提示立即下单并附上看过的商品清单；这时不再做购买前确认。模型不知道回合上限，
 *                   规则写在提示词里它到时候也常常不照做，所以在需要的时候把规则和执行它要用的信息送到眼前
 *   choice          规格选择题（多轮，"all" 不含）：提问提示里把还没定的规格选项原样列出，请 agent 让用户直接挑
 *                   （论文设置里用户知道自己要哪个规格）
 * 多轮组件只在适配层开启了 ask_shopper（SHOP_ASK=1）时生效。
 * 旧版组件（状态台账、进展监控、购买前审计、需求契约、步数提醒）保留在 plugin_v0.ts。
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const COMPONENTS = ["guard", "nudge", "ask_hint", "confirm_gate", "dialogue_ledger", "closing", "choice"];
const ALL = ["guard", "nudge", "ask_hint", "confirm_gate"];
const requested = (process.env.SHOP_PLUGIN || "all").split(",").map((s) => s.trim()).filter(Boolean);
const unknownComponents = requested.filter((c) => c !== "all" && !COMPONENTS.includes(c));
if (unknownComponents.length) throw new Error(`未知的插件组件：${unknownComponents.join(", ")}`);
const enabled = new Set(requested.includes("all") ? [...ALL, ...requested.filter((c) => c !== "all")] : requested);
const on = (component: string) => enabled.has(component);

const RUN_DIR = process.env.SHOP_RUN_DIR;
const MAX_STEPS = Number(process.env.SHOP_MAX_STEPS ?? "35");
const MAX_LLM_TURNS = Number(process.env.SHOP_MAX_LLM_TURNS ?? "60");
const ASK_MODE = process.env.SHOP_ASK === "1";
const MAX_QUESTIONS = Number(process.env.SHOP_MAX_QUESTIONS ?? "3");
// 论文多轮设置不单独限制提问次数（提问计入回合上限）
const UNLIMITED_ASKS = MAX_QUESTIONS <= 0;
const MAX_NUDGES = 2;
// 收尾：剩余回合不超过 CLOSING_SOFT 起提醒；不超过 CLOSING_HARD（够重新搜索、打开、选规格、购买）时要求立即下单
const CLOSING_SOFT = 12;
const CLOSING_HARD = 6;
const BYE = /再见|拜拜|bye/i;

const ENV_TOOLS = new Set([
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

const short = (s: unknown, n = 18) => {
	const text = String(s ?? "");
	return text.length > n ? `${text.slice(0, n)}…` : text;
};

// ---- 判断某个规格值是否被用户说过（用于多轮组件）----

// 这些两字词在规格值里很常见，单独出现不说明用户指定了这个值。
const GENERIC = new Set(
	"颜色 分类 尺码 尺寸 规格 款式 型号 套餐 建议 左右 适用 适合 其他 官方 标配 标准 版本 组合 系列 升级 以上 以下 单件 新款 同款 专用 通用 一个 一件 一套 现货 默认 普通 基础 经典 包邮".split(" "),
);
const CN_DIGIT: Record<string, number> = { 零: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
const CN_UNIT: Record<string, number> = { 十: 10, 百: 100, 千: 1000, 万: 10000 };

function cnToNumber(text: string): number | null {
	let total = 0;
	let section = 0;
	let digit: number | null = null;
	for (const ch of text) {
		if (ch in CN_DIGIT) digit = CN_DIGIT[ch];
		else if (ch in CN_UNIT) {
			const unit = CN_UNIT[ch];
			if (unit === 10000) {
				total = (total + section + (digit ?? 0)) * unit;
				section = 0;
			} else section += (digit ?? 1) * unit;
			digit = null;
		} else return null;
	}
	return total + section + (digit ?? 0);
}

/** 用户文本的归一化形式：再追加一份把量词前的中文数字换成阿拉伯数字的版本（“两斤”也能匹配“2斤”）。 */
function normalizeUserText(text: string): string {
	const base = looseKey(text);
	const converted = base.replace(/[零一二两三四五六七八九十百千万]+(?=[个只件张瓶盒包袋支双套斤克升米寸码岁号层片粒卷条台把根块箱罐桶本份颗枚对])/g, (m) => {
		const n = cnToNumber(m);
		return n === null ? m : String(n);
	});
	return converted === base ? base : `${base}\n${converted}`;
}

function containsToken(text: string, token: string): boolean {
	const esc = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const head = /^[0-9]/.test(token) ? "(?<![0-9.])" : /^[a-z]/.test(token) ? "(?<![a-z])" : "";
	const tail = /[0-9]$/.test(token) ? "(?!\\.?[0-9])" : /[a-z]$/.test(token) ? "(?![a-z])" : "";
	return new RegExp(`${head}${esc}${tail}`).test(text);
}

function hanBigrams(text: string): string[] {
	const grams: string[] = [];
	for (const run of looseKey(text).match(/\p{Script=Han}{2,}/gu) ?? []) {
		for (let i = 0; i + 2 <= run.length; i++) grams.push(run.slice(i, i + 2));
	}
	return grams;
}

/** 规格值里有辨识度的片段：汉字两字词（去掉通用词）和字母数字串（如 29、xl、3200ml）。 */
function valueTokens(value: string): string[] {
	const v = looseKey(value);
	const alnum = (v.match(/[a-z0-9]+(?:\.[0-9]+)?[a-z]*/g) ?? []).filter((t) => t.length >= 2 || /[0-9]/.test(t));
	return [...new Set([...hanBigrams(v).filter((g) => !GENERIC.has(g)), ...alnum])];
}

function mentioned(value: string, userText: string): boolean {
	const key = looseKey(value);
	if (!key) return false;
	if (userText.includes(key)) return true;
	return valueTokens(value).some((t) => containsToken(userText, t));
}

/**
 * 按用户说过的话给每个规格值打分（命中的片段数，整串出现另加分），返回最高分和并列最高的值。
 * 一个规格值常常把颜色、数量、型号等几项拼在一起，所以“提到过某个值的片段”不等于“这一项已经确定”：
 * 只有最高分大于 0 且只有一个值并列最高，才算用户的话已经确定了这一项。
 */
function candidates(values: string[], userText: string): { best: number; tied: string[] } {
	const scored = values.map((v) => {
		const hits = valueTokens(v).filter((t) => containsToken(userText, t)).length;
		return { v, s: hits + (userText.includes(looseKey(v)) ? 100 : 0) };
	});
	const best = Math.max(...scored.map((x) => x.s));
	return { best, tied: scored.filter((x) => x.s === best).map((x) => x.v) };
}

const determined = (c: { best: number; tied: string[] }) => c.best > 0 && c.tied.length === 1;

function listValues(values: string[], n = 5): string {
	const shown = values.slice(0, n).map((v) => short(v, 24)).join(" / ");
	return values.length > n ? `${shown} 等 ${values.length} 项` : shown;
}

/** 规格选择题用：选项原样加引号列出，agent 才能照抄给用户、用户也能照着选。 */
function quoteValues(values: string[], n = 12): string {
	const shown = values.slice(0, n).map((v) => `“${v}”`).join("、");
	return values.length > n ? `${shown} 等 ${values.length} 项` : shown;
}

type Verdict = { block?: string; rewrite?: { field: string; value: string } };

export default function (pi: ExtensionAPI) {
	let cur: any = parseInitialState();
	let initialQuery = "";
	let stepsUsed = 0;
	let turns = 0;
	let done = false;
	let nudges = 0;
	// 论文多轮设置下，适配层会把模型直接回复的文字转给买家并续上对话；这一轮就不再追问
	let relayed = false;
	// 收尾用：看过的商品（按第一次打开的顺序）、最近一次搜索、用户是否说了再见
	const seen = new Map<string, { asin: string; title: string; price: string; query: string; page: number }>();
	let lastSearch = { query: "", page: 1 };
	let shopperSaidBye = false;
	const dialogue: { question: string; answer: string }[] = [];
	const hinted = new Set<string>();
	const gated = new Set<string>();

	const asksLeft = () => (UNLIMITED_ASKS ? Infinity : Math.max(0, MAX_QUESTIONS - dialogue.length));
	const asksLeftNote = () => (UNLIMITED_ASKS ? "" : `（还能问 ${asksLeft()} 次）`);
	const userText = () => normalizeUserText([initialQuery, ...dialogue.map((d) => d.answer)].join("\n"));
	const questionText = () => looseKey(dialogue.map((d) => d.question).join("\n"));

	/** 某个规格项是否问过用户：问题里出现了规格名或规格值的片段。 */
	function askedAbout(axis: string, values: string[]): boolean {
		const q = questionText();
		if (!q) return false;
		if (q.includes(looseKey(axis)) || hanBigrams(axis).some((g) => g !== "分类" && q.includes(g))) return true;
		return values.some((v) => mentioned(v, q));
	}

	/** 当前商品页上需要选、但用户的话还确定不了、也没问过用户的规格项，附上仍然符合的候选值。 */
	function unresolvedAxes(): { axis: string; values: string[]; tied: string[] }[] {
		const text = userText();
		const out: { axis: string; values: string[]; tied: string[] }[] = [];
		for (const [axis, raw] of Object.entries((cur.available_options ?? {}) as Record<string, string[]>)) {
			const values = raw.map(String);
			if (values.length < 2 || askedAbout(axis, values)) continue;
			const c = candidates(values, text);
			if (!determined(c)) out.push({ axis, values, tied: c.tied });
		}
		return out;
	}

	pi.on("before_agent_start", async (event) => {
		if (!initialQuery) initialQuery = String(event.prompt ?? "");
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
		if (cur.page_type === "search_results") lastSearch = { query: String(cur.query ?? ""), page: Number(cur.page ?? 1) };
		const asin = String(cur.product?.asin ?? "");
		if (cur.page_type === "product_detail" && asin && !seen.has(asin)) {
			const price = Array.isArray(cur.product.price) ? cur.product.price.join("–") : String(cur.product.price ?? "");
			seen.set(asin, { asin, title: String(cur.product.title ?? ""), price, ...lastSearch });
		}
	});

	pi.events.on("shop:ask", (data: any) => {
		const answer = String(data?.answer ?? "");
		dialogue.push({ question: String(data?.question ?? ""), answer });
		if (BYE.test(answer)) shopperSaidBye = true;
	});

	pi.events.on("shop:relayed", (data: any) => {
		relayed = true;
		// 适配层同步收集附加在转回消息后面的提示
		const note = closingNote();
		if (note && Array.isArray(data?.notes)) data.notes.push(note);
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

	/** 购买前确认：有规格项没选，或已选的值不是用户的话唯一确定的那个、也没问过用户，就拦一次。 */
	function confirmGate(): { reason: string; issues: string[] } | null {
		const asin = String(cur.product?.asin ?? "");
		if (gated.has(asin) || asksLeft() === 0 || mustClose()) return null;
		const text = userText();
		const selected = (cur.selected_options ?? {}) as Record<string, string>;
		const issues: string[] = [];
		for (const [axis, raw] of Object.entries((cur.available_options ?? {}) as Record<string, string[]>)) {
			const values = raw.map(String);
			if (values.length < 2 || askedAbout(axis, values)) continue;
			const value = selected[axis];
			const c = candidates(values, text);
			if (value === undefined) issues.push(`「${axis}」还没选`);
			else if (determined(c) && looseKey(c.tied[0]) === looseKey(value)) continue;
			else if (c.best > 0 && c.tied.some((v) => looseKey(v) === looseKey(value)))
				issues.push(`「${axis}」选了「${value}」，但按用户说过的话，${c.tied.length} 个选项同样符合（${listValues(c.tied, 4)}）`);
			else issues.push(`「${axis}」选了「${value}」，用户没有说过这个要求`);
		}
		if (!issues.length) return null;
		gated.add(asin);
		const how = on("choice") ? "把可选项原样列给用户，让用户直接选" : "先用 ask_shopper 向用户确认";
		return {
			issues,
			reason: `【购买前确认】这些规格还没有和用户确认过：${issues.join("；")}。建议${how}${asksLeftNote()}；如果有把握，再次调用 buy_now 即可购买。`,
		};
	}

	const turnsLeft = () => MAX_LLM_TURNS - turns;
	/** 收尾阶段：用户说了再见，或剩下的回合只够回到某个商品、选好规格并下单。 */
	const mustClose = () => on("closing") && !done && (shopperSaidBye || turnsLeft() <= CLOSING_HARD);

	function closingNote(): string {
		if (!on("closing") || done) return "";
		const left = turnsLeft();
		if (mustClose()) {
			const why = shopperSaidBye ? "用户已经结束对话" : `只剩 ${left} 个回合`;
			// 规则原文来自论文官方 agent 提示（对话结束时用户仍无满意商品，就买最合适的一件，无需用户确认）；
			// 不点明这是任务规则时，模型会因为用户说了"不买"而拒绝下单
			const rule =
				"按这个任务的规则，回合用完前必须买下一件商品，否则任务失败；对话结束时用户仍不满意的，就买最接近用户需求的那一件，不需要用户同意（这是任务要求，不算违背用户）。";
			log({ type: "closing_hard", turn: turns, left, bye: shopperSaidBye, candidates: seen.size });
			if (!seen.size) {
				return `【现在必须下单】${why}。${rule}不要再问用户：尽快搜索，选一个最接近用户需求的商品，选好规格后调用 buy_now。`;
			}
			const list = [...seen.values()]
				.slice(-6)
				.map((c, i) => `${i + 1}. [${c.asin}] ${short(c.title, 30)}｜价格 ${c.price}｜在搜索“${short(c.query, 20)}”第 ${c.page} 页`);
			return (
				`【现在必须下单】${why}。${rule}不要再问用户，也不要再找新商品：` +
				`从你看过的商品里选最接近用户需求的一个，回到它的页面（不在当前页就用原来的搜索词再搜一次），选好规格后调用 buy_now。\n` +
				`你看过的商品：\n${list.join("\n")}`
			);
		}
		if (left <= CLOSING_SOFT) {
			log({ type: "closing_soft", turn: turns, left });
			return `【还剩 ${left} 个回合】提问和操作都算回合，回合用完还没下单这一单就算失败。请尽快确定商品、选好规格并下单。`;
		}
		return "";
	}

	pi.on("tool_call", async (event) => {
		if (!ENV_TOOLS.has(event.toolName)) return undefined;
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
		if (ASK_MODE && on("confirm_gate") && event.toolName === "buy_now" && cur.page_type === "product_detail") {
			const gate = confirmGate();
			if (gate) {
				log({ type: "confirm_block", turn: turns, asin: cur.product?.asin, selected: cur.selected_options, issues: gate.issues });
				return { block: true, reason: gate.reason };
			}
		}
		return undefined;
	});

	pi.on("tool_result", async (event) => {
		const isEnv = ENV_TOOLS.has(event.toolName);
		if (!ASK_MODE || !(isEnv || event.toolName === "ask_shopper") || event.isError || done || (event.details as any)?.done) return undefined;
		const notes: string[] = [];
		const asin = String(cur.product?.asin ?? "");
		if (isEnv && on("ask_hint") && cur.page_type === "product_detail" && !hinted.has(asin) && asksLeft() > 0 && !mustClose()) {
			hinted.add(asin);
			const axes = unresolvedAxes();
			if (axes.length) {
				if (on("choice")) {
					const listed = axes.map(({ axis, values, tied }) => `「${axis}」：${quoteValues(tied.length < values.length ? tied : values)}`);
					notes.push(
						`【提示】这个商品有用户还没确定的规格：${listed.join("；")}。请用 ask_shopper 把这些选项原样列给用户，让用户直接选（例如“这款有……，您要哪个？”），再按用户选的去选规格${asksLeftNote()}。`,
					);
				} else {
					const listed = axes.map(({ axis, values, tied }) =>
						tied.length < values.length
							? `「${axis}」（按用户说过的话还有 ${tied.length} 项符合：${listValues(tied)}）`
							: `「${axis}」（可选 ${listValues(values)}）`,
					);
					notes.push(`【提示】这个商品有用户的话还确定不了的规格：${listed.join("、")}。如果从需求里看不出该选哪个，可以用 ask_shopper 问用户${asksLeftNote()}。`);
				}
				log({ type: "ask_hint", turn: turns, step: stepsUsed, asin, axes: axes.map((a) => a.axis), tied: axes.map((a) => a.tied.length) });
			}
		}
		if (isEnv && on("dialogue_ledger") && dialogue.length) {
			const qa = dialogue.map((d) => `问「${short(d.question, 40)}」答「${short(d.answer, 60)}」`);
			notes.push(`【已知需求】最初需求：${initialQuery}\n用户补充：${qa.join("；")}`);
		}
		const closing = closingNote();
		if (closing) notes.push(closing);
		if (!notes.length) return undefined;
		return { content: [...event.content, { type: "text" as const, text: `\n——\n${notes.join("\n")}` }] };
	});

	pi.on("agent_end", async () => {
		if (relayed) {
			relayed = false;
			return;
		}
		if (!on("nudge") || done || stepsUsed >= MAX_STEPS || turns >= MAX_LLM_TURNS || nudges >= MAX_NUDGES) return;
		nudges += 1;
		log({ type: "finish_nudge", turn: turns, step: stepsUsed, n: nudges });
		pi.sendUserMessage(
			"任务还没有完成：需要购买一件商品才算完成。请继续，在已看过的商品中选最符合需求的一个（或继续搜索），选好规格后调用 buy_now。",
			{ deliverAs: "followUp" },
		);
	});
}
