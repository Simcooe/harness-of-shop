/**
 * 任务状态层（第三版 harness）：把"用户要什么、看过的商品各满足哪几条"从对话里拿出来，单独维护成两张表，
 * 提问、下单前核对和收尾都按这两张表来做。和 plugin.ts 的 guard、nudge 一起加载：那两个只管动作在页面上是否有效，
 * 和任务状态无关，原样保留。
 *
 * - 需求表：用户每说一次话，调一次小模型从对话里重新整理。只依据用户的话，每条都附用户原话；原话在用户消息里
 *   找不到的条目丢掉。品类、品牌、预算、规格这几个方面一条都没有时记为"还不知道"。
 * - 商品表：打开商品详情页时记下页面上的事实（标题、店铺、类目、价格、属性、规格）；再调小模型把需求表里已确认的
 *   每一条对照页面判成符合 / 不符合 / 页面没写，证据必须能在页面原文里找到，找不到就算页面没写。需求表变了就重判。
 * - 状态卡：每次调用模型前附在发给模型的最后一条消息后面（只改发给模型的内容，不改会话记录）。
 * - 上下文窗口（SHOP_STATE_WINDOW=K，K>0 时生效）：只保留最近 K 步原文，更早的换成一句说明，信息以状态卡为准。
 *
 * 组件（SHOP_STATE："all" = tables,ask,verify,close）：
 *   tables  维护两张表并显示状态卡；其余组件都依赖它
 *   ask     刚打开的商品有用户还没定的规格时，提示把选项原样列给用户挑（对应 v2 的规格选择题）
 *   verify  下单前对表核对：页面不符合需求、规格没定、用户说过不要这件、需求表还有空缺（预算不知道、用户说有要求
 *           但没说具体）时，每件商品拦一次（对应 v2 的购买前确认）
 *   close   回合快用完或用户说了再见时，按商品表排好序，要求立即下单（对应 v2 的收尾）
 *   consent 可选，"all" 不含：核对时还要求用户已经同意买这件（论文官方提示里的规则）。论文设置里买家在每个细节
 *           都确认前一律拒绝，开这条会多花不少回合
 * 只用 agent 也能看到的信息（对话和页面），不读 gold 和模拟买家的配置。
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const COMPONENTS = ["tables", "ask", "verify", "close", "consent"];
const ALL = ["tables", "ask", "verify", "close"];
const requested = (process.env.SHOP_STATE || "all").split(",").map((s) => s.trim()).filter(Boolean);
const unknownComponents = requested.filter((c) => c !== "all" && !COMPONENTS.includes(c));
if (unknownComponents.length) throw new Error(`未知的任务状态组件：${unknownComponents.join(", ")}`);
const enabled = new Set(requested.includes("all") ? [...ALL, ...requested.filter((c) => c !== "all")] : requested);
const on = (component: string) => enabled.has("tables") && enabled.has(component);

const RUN_DIR = process.env.SHOP_RUN_DIR;
const WINDOW = Number(process.env.SHOP_STATE_WINDOW ?? "6");
const STATE_MODEL = process.env.SHOP_STATE_MODEL || "deepseek-v4-flash";
const MAX_LLM_TURNS = Number(process.env.SHOP_MAX_LLM_TURNS ?? "60");
const ASK_MODE = process.env.SHOP_ASK === "1";
const PROXY_URL = (process.env.MODEL_PROXY_BASE_URL ?? "").replace(/\/+$/, "");
// 收尾：和 v2 一样，含当前这一轮还剩不超过 CLOSING_SOFT 轮起提醒，不超过 CLOSING_HARD 轮或用户说了再见时要求立即下单
const CLOSING_SOFT = 12;
const CLOSING_HARD = 6;
const BYE = /再见|拜拜|bye/i;
const SLOTS = ["品类", "品牌", "型号款式", "规格", "预算", "其他"];
// 一条都没有时在状态卡上显示"还不知道"的方面
const BASIC_SLOTS = ["品类", "品牌", "预算", "规格"];
const STATUSES = ["已确认", "无要求", "待补充"];
// 论文设置里买家在信息问全之前一律拒绝下单，这种"还要确认"不等于商品不对，只有"不要这件"才是否定这件商品
const ATTITUDES = ["同意购买", "不要这件", "还要确认", "未表态"];

const REQ_SYSTEM = `你负责整理购物对话里的“用户需求表”。只能依据用户说过的话；助手说的话只用来理解用户在回答什么，助手的推荐、介绍和商品页面上的信息都不能当成用户的需求。

输出一个 JSON 对象，格式如下：
{"requirements": [{"slot": "品类", "aspect": "品类", "value": "粉底液", "status": "已确认", "quote": "我想买一瓶粉底液"}],
 "no_more": {"said": false, "quote": ""},
 "attitude": {"status": "未表态", "product": "", "quote": ""},
 "pending": []}

字段说明：
- requirements：用户提出的每一条要求。一条只写一个方面，用户一句话说了几个方面就拆成几条。
  - slot 只能是：品类、品牌、型号款式、规格、预算、其他。规格指商品页上要选的那类（颜色、色号、尺码、容量、数量、套餐、款式编号等）；用途、功能、材质、产地、口味、效果、送礼对象等放进“其他”。
  - aspect：这条说的是哪方面，两到四个字（如 颜色、尺码、材质、产地、用途）。
  - value：用户要求的内容，尽量保留用户的说法；用户从助手列出的选项里挑了一个时，照抄那个选项的原文。
  - status：用户明确给出了要求是“已确认”；用户说这方面无所谓、没要求、都可以，是“无要求”（value 留空）；用户只说了有这方面的要求、还没说具体（比如“颜色我有要求”，或者说信息还不完整、指出还缺哪类），是“待补充”（value 写缺的是哪方面）。
  - 用户用提问的方式说出的条件也是要求（比如“这个是乌木的吗？”“组合装里有卸妆水吗？”“价格在我的预算 115-125 元内吗？”），记为“已确认”。
  - quote：用户原话里支持这一条的一小段，逐字摘抄，不要改写。
  - 用户后来改了主意，只保留最新的说法。
- no_more：用户是否明确说过没有别的要求了；said 为 true 时 quote 逐字摘抄那句话。
- attitude：用户最后一次说话时对购买的态度，status 只能是：同意购买（同意买这件）、不要这件（明确说这件商品不符合、不想要）、还要确认（还不肯买，要先确认信息，或者说信息还不完整）、未表态。态度针对某件商品时，product 填这件商品的商品ID（从“看过的商品”里对应，对应不上就留空）；quote 逐字摘抄用户表态的话。
- pending：用户最后一次说话里问助手、还等着助手回答的问题，逐字摘抄，没有就是空列表。`;

const EVAL_SYSTEM = `你负责把一件商品对照用户需求逐条核对。只能依据给出的商品页面信息（标题、店铺、类目、价格、属性、可选规格）；页面上没写的就判“页面没写”，不要凭常识或商品名推测页面上没写出来的信息。

输出一个 JSON 对象，格式如下：
{"checks": [{"id": "R1", "verdict": "符合", "evidence": "页面原文片段"}],
 "options": [{"axis": "颜色分类", "wanted": ["01象牙白"], "any_ok": false}]}

- checks：每条需求一项。verdict 只能是：符合、不符合、页面没写。evidence 从页面信息里逐字摘抄支持判断的一小段（预算就写页面上的价格）；判“页面没写”时留空。
  - 预算：按页面价格判断；价格是区间时，区间里有价格在预算内就算符合。
  - 规格类需求：可选规格里有用户要的选项就算符合；所有选项都不满足才算不符合。
- options：页面上每个可选规格项一项（选项少于两个的跳过）。wanted 列出按用户的需求应该选的选项，照抄页面上的选项原文：用户的需求只对得上一个选项时只列这一个；有几个选项都对得上就都列出来；用户没提过这方面就列出全部选项。用户明确说过这方面无所谓时 any_ok 为 true。`;

type Status = "已确认" | "无要求" | "待补充";

interface Req {
	id: string;
	slot: string;
	aspect: string;
	value: string;
	status: Status;
	quote: string;
}

interface Exchange {
	question: string;
	answer: string;
	turn: number;
	/** 提问时 agent 所在（或最近打开）的商品，用户表态没指明商品时归到这件 */
	asin: string;
}

interface Product {
	asin: string;
	title: string;
	brand: string;
	category: string;
	price: string;
	prices: number[];
	attrs: string[];
	options: Record<string, string[]>;
	selected: Record<string, string>;
	selectedPrice: string;
	/** 看过的详情子页（Description / Features / Reviews）上写了什么 */
	subpages: Record<string, string>;
	query: string;
	page: number;
	order: number;
}

interface Check {
	verdict: "符合" | "不符合" | "页面没写";
	evidence: string;
}

interface Axis {
	axis: string;
	values: string[];
	wanted: string[];
	anyOk: boolean;
}

interface Eval {
	key: string;
	checks: Map<string, Check>;
	axes: Axis[];
}

function logTo(file: string, record: Record<string, unknown>): void {
	if (!RUN_DIR) return;
	mkdirSync(RUN_DIR, { recursive: true });
	appendFileSync(join(RUN_DIR, file), `${JSON.stringify({ t: new Date().toISOString(), ...record })}\n`);
}

const logEvent = (record: Record<string, unknown>) => logTo("harness_events.jsonl", record);

const short = (s: unknown, n = 30) => {
	const text = String(s ?? "").replace(/\s+/g, " ").trim();
	return text.length > n ? `${text.slice(0, n)}…` : text;
};

const looseKey = (s: unknown) => String(s ?? "").normalize("NFKC").replace(/\s+/g, "").toLowerCase();
// 核对原话和页面证据时忽略空白、标点、markdown 符号和货币单位（模型摘抄价格时常写成"89元"）
const squash = (s: unknown) =>
	String(s ?? "")
		.normalize("NFKC")
		.toLowerCase()
		.replace(/[\s，。,.！!？?、；;：:“”"'‘’（）()【】[\]{}|/~～·…—_\-*#元¥￥]/g, "");

/** 摘抄的片段是否出自原文：忽略标点后是原文的子串，或者（较长的片段）八成以上的相邻两字出现在原文里。 */
function quotedFrom(quote: string, source: string): boolean {
	const q = squash(quote);
	if (!q) return false;
	const src = squash(source);
	if (src.includes(q)) return true;
	if (q.length < 6) return false;
	let hit = 0;
	for (let i = 0; i + 2 <= q.length; i++) if (src.includes(q.slice(i, i + 2))) hit += 1;
	return hit / (q.length - 1) >= 0.8;
}

const FIELD_LABEL = /^\s*(商品ID|标题|店铺|类目|价格|属性|可选规格|当前规格价格|描述|卖点|评价)\s*[：:]/;

/** 页面证据是否出自页面：去掉开头的字段名；模型常把几个属性拼成一段（如“速溶；无糖”），每一段都在页面上也算。 */
function evidenceFrom(evidence: string, facts: string): boolean {
	const e = evidence.replace(FIELD_LABEL, "").trim();
	if (quotedFrom(e, facts)) return true;
	const parts = e.split(/[；;，,、\s]+/).map((s) => s.replace(FIELD_LABEL, "").trim()).filter(Boolean);
	return parts.length > 0 && parts.every((s) => quotedFrom(s, facts));
}

function parseJson(text: string): any {
	let body = text.trim();
	const fenced = body.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenced) body = fenced[1].trim();
	try {
		return JSON.parse(body);
	} catch {
		const start = body.indexOf("{");
		const end = body.lastIndexOf("}");
		if (start >= 0 && end > start) return JSON.parse(body.slice(start, end + 1));
		throw new Error("不是 JSON");
	}
}

/** 调小模型（temperature 0、关闭思考、要求 JSON），每次调用的用量都记进 state_log.jsonl，失败返回 null。 */
async function chatJson(kind: string, system: string, user: string, maxTokens: number, turn: number): Promise<any | null> {
	const started = Date.now();
	let lastError = "";
	for (let attempt = 0; attempt < 3; attempt++) {
		let usage: unknown = null;
		try {
			const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.MODEL_PROXY_API_KEY ?? ""}` },
				body: JSON.stringify({
					model: STATE_MODEL,
					messages: [
						{ role: "system", content: system },
						{ role: "user", content: user },
					],
					temperature: 0,
					max_tokens: maxTokens,
					thinking: { type: "disabled" },
					response_format: { type: "json_object" },
				}),
				signal: AbortSignal.timeout(90_000),
			});
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const body = (await res.json()) as any;
			usage = body?.usage ?? null;
			const parsed = parseJson(String(body?.choices?.[0]?.message?.content ?? ""));
			logTo("state_log.jsonl", { kind, turn, attempt, ms: Date.now() - started, usage, input: user, output: parsed });
			return parsed;
		} catch (err) {
			lastError = err instanceof Error ? err.message : String(err);
			logTo("state_log.jsonl", { kind, turn, attempt, ms: Date.now() - started, usage, error: lastError });
			await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
		}
	}
	return null;
}

function priceText(price: unknown): string {
	if (Array.isArray(price)) return price.map(String).join("–");
	return price === undefined || price === null || price === "" ? "未知" : String(price);
}

function priceNumbers(price: unknown): number[] {
	return (String(Array.isArray(price) ? price.join(" ") : (price ?? "")).match(/\d+(?:\.\d+)?/g) ?? []).map(Number);
}

const CN_DIGIT: Record<string, number> = { 零: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
const CN_UNIT: Record<string, number> = { 十: 10, 百: 100, 千: 1000, 万: 10000 };

function toNumber(text: string): number | null {
	if (/^\d+(?:\.\d+)?$/.test(text)) return Number(text);
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

const NUM = "(\\d+(?:\\.\\d+)?|[零一二两三四五六七八九十百千万]+)";
const UNIT = "(?:元|块钱|块|rmb)?";

/**
 * 把预算的说法换成价格区间，口径和评测器一致：区间照用；“以内/不超过”是上限；“左右/上下”上下浮动 20%；“出头”上浮 20%。
 * 解析不了的预算交给小模型判断。
 */
function parseBudget(text: string): { lo: number; hi: number } | null {
	const t = text.replace(/\s+/g, "").replace(/[,，]/g, "").toLowerCase();
	const num = (s: string) => toNumber(s);
	let m = t.match(new RegExp(`${NUM}${UNIT}(?:-|~|～|—|–|到|至)${NUM}`));
	if (m) {
		const a = num(m[1]);
		const b = num(m[2]);
		if (a !== null && b !== null) return { lo: Math.min(a, b), hi: Math.max(a, b) };
	}
	m = t.match(new RegExp(`(?:不超过|不高于|最多|顶多|低于|少于|不到)${NUM}`)) ?? t.match(new RegExp(`${NUM}${UNIT}(?:以内|以下|之内)`));
	if (m) {
		const a = num(m[1]);
		if (a !== null) return { lo: 0, hi: a };
	}
	m = t.match(new RegExp(`${NUM}${UNIT}出头`));
	if (m) {
		const a = num(m[1]);
		if (a !== null) return { lo: a, hi: a * 1.2 };
	}
	m = t.match(new RegExp(`${NUM}${UNIT}(?:以上|起)`));
	if (m) {
		const a = num(m[1]);
		if (a !== null) return { lo: a, hi: Infinity };
	}
	m = t.match(new RegExp(`^(?:大概|大约|差不多|预算)?${NUM}${UNIT}(?:左右|上下|附近)?$`)) ?? t.match(new RegExp(`${NUM}${UNIT}(?:左右|上下)`));
	if (m) {
		const a = num(m[1]);
		if (a !== null && a > 0) return { lo: a * 0.8, hi: a * 1.2 };
	}
	return null;
}

const inBudget = (prices: number[], b: { lo: number; hi: number }) =>
	prices.length > 0 && Math.max(...prices) >= b.lo - 1e-6 && Math.min(...prices) <= b.hi + 1e-6;

function quoteValues(values: string[], n = 12): string {
	const shown = values.slice(0, n).map((v) => `“${v}”`).join("、");
	return values.length > n ? `${shown} 等 ${values.length} 项` : shown;
}

const rowKey = (r: Req) => `${r.slot}|${r.aspect}|${r.value}`;

function factsText(p: Product): string {
	const lines = [
		`商品ID：${p.asin}`,
		`标题：${p.title}`,
		`店铺：${p.brand}｜类目：${p.category}`,
		`价格：${p.price}`,
		`属性：${p.attrs.length ? p.attrs.join("；") : "无"}`,
		`可选规格：${Object.keys(p.options).length ? JSON.stringify(p.options) : "无"}`,
	];
	for (const [name, text] of Object.entries(p.subpages)) if (!text.startsWith("（")) lines.push(`${name}：${text}`);
	return lines.join("\n");
}

function parseInitialState(): any {
	try {
		const state = JSON.parse(process.env.SHOP_INITIAL_STATE || "{}");
		if (state && typeof state === "object" && state.page_type) return state;
	} catch {
		// 用默认的搜索首页
	}
	return { page_type: "search_home" };
}

export default function (pi: ExtensionAPI) {
	let cur: any = parseInitialState();
	let opening = "";
	let turns = 0;
	let done = false;
	let shopperSaidBye = false;
	let lastSearch = { query: "", page: 1 };
	let lastOpened = "";
	// 刚到达、还没给过规格提示的商品页
	let freshAsin = "";

	const dialogue: Exchange[] = [];
	let reqs: Req[] = [];
	let noMore = "";
	let pending: string[] = [];
	let pendingTurn = 0;
	const attitudes = new Map<string, { status: string; quote: string; turn: number }>();
	const products = new Map<string, Product>();
	const evals = new Map<string, Eval>();
	const evalJobs = new Map<string, Promise<void>>();
	let reqJob: Promise<void> = Promise.resolve();
	const hinted = new Set<string>();
	const verified = new Set<string>();

	const turnsLeft = () => MAX_LLM_TURNS - turns + 1;
	const mustClose = () => on("close") && ASK_MODE && !done && (shopperSaidBye || turnsLeft() <= CLOSING_HARD);

	// ---- 需求表 ----

	async function extractRequirements(): Promise<void> {
		const snapshot = dialogue.slice();
		const lines = [`用户：${opening}`];
		for (const d of snapshot) lines.push(`助手：${short(d.question, 600)}`, `用户：${d.answer}`);
		const seen = [...products.values()].map((p) => `[${p.asin}] ${short(p.title, 40)}`);
		const input = `看过的商品：\n${seen.length ? seen.join("\n") : "（还没有）"}\n\n对话（按时间顺序）：\n${lines.join("\n")}`;
		const out = await chatJson("requirements", REQ_SYSTEM, input, 1500, turns);
		if (!out) return;

		const userText = [opening, ...snapshot.map((d) => d.answer)].join("\n");
		const lastAnswer = snapshot.length ? snapshot[snapshot.length - 1].answer : opening;
		const rows: Req[] = [];
		const dropped: unknown[] = [];
		const keys = new Set<string>();
		for (const raw of Array.isArray(out.requirements) ? out.requirements : []) {
			const status = String(raw?.status ?? "");
			const quote = String(raw?.quote ?? "").trim();
			const slot = SLOTS.includes(String(raw?.slot)) ? String(raw.slot) : "其他";
			const aspect = short(String(raw?.aspect ?? "").trim() || slot, 8);
			const value = String(raw?.value ?? "").trim();
			if (!STATUSES.includes(status) || (status === "已确认" && !value) || !quotedFrom(quote, userText)) {
				dropped.push(raw);
				continue;
			}
			const row: Req = { id: "", slot, aspect, value: status === "无要求" ? "" : value, status: status as Status, quote };
			if (keys.has(rowKey(row))) continue;
			keys.add(rowKey(row));
			rows.push(row);
		}
		rows.forEach((r, i) => (r.id = `R${i + 1}`));
		reqs = rows;

		const nm = out.no_more ?? {};
		if (nm.said === true && quotedFrom(String(nm.quote ?? ""), userText)) noMore = String(nm.quote);
		pending = (Array.isArray(out.pending) ? out.pending : [])
			.map((q: unknown) => String(q ?? "").trim())
			.filter((q: string) => q && quotedFrom(q, lastAnswer))
			.slice(0, 3);
		// 用户的问题只在它刚出现时提一次；一直挂在状态卡上，模型会被它牵着走、反复去找答案
		pendingTurn = snapshot.length ? snapshot[snapshot.length - 1].turn : 0;
		const att = out.attitude ?? {};
		const attStatus = String(att.status ?? "");
		if (snapshot.length && ATTITUDES.includes(attStatus) && attStatus !== "未表态" && quotedFrom(String(att.quote ?? ""), lastAnswer)) {
			const named = String(att.product ?? "").trim();
			const asin = products.has(named) ? named : snapshot[snapshot.length - 1].asin;
			if (asin) attitudes.set(asin, { status: attStatus, quote: String(att.quote), turn: snapshot[snapshot.length - 1].turn });
		}
		if (dropped.length) logTo("state_log.jsonl", { kind: "requirements_dropped", turn: turns, dropped });
	}

	function scheduleRequirements(): void {
		reqJob = reqJob.then(extractRequirements).catch((err) => {
			logTo("state_log.jsonl", { kind: "requirements_error", turn: turns, error: String(err) });
		});
	}

	const unknownSlots = () => BASIC_SLOTS.filter((s) => !reqs.some((r) => r.slot === s));

	// ---- 商品表 ----

	/** 需求表里会影响商品核对的部分；它变了，看过的商品都要重判。 */
	const evalKey = () =>
		JSON.stringify(
			reqs
				.filter((r) => r.status === "已确认" || (r.status === "无要求" && (r.slot === "规格" || r.slot === "型号款式")))
				.map((r) => `${r.status}|${rowKey(r)}`)
				.sort(),
		);

	function axesOf(p: Product): Axis[] {
		return Object.entries(p.options)
			.filter(([, values]) => values.length >= 2)
			.map(([axis, values]) => ({ axis, values, wanted: values, anyOk: false }));
	}

	async function evaluateProduct(p: Product, key: string): Promise<void> {
		const rows = reqs.filter((r) => r.status === "已确认");
		const relaxed = reqs.filter((r) => r.status === "无要求");
		if (!rows.length && !relaxed.length) {
			evals.set(p.asin, { key, checks: new Map(), axes: axesOf(p) });
			return;
		}
		const listing = rows.map((r) => `${r.id} ${r.slot === r.aspect ? r.slot : `${r.slot}·${r.aspect}`}｜${r.value}`);
		if (relaxed.length) listing.push(`（用户说无所谓的方面：${relaxed.map((r) => r.aspect).join("、")}）`);
		const input = `用户需求：\n${listing.join("\n") || "（还没有已确认的需求）"}\n\n商品页面：\n${factsText(p)}`;
		const out = await chatJson("product", EVAL_SYSTEM, input, 1200, turns);
		if (!out) return;

		const facts = factsText(p);
		const byId = new Map(rows.map((r) => [r.id, r]));
		const checks = new Map<string, Check>();
		const downgraded: unknown[] = [];
		for (const raw of Array.isArray(out.checks) ? out.checks : []) {
			const row = byId.get(String(raw?.id ?? ""));
			let verdict = String(raw?.verdict ?? "");
			const evidence = String(raw?.evidence ?? "").trim();
			if (!row || !["符合", "不符合", "页面没写"].includes(verdict)) continue;
			if (verdict !== "页面没写" && !evidenceFrom(evidence, facts)) {
				downgraded.push(raw);
				verdict = "页面没写";
			}
			checks.set(rowKey(row), { verdict: verdict as Check["verdict"], evidence: verdict === "页面没写" ? "" : evidence });
		}
		// 预算能换成价格区间时按规则判，不用小模型（它会把 64 判成在 115-125 元之内）
		for (const row of rows.filter((r) => r.slot === "预算")) {
			const budget = parseBudget(row.value);
			if (!budget || !p.prices.length) continue;
			const fits = inBudget(p.prices, budget);
			checks.set(rowKey(row), { verdict: fits ? "符合" : "不符合", evidence: `价格 ${p.price}` });
		}
		const picks = new Map<string, any>((Array.isArray(out.options) ? out.options : []).map((o: any) => [String(o?.axis ?? ""), o]));
		const axes = axesOf(p).map((ax) => {
			const raw = picks.get(ax.axis);
			if (!raw) return ax;
			const wanted = (Array.isArray(raw.wanted) ? raw.wanted : [])
				.map((w: unknown) => ax.values.find((v) => looseKey(v) === looseKey(w)))
				.filter((v: string | undefined): v is string => Boolean(v));
			return { ...ax, wanted: wanted.length ? [...new Set<string>(wanted)] : ax.values, anyOk: raw.any_ok === true };
		});
		evals.set(p.asin, { key, checks, axes });
		if (downgraded.length) logTo("state_log.jsonl", { kind: "product_downgraded", turn: turns, asin: p.asin, downgraded });
	}

	/** 等需求表整理完，再把看过的商品都按最新的需求表核对一遍（没变的不重判）。 */
	async function settle(): Promise<void> {
		await reqJob;
		const key = evalKey();
		const jobs: Promise<void>[] = [];
		for (const p of products.values()) {
			if (evals.get(p.asin)?.key === key) continue;
			const id = `${p.asin}#${key}`;
			let job = evalJobs.get(id);
			if (!job) {
				job = evaluateProduct(p, key).catch((err) => {
					logTo("state_log.jsonl", { kind: "product_error", turn: turns, asin: p.asin, error: String(err) });
				});
				evalJobs.set(id, job);
			}
			jobs.push(job);
		}
		await Promise.all(jobs);
	}

	function verdicts(asin: string) {
		const ev = evals.get(asin);
		const out = { ok: [] as Req[], bad: [] as Req[], unknown: [] as Req[] };
		for (const r of reqs.filter((x) => x.status === "已确认")) {
			const c = ev?.checks.get(rowKey(r));
			if (c?.verdict === "符合") out.ok.push(r);
			else if (c?.verdict === "不符合") out.bad.push(r);
			else out.unknown.push(r);
		}
		return out;
	}

	/** 商品表排序：符合的需求多、不符合的少的在前；用户同意过的最前，明确不要的靠后；同分时最近看的在前。 */
	function ranked(): Product[] {
		const score = (p: Product) => {
			const v = verdicts(p.asin);
			const att = attitudes.get(p.asin)?.status;
			return 2 * v.ok.length - 3 * v.bad.length + (att === "同意购买" ? 10 : att === "不要这件" ? -10 : 0);
		};
		return [...products.values()].sort((a, b) => score(b) - score(a) || b.order - a.order);
	}

	function axisLine(p: Product, ax: Axis): string {
		const want = ax.anyOk ? "用户说无所谓" : ax.wanted.length === 1 ? `用户要“${ax.wanted[0]}”` : `用户还没定（符合的有 ${quoteValues(ax.wanted, 6)}）`;
		const sel = p.selected[ax.axis];
		return `规格「${ax.axis}」：${want}｜已选 ${sel === undefined ? "无" : sel}`;
	}

	// ---- 状态卡 ----

	function pageLine(): string {
		switch (cur?.page_type) {
			case "search_results":
				return `搜索结果（搜索词“${cur.query ?? ""}”，第 ${cur.page ?? 1} 页）`;
			case "product_detail":
				return `商品详情页 [${cur.product?.asin ?? ""}]`;
			case "information_subpage":
				return `商品 [${cur.product?.asin ?? ""}] 的 ${cur.subpage ?? "详情"} 子页`;
			case "search_home":
				return "搜索首页";
			default:
				return "（未知）";
		}
	}

	function renderCard(): string {
		const L = ["【任务状态】（系统每一步自动整理：需求只记用户说过的话，商品只记页面上写的事实）"];
		L.push(`回合：第 ${turns} 轮，共 ${MAX_LLM_TURNS} 轮（含这一轮还剩 ${turnsLeft()} 轮）｜当前页面：${pageLine()}`);
		L.push(`用户最开始说：「${opening}」`);
		L.push("需求表：");
		if (!reqs.length) L.push("  （还没有）");
		for (const r of reqs) {
			const value = r.status === "无要求" ? "用户说无所谓" : r.status === "待补充" ? `用户说有要求、还没说具体（${r.value || r.aspect}）` : r.value;
			// 这条在每件看过的商品页面上都没写：多半是页面上本来就没有的信息
			const checked = [...products.keys()].map((a) => evals.get(a)?.checks.get(rowKey(r))).filter(Boolean);
			const nowhere = r.status === "已确认" && checked.length >= 2 && checked.every((c) => c!.verdict === "页面没写") ? `（看过的 ${checked.length} 件商品页面上都没写）` : "";
			L.push(`  ${r.id} ${r.aspect}：${value}${nowhere}｜原话「${short(r.quote, 40)}」`);
		}
		const unknown = unknownSlots();
		if (unknown.length) L.push(`  还不知道：${unknown.join("、")}`);
		L.push(noMore ? `  用户说过没有别的要求了：「${short(noMore, 30)}」` : "  用户还没说过“没有别的要求了”");
		const atts = [...attitudes.entries()].sort((a, b) => b[1].turn - a[1].turn).slice(0, 3);
		const attLabel: Record<string, string> = { 同意购买: "同意买", 不要这件: "说过不要", 还要确认: "对这件还要确认" };
		for (const [asin, a] of atts) L.push(`用户${attLabel[a.status] ?? a.status} [${asin}]：「${short(a.quote, 40)}」`);
		if (pending.length && pendingTurn >= turns - 1) L.push(`用户刚问你：${pending.map((q) => `「${short(q, 50)}」`).join("；")}`);
		if (!products.size) {
			L.push("商品表：（还没打开过商品）");
			return L.join("\n");
		}
		L.push(`商品表（打开过的 ${products.size} 件，按符合程度排序）：`);
		ranked()
			.slice(0, 8)
			.forEach((p, i) => {
				const variant = p.selectedPrice ? `（当前规格 ${p.selectedPrice}）` : "";
				L.push(`  ${i + 1}. [${p.asin}] ${short(p.title, 30)}｜价格 ${p.price}${variant}｜在搜索“${short(p.query, 20)}”第 ${p.page} 页`);
				const ev = evals.get(p.asin);
				if (!ev) {
					L.push("     （还没核对）");
					return;
				}
				const v = verdicts(p.asin);
				const ids = (rows: Req[]) => (rows.length ? rows.map((r) => r.id).join(" ") : "无");
				const bad = v.bad.map((r) => `${r.id}（页面：${short(ev.checks.get(rowKey(r))?.evidence, 20)}）`).join(" ");
				L.push(`     符合 ${ids(v.ok)}｜不符合 ${bad || "无"}｜页面没写 ${ids(v.unknown)}`);
				for (const ax of ev.axes) L.push(`     ${axisLine(p, ax)}`);
				const subs = Object.entries(p.subpages);
				if (subs.length) L.push(`     看过的详情子页：${subs.map(([name, text]) => (text.startsWith("（") ? `${name}${text}` : `${name}：${short(text, 30)}`)).join("、")}`);
			});
		return L.join("\n");
	}

	// ---- 上下文窗口 ----

	function windowed(messages: any[]): { messages: any[]; dropped: number } {
		if (WINDOW <= 0) return { messages, dropped: 0 };
		const starts = messages.map((m, i) => (m?.role === "assistant" ? i : -1)).filter((i) => i >= 0);
		if (starts.length <= WINDOW) return { messages, dropped: 0 };
		// 从某一步的模型消息切开，工具调用和它的结果不会被拆散
		const from = starts[starts.length - WINDOW];
		const dropped = starts.length - WINDOW;
		const stub = {
			role: "user",
			content: [{ type: "text", text: `（前面 ${dropped} 步的对话和页面已省略，需要的信息都整理在最后的【任务状态】里。）` }],
			timestamp: messages[0]?.timestamp ?? Date.now(),
		};
		return { messages: [stub, ...messages.slice(from)], dropped };
	}

	function appendText(messages: any[], text: string): void {
		const last = messages[messages.length - 1];
		if (!last || last.role === "assistant") {
			messages.push({ role: "user", content: [{ type: "text", text }], timestamp: Date.now() });
			return;
		}
		const block = { type: "text", text: `\n\n${text}` };
		last.content = typeof last.content === "string" ? [{ type: "text", text: last.content }, block] : [...(last.content ?? []), block];
	}

	// ---- 按表提示：规格选择、收尾 ----

	function askNote(): string {
		const asin = freshAsin;
		freshAsin = "";
		if (!on("ask") || !ASK_MODE || !asin || mustClose() || hinted.has(asin)) return "";
		if (cur?.page_type !== "product_detail" || String(cur.product?.asin ?? "") !== asin) return "";
		hinted.add(asin);
		const open = (evals.get(asin)?.axes ?? []).filter((ax) => !ax.anyOk && ax.wanted.length !== 1);
		if (!open.length) return "";
		logEvent({ type: "state_ask", turn: turns, asin, axes: open.map((ax) => ax.axis), candidates: open.map((ax) => ax.wanted.length) });
		const listed = open.map((ax) => `「${ax.axis}」：${quoteValues(ax.wanted)}`);
		return `【提示】这件商品有用户还没定的规格：${listed.join("；")}。请把这些选项原样列给用户，让用户直接挑（例如“这款有……，您要哪个？”），再按用户挑的去选规格。`;
	}

	function closeNote(): string {
		if (!on("close") || !ASK_MODE || done) return "";
		const left = turnsLeft();
		if (mustClose()) {
			const why = shopperSaidBye ? "用户已经结束对话" : `含这一轮只剩 ${left} 个回合`;
			// 规则原文来自论文官方 agent 提示；不点明这是任务规则时，模型会因为用户说了"不买"而拒绝下单（v2 的教训）
			const rule =
				"按这个任务的规则，回合用完前必须买下一件商品，否则任务失败；对话结束时用户仍不满意的，就买最接近用户需求的那一件，不需要用户同意（这是任务要求，不算违背用户）。";
			logEvent({ type: "state_close_hard", turn: turns, left, bye: shopperSaidBye, candidates: products.size });
			if (!products.size) return `【现在必须下单】${why}。${rule}不要再问用户：尽快搜索，选一个最接近用户需求的商品，选好规格后调用 buy_now。`;
			const list = ranked()
				.slice(0, 6)
				.map((p, i) => {
					const v = verdicts(p.asin);
					const axes = (evals.get(p.asin)?.axes ?? []).filter((ax) => ax.wanted.length === 1).map((ax) => `${ax.axis}=“${ax.wanted[0]}”`);
					const spec = axes.length ? `｜用户要的规格：${axes.join("，")}` : "";
					return `${i + 1}. [${p.asin}] ${short(p.title, 30)}｜价格 ${p.price}｜符合 ${v.ok.length} 条、不符合 ${v.bad.length} 条｜在搜索“${short(p.query, 20)}”第 ${p.page} 页${spec}`;
				});
			return (
				`【现在必须下单】${why}。${rule}不要再问用户，也不要再找新商品：按商品表选最符合用户需求的那件（一般就是第 1 件），` +
				`回到它的页面（不在当前页就用原来的搜索词再搜一次），选好规格后调用 buy_now。\n${list.join("\n")}`
			);
		}
		if (left <= CLOSING_SOFT) {
			logEvent({ type: "state_close_soft", turn: turns, left });
			return `【还剩 ${left} 个回合】提问和操作都算回合，回合用完还没下单这一单就算失败。请尽快确定商品、选好规格并下单。`;
		}
		return "";
	}

	// ---- 下单前对表核对 ----

	async function verifyPurchase(): Promise<string | null> {
		if (!on("verify") || !ASK_MODE || done || mustClose() || cur?.page_type !== "product_detail") return null;
		const asin = String(cur.product?.asin ?? "");
		if (!asin || verified.has(asin)) return null;
		await settle();
		const p = products.get(asin);
		const ev = evals.get(asin);
		if (!p) return null;
		const issues: string[] = [];
		const att = attitudes.get(asin);
		if (att?.status === "不要这件") issues.push(`用户说过不要这件：「${short(att.quote, 40)}」`);
		else if (on("consent") && att?.status !== "同意购买") issues.push("用户还没同意买这件");
		const gaps = [...(unknownSlots().includes("预算") ? ["预算"] : []), ...reqs.filter((r) => r.status === "待补充").map((r) => r.value || r.aspect)];
		if (gaps.length) issues.push(`需求表里还不清楚：${gaps.join("、")}`);
		const v = verdicts(asin);
		if (v.bad.length) {
			const bad = v.bad.map((r) => `${r.id} ${r.aspect}（用户要“${short(r.value, 20)}”，页面：${short(ev?.checks.get(rowKey(r))?.evidence, 20)}）`);
			issues.push(`按页面看不符合：${bad.join("；")}`);
		}
		const selectedPrices = priceNumbers(p.selectedPrice);
		for (const r of reqs.filter((x) => x.status === "已确认" && x.slot === "预算")) {
			const budget = parseBudget(r.value);
			if (budget && selectedPrices.length && !inBudget(selectedPrices, budget) && !v.bad.includes(r))
				issues.push(`当前规格价格 ${p.selectedPrice} 不在用户的预算（“${short(r.value, 20)}”）内`);
		}
		for (const ax of ev?.axes ?? []) {
			const sel = p.selected[ax.axis];
			if (ax.anyOk) {
				if (sel === undefined) issues.push(`规格「${ax.axis}」还没选`);
			} else if (ax.wanted.length === 1) {
				if (sel === undefined) issues.push(`规格「${ax.axis}」还没选（用户要“${ax.wanted[0]}”）`);
				else if (looseKey(sel) !== looseKey(ax.wanted[0])) issues.push(`规格「${ax.axis}」选了“${sel}”，用户要的是“${ax.wanted[0]}”`);
			} else {
				issues.push(`规格「${ax.axis}」用户还没定（${sel === undefined ? "还没选" : `现在选的是“${sel}”`}；符合的有 ${quoteValues(ax.wanted, 8)}）`);
			}
		}
		if (!issues.length) return null;
		verified.add(asin);
		const askAlso = unknownSlots().filter((s) => s !== "规格" && s !== "预算");
		if (!noMore) askAlso.push("还有没有别的要求");
		logEvent({ type: "state_verify_block", turn: turns, asin, selected: p.selected, issues, ask_also: askAlso });
		const also = askAlso.length ? `确认时可以顺便问清：${askAlso.join("、")}（可以一次问完）。` : "";
		return (
			`【下单前核对】按需求表和商品表，这件还有问题：${issues.map((s, i) => `${i + 1}. ${s}`).join("；")}。` +
			`建议先把商品、规格和价格报给用户，请用户确认；${also}确实有把握的话，再调用一次 buy_now 就会直接购买。`
		);
	}

	// ---- 事件 ----

	pi.on("before_agent_start", async (event) => {
		if (!opening) {
			opening = String(event.prompt ?? "");
			if (on("tables")) scheduleRequirements();
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
		if (cur.page_type === "search_results") lastSearch = { query: String(cur.query ?? ""), page: Number(cur.page ?? 1) };
		if (cur.page_type !== "product_detail" && cur.page_type !== "information_subpage") return;
		const p = cur.product ?? {};
		const asin = String(p.asin ?? "");
		if (!asin) return;
		let rec = products.get(asin);
		if (!rec) {
			rec = {
				asin,
				title: String(p.title ?? ""),
				brand: String(p.brand ?? ""),
				category: String(p.category ?? ""),
				price: priceText(p.price),
				prices: priceNumbers(p.price),
				attrs: Array.isArray(p.key_attributes) ? p.key_attributes.map(String) : [],
				options: Object.fromEntries(Object.entries((cur.available_options ?? {}) as Record<string, unknown[]>).map(([k, vs]) => [k, (vs ?? []).map(String)])),
				selected: {},
				selectedPrice: "",
				subpages: {},
				...lastSearch,
				order: products.size + 1,
			};
			products.set(asin, rec);
		}
		rec.selected = { ...((cur.selected_options ?? {}) as Record<string, string>) };
		rec.selectedPrice = cur.selected_price === undefined || cur.selected_price === null ? "" : String(cur.selected_price);
		if (cur.page_type === "information_subpage") {
			// 和适配层渲染页面时一样：列表就把各项文字拼起来
			const raw = Array.isArray(cur.content)
				? cur.content.map((x: unknown) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n")
				: typeof cur.content === "string"
					? cur.content
					: cur.content
						? JSON.stringify(cur.content)
						: "";
			const text = raw.trim();
			rec.subpages[String(cur.subpage ?? "详情")] = !text ? "（空）" : squash(text) === squash(rec.title) ? "（和标题一样）" : short(text, 200);
		}
		lastOpened = asin;
		if (cur.page_type === "product_detail" && !hinted.has(asin)) freshAsin = asin;
	});

	pi.events.on("shop:ask", (data: any) => {
		const answer = String(data?.answer ?? "");
		const onProduct = cur?.page_type === "product_detail" || cur?.page_type === "information_subpage";
		dialogue.push({ question: String(data?.question ?? ""), answer, turn: turns, asin: onProduct ? String(cur.product?.asin ?? "") : lastOpened });
		if (BYE.test(answer)) shopperSaidBye = true;
		if (on("tables")) scheduleRequirements();
	});

	pi.on("context", async (event) => {
		if (!on("tables") || done) return undefined;
		await settle();
		const { messages, dropped } = windowed(event.messages as any[]);
		const notes = [askNote(), closeNote()].filter(Boolean);
		const card = renderCard();
		appendText(messages, notes.length ? `${card}\n——\n${notes.join("\n")}` : card);
		logTo("state_cards.jsonl", { turn: turns, sent: messages.length, dropped, card, notes });
		return { messages };
	});

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "buy_now") return undefined;
		const reason = await verifyPurchase();
		return reason ? { block: true, reason } : undefined;
	});
}

export { evidenceFrom, parseBudget, quotedFrom };
