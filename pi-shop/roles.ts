/**
 * 第四版 harness：前台导购 + 后台买手。和适配层的分工模式（SHOP_ROLES=1）一起加载，只用于论文多轮设置。
 *
 * 为什么这样分：v2 的几个插件补的是同一个洞——一个 agent 既要陪用户聊，又要自己在商店里翻商品，两头顾不上：
 * 在商店里忙完就直接下单，忘了回去问用户同不同意；对话一长，忘了前面看过的好商品；用户问一句页面上没写的信息，
 * 就一头扎进商店里找到回合用完。这里像真实门店一样分工：
 * - 导购（主 agent）只跟用户打交道，看不到商店页面。工具：find_products（派买手去找）、show_candidates（按最新的
 *   需求单把看过的商品重新排名，或者查一件商品页面上记下的全部内容，都不花回合）、buy（下单）。直接回复的文字由适配层转给用户。
 * - 买手每次 find_products 新开一个 pi 会话（同一个模型），只在商店里干活：搜索、翻页、打开商品、看规格，不下单、
 *   不和用户说话。每个商店动作先过 v2 的合法性校验（plugin.ts 的 guardCheck）；在商品页想打开搜索结果里的另一件时，
 *   系统先替它返回上一页；同一件商品的详情页看过一次就记下，不再重复看。
 * - 两边之间只传核对过的事实：需求单按用户原话整理（第三版的整理方法，原话在用户消息里找不到的条目丢掉）；
 *   候选单由系统把看过的所有商品按需求单排名，每条判成符合 / 不符合 / 页面没写，证据必须能在页面原文里找到，预算按规则判。
 * - 规矩只有一条：用户最后一句话明确同意买这件，导购才能下单。回合快用完（含环境的步数上限）或用户说了再见时，
 *   按官方规则兜底：买用户最倾向的那件，不再要求用户同意；提示过必须下单、导购还是只顾着说话时，系统替它下单。
 *   导购在还没到收尾时把对用户说的话和下单写在一条消息里（这种文字不会转给用户），被拦后又一直重复时，
 *   系统替它把这段话转给用户，对话接着往下走。
 *   下单由系统按候选单记下的位置回到商品页、点好规格（只有一个选项的规格自动点上）再购买。
 * - 连着两次找货候选单第 1 件都没变、它也没有和用户要求冲突的地方时，不再派买手去找（剩下的只是页面没写的信息）。
 * - 画像分级（--profile-tiers，画像设置下的改进）：整理需求单时一起看用户画像。最近的搜索词里和这次商品有关的写成要求、
 *   标明来自画像，排候选、推荐、派买手都照它对；平时的偏好（品牌、颜色等）只在几件商品差不多时参考，不当要求，买手也不只拿它去搜。
 *   和用户说的话冲突时听用户的；每条都要能在画像原文里找到。
 * 只用 agent 能看到的信息（对话、页面和画像设置下的用户画像），不读 gold 和模拟买家的配置。
 */
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	createAgentSession,
	createExtensionRuntime,
	defineTool,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type ActResult, renderPage, SHOP_TOOLS, type ShopApi } from "./adapter.ts";
import { guardCheck } from "./plugin.ts";
import {
	ATTITUDES,
	BASIC_SLOTS,
	chatJson,
	evidenceFrom,
	inBudget,
	looseKey,
	parseBudget,
	priceNumbers,
	priceText,
	quotedFrom,
	quoteValues,
	REQ_SYSTEM,
	short,
	SLOTS,
	squash,
	STATUSES,
} from "./task_state.ts";

const RUN_DIR = process.env.SHOP_RUN_DIR;
const MAX_STEPS = Number(process.env.SHOP_MAX_STEPS ?? "35");
const BUYER_PROMPT_FILE = process.env.SHOP_BUYER_PROMPT_FILE ?? "";
// 画像分级（画像设置下的改进，--profile-tiers）：画像里和这次商品有关的内容按来源写进需求单
const PROFILE_TIERS = process.env.SHOP_PROFILE_TIERS === "1";
const PERSONA_FILE = process.env.SHOP_PERSONA_FILE ?? "";
// 画像分级的消融（--profile-ablate）：flat 让平时的偏好也当要求，search_only 只用最近的搜索词
const PROFILE_ABLATE = process.env.SHOP_PROFILE_ABLATE ?? "";
// 每次找货买手最多做几个商店动作；派买手时给对话和下单留的回合
const BUYER_STEPS = 8;
const RESERVE = 6;
// 收尾阈值和 v2 一样：剩 12 个回合起提醒，剩 6 个（或环境步数只剩 6 步）或用户说了再见时必须下单
const CLOSING_SOFT = 12;
const CLOSING_HARD = 6;
const BYE = /再见|拜拜|bye/i;
const SHEET_SIZE = 3;
const POOL_LIMIT = 40;
// 连着几次内部调用（不对用户说话、也不派买手）之后，要求导购先对外做点什么
const MAX_INTERNAL = 3;
const BUYER_TOOLS = SHOP_TOOLS.filter((t) => t.name !== "buy_now");
const TEXT_ONLY = "要对用户说话，只回复文字、不要同时调用工具：和工具调用写在同一条消息里的文字不会发给用户。";

const SHEET_SYSTEM = `你是导购的助手，负责把买手看过的商品按用户的需求单排名。只能依据给出的商品信息（标题、店铺、类目、价格、属性、规格、详情页内容），不要凭常识或商品名推测页面上没写出来的信息；但页面上写了意思相同或更具体的说法，就算写了（比如用户要“便携”，页面写着“户外适用、迷你设计”；用户要“锁定位置”，页面写着“人体定位追踪”）。

从这些商品里挑出最符合需求单的 ${SHEET_SIZE} 件（不足 ${SHEET_SIZE} 件就全列），按符合程度从高到低排列，输出一个 JSON 对象：
{"top": [{"asin": "商品ID", "checks": [{"id": "R1", "verdict": "符合", "evidence": "商品信息里的原文片段"}], "options": [{"axis": "颜色分类", "wanted": ["01象牙白"]}]}]}

- checks：需求单里每条需求一项。verdict 只能是：符合、不符合、页面没写。判符合或不符合时，evidence 从这件商品的信息里逐字摘抄支持判断的一小段（预算就写价格）；页面上找不到相关的说法才判“页面没写”，evidence 留空。
  - 预算：按价格判断；价格是区间时，区间里有价格在预算内就算符合。
  - 规格类需求：可选规格里有用户要的选项就算符合；所有选项都不满足才算不符合；规格还没打开看过的商品判“页面没写”。
- options：只给打开看过规格的商品填，每个有两个以上选项的规格项一项。wanted 列出按需求应该选的选项，照抄选项原文：需求只对得上一个选项就只列这一个；几个都对得上就都列；用户没提过这方面就列出全部选项。`;

/**
 * 画像分级：整理完用户的需求表之后，再从画像里挑这次用得上的信息。最近的搜索词很可能就是这次要买的，当要求用；
 * 平时的偏好只做参考。用户说过具体要求的方面以用户为准；用户说无所谓、没想好的方面按画像补。
 */
const PROFILE_SYSTEM = `你负责从用户的个人资料里挑出这次购物用得上的信息。个人资料是平台记录的，不是用户在对话里说的话，分两部分：最近的搜索词，平时的偏好（人口属性、兴趣偏好、用户标签）。另外给你用户在对话里已经说过的要求（需求表）。

输出一个 JSON 对象：{"cues": [{"source": "画像搜索", "slot": "其他", "aspect": "款式", "value": "平底", "quote": "可搭配汉服的平底女式绣花鞋"}]}

- 画像搜索：最近的搜索词里和这次要买的商品有关的具体要求（功能、材质、款式、适用对象、使用场景、规格、数量、服务等），一条只写一个方面。搜索词说的是别的商品就不要写。
- 画像偏好：平时的偏好里和这次商品最相关的，最多 3 条（比如这一类商品的品牌、颜色、风格）。和这次商品无关的不要写。
- 需求表里用户已经给了具体要求的方面：画像和用户说的不一样就不要写（以用户为准），一样的也不用重复写。用户说无所谓或者没想好的方面，画像里有就写。
- slot 只能是：品类、品牌、型号款式、规格、预算、其他；aspect 两到四个字；value 尽量用资料里的说法；quote 从个人资料里逐字摘抄支持这一条的一小段。`;

const SHEET_PREF_NOTE = `
- 需求单后面附的“用户平时的偏好”不是需求：只在几件商品其他方面差不多时用来排先后，不用逐条判，也不要因为不符合偏好就把更符合需求单的商品排到后面。`;

const MAX_PREF_CUES = 3;

/** 平时的偏好只做参考、不当要求；消融 flat 时和最近的搜索一样当要求。 */
const isReference = (r: { source: string }) => r.source === "画像偏好" && PROFILE_ABLATE !== "flat";

type Verdict = "符合" | "不符合" | "页面没写";

interface Req {
	id: string;
	slot: string;
	aspect: string;
	value: string;
	status: string;
	quote: string;
	/** 用户 / 画像搜索 / 画像偏好；不开画像分级时都是“用户” */
	source: string;
}

interface Item {
	asin: string;
	title: string;
	brand: string;
	category: string;
	price: string;
	prices: number[];
	attrs: string[];
	/** 打开过商品页才知道规格；没打开过是 null */
	options: Record<string, string[]> | null;
	selected: Record<string, string>;
	subpages: Record<string, string>;
	/** 下单时回到这件商品用：在哪个搜索词的第几页见过（取页码最小的那次） */
	query: string;
	page: number;
	opened: boolean;
	seen: number;
}

interface Pick {
	item: Item;
	checks: Map<string, { verdict: Verdict; evidence: string }>;
	wanted: Record<string, string[]>;
}

function log(file: string, record: Record<string, unknown>): void {
	if (!RUN_DIR) return;
	mkdirSync(RUN_DIR, { recursive: true });
	appendFileSync(join(RUN_DIR, file), `${JSON.stringify({ t: new Date().toISOString(), ...record })}\n`);
}

const logEvent = (record: Record<string, unknown>) => log("harness_events.jsonl", record);

const reply = (text: string) => ({ content: [{ type: "text" as const, text }], details: {} });

function textOf(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((c: any) => c?.type === "text")
		.map((c: any) => String(c.text ?? ""))
		.join("\n")
		.trim();
}

function parseInitialState(): any {
	try {
		const state = JSON.parse(process.env.SHOP_INITIAL_STATE || "{}");
		if (state && typeof state === "object" && state.page_type) return state;
	} catch {
		// 用默认的搜索首页
	}
	return { page_type: "search_home", actions: ["search"] };
}

interface Profile {
	/** 最近的行为：搜索词，以及其他有内容的“最近……”栏（官方数据里加购、收藏全是空的） */
	recent: string[];
	/** 平时的偏好：人口属性、兴趣偏好、用户标签，原样给小模型看 */
	prefs: string;
}

function loadProfile(): Profile | null {
	if (!PROFILE_TIERS || !PERSONA_FILE) return null;
	const p = JSON.parse(readFileSync(PERSONA_FILE, "utf-8"));
	const recent: string[] = [];
	for (const [key, value] of Object.entries((p?.["行为特征"] ?? {}) as Record<string, unknown>)) {
		if (!key.includes("最近") || !Array.isArray(value)) continue;
		for (const v of value) {
			const text = (typeof v === "string" ? v : JSON.stringify(v)).trim();
			if (text) recent.push(key.includes("搜索") ? text : `${key.replace(/^最近\d+天/, "")}：${text}`);
		}
	}
	const prefs = JSON.stringify({ 人口属性: p?.["人口属性"] ?? {}, 兴趣偏好: p?.["兴趣偏好"] ?? {}, 用户标签: p?.["用户标签"] ?? [] });
	return { recent, prefs };
}

/** 买手会话不发现任何扩展、技能和上下文文件，只用给定的系统提示和工具。 */
function minimalLoader(system: string): any {
	return {
		getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
		getSkills: () => ({ skills: [], diagnostics: [] }),
		getPrompts: () => ({ prompts: [], diagnostics: [] }),
		getThemes: () => ({ themes: [], diagnostics: [] }),
		getAgentsFiles: () => ({ agentsFiles: [] }),
		getSystemPrompt: () => system,
		getSystemPromptSource: () => undefined,
		getAppendSystemPrompt: () => [],
		getAppendSystemPromptSources: () => [],
		extendResources: () => {},
		reload: async () => {},
	};
}

/** 导购写的规格值对回选项原文：完全一致优先，否则取唯一一个互相包含的。 */
function matchOption(axes: Record<string, string[]>, wanted: string): [string, string] | null {
	const w = looseKey(wanted);
	if (!w) return null;
	const exact: [string, string][] = [];
	const near: [string, string][] = [];
	for (const [axis, values] of Object.entries(axes)) {
		for (const v of values.map(String)) {
			const k = looseKey(v);
			if (k === w) exact.push([axis, v]);
			else if (k.includes(w) || w.includes(k)) near.push([axis, v]);
		}
	}
	if (exact.length) return exact[0];
	return near.length === 1 ? near[0] : null;
}

export default function (pi: ExtensionAPI) {
	let handle: ShopApi | undefined;
	pi.events.emit("shop:api", (api: ShopApi) => {
		handle = api;
	});
	if (!handle) throw new Error("roles.ts 要和分工模式的适配层一起加载（SHOP_ROLES=1，先加载 adapter.ts）");
	const shop: ShopApi = handle;

	let cur: any = parseInitialState();
	let opening = "";
	const dialogue: { question: string; answer: string; turn: number }[] = [];
	let reqs: Req[] = [];
	// 最近一次整理需求单时，用户最后一句话的购买态度；n 是当时的对话条数
	let attitude = { status: "未表态", product: "", quote: "", n: 0 };
	let userSaidBye = false;
	let reqJob: Promise<void> = Promise.resolve();
	const pool = new Map<string, Item>();
	const searches = new Map<string, number>();
	let seenCounter = 0;
	let lastSheet: Pick[] = [];
	let internalCalls = 0;
	let lastExternalTurn = -1;
	let runtime: ModelRuntime | null = null;
	let buyerPrompt = "";
	// 连着几次找货候选单第 1 件都没变；最近一次排候选单时的需求单
	let staleFinds = 0;
	let lastTop = "";
	let sheetKey = "";
	// 提示过必须下单之后，导购又对用户说话的次数；系统替它下过单没有
	let closeIgnored = 0;
	let closeAnnounced = false;
	let autoTried = false;
	// 导购最近一条消息里写的文字：和工具调用写在一起时不会转给用户，卡住时由系统替它转
	let lastAssistantText = "";
	let gateWarned = false;
	const profile = loadProfile();
	const recentText = profile ? profile.recent.join("\n") : "";
	if (profile) logEvent({ type: "roles_profile", recent: profile.recent.length });

	const turnsLeft = () => shop.maxTurns - shop.turn();
	const stepsLeft = () => MAX_STEPS - shop.steps();
	const mustClose = () => !shop.isDone() && (userSaidBye || turnsLeft() <= CLOSING_HARD || stepsLeft() <= CLOSING_HARD);

	// ---- 看过的商品 ----

	function itemFrom(p: any, where: { query: string; page: number }): Item {
		return {
			asin: String(p.asin ?? ""),
			title: String(p.title ?? ""),
			brand: String(p.brand ?? ""),
			category: String(p.category ?? ""),
			price: priceText(p.price),
			prices: priceNumbers(p.price),
			attrs: Array.isArray(p.key_attributes) ? p.key_attributes.map(String) : [],
			options: null,
			selected: {},
			subpages: {},
			query: where.query,
			page: where.page,
			opened: false,
			seen: 0,
		};
	}

	let lastSearch = { query: "", page: 1 };
	// 最近一页搜索结果里的商品：买手在商品页想打开其中另一件时，系统先替它返回搜索结果
	let lastResults = new Set<string>();

	function record(st: any): void {
		const page = st?.page_type;
		if (page === "search_results") {
			lastSearch = { query: String(st.query ?? ""), page: Number(st.page ?? 1) };
			lastResults = new Set((st.products ?? []).map((p: any) => String(p.asin ?? "")));
			searches.set(lastSearch.query, Math.max(searches.get(lastSearch.query) ?? 0, lastSearch.page));
			for (const p of st.products ?? []) {
				const asin = String(p.asin ?? "");
				if (!asin) continue;
				let it = pool.get(asin);
				if (!it) {
					it = itemFrom(p, lastSearch);
					pool.set(asin, it);
				} else if (lastSearch.page < it.page) {
					it.query = lastSearch.query;
					it.page = lastSearch.page;
				}
				it.seen = ++seenCounter;
			}
			return;
		}
		if (page !== "product_detail" && page !== "information_subpage") return;
		const p = st.product ?? {};
		const asin = String(p.asin ?? "");
		if (!asin) return;
		let it = pool.get(asin);
		if (!it) {
			it = itemFrom(p, lastSearch);
			pool.set(asin, it);
		}
		it.opened = true;
		it.title = String(p.title ?? it.title);
		it.price = priceText(p.price);
		it.prices = priceNumbers(p.price);
		if (Array.isArray(p.key_attributes)) it.attrs = p.key_attributes.map(String);
		it.options = Object.fromEntries(
			Object.entries((st.available_options ?? {}) as Record<string, unknown[]>).map(([k, vs]) => [k, (vs ?? []).map(String)]),
		);
		it.selected = { ...((st.selected_options ?? {}) as Record<string, string>) };
		if (page === "information_subpage") {
			// 和适配层渲染页面时一样：列表就把各项文字拼起来
			const raw = Array.isArray(st.content)
				? st.content.map((x: unknown) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n")
				: typeof st.content === "string"
					? st.content
					: st.content
						? JSON.stringify(st.content)
						: "";
			const text = raw.trim();
			it.subpages[String(st.subpage ?? "详情")] = !text ? "（空）" : squash(text) === squash(it.title) ? "（和标题一样）" : short(text, 400);
		}
		it.seen = ++seenCounter;
	}

	function factsOf(it: Item): string {
		const parts = [`标题：${it.title}`, `店铺：${it.brand}`, `类目：${it.category}`, `价格：${it.price}`, `属性：${it.attrs.length ? it.attrs.join("；") : "无"}`];
		parts.push(it.options ? `可选规格：${Object.keys(it.options).length ? JSON.stringify(it.options) : "无"}` : "可选规格：没打开看过");
		for (const [name, text] of Object.entries(it.subpages)) if (!text.startsWith("（")) parts.push(`${name}：${text}`);
		return parts.join("｜");
	}

	const SUBPAGES = ["Description", "Features", "Reviews"];

	/** 三个详情页（描述、卖点、评价）哪些看过、看到了什么。 */
	function subpageNote(it: Item): string {
		if (!it.opened) return "没打开过";
		const seen = SUBPAGES.filter((n) => n in it.subpages).map((n) => `${n}${it.subpages[n].startsWith("（") ? it.subpages[n] : "（有内容）"}`);
		const unseen = SUBPAGES.filter((n) => !(n in it.subpages));
		if (!unseen.length) return `都看过了：${seen.join("、")}`;
		return `${seen.length ? `看过 ${seen.join("、")}；` : ""}没看 ${unseen.join("、")}`;
	}

	/** 一件商品页面上记下的全部内容，给导购查细节用。 */
	function itemCard(it: Item): string {
		const L = [`【商品 [${it.asin}]】页面上记下的全部内容（只认这些字）：`, `标题：${it.title}`, `店铺：${it.brand}｜类目：${it.category}`, `价格：${it.price}`];
		L.push(`属性：${it.attrs.length ? it.attrs.join("；") : "无"}`);
		if (!it.options) L.push("规格：还没打开看过");
		else if (!Object.keys(it.options).length) L.push("规格：无");
		else for (const [axis, values] of Object.entries(it.options)) L.push(`规格「${axis}」：${quoteValues(values, 20)}`);
		for (const n of SUBPAGES) if (n in it.subpages) L.push(`${n}：${it.subpages[n]}`);
		L.push(`详情页：${subpageNote(it)}`);
		return L.join("\n");
	}

	// ---- 需求单：第三版的整理方法，用户每说一次话重新整理 ----

	async function extractRequirements(): Promise<void> {
		const snapshot = dialogue.slice();
		const lines = [`用户：${opening}`];
		for (const d of snapshot) lines.push(`助手：${short(d.question, 600)}`, `用户：${d.answer}`);
		const opened = [...pool.values()].filter((i) => i.opened).map((i) => `[${i.asin}] ${short(i.title, 40)}`);
		const input = `看过的商品：\n${opened.length ? opened.join("\n") : "（还没有）"}\n\n对话（按时间顺序）：\n${lines.join("\n")}`;
		const out = await chatJson("requirements", REQ_SYSTEM, input, 1500, shop.turn());
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
			const row: Req = { id: "", slot, aspect, value: status === "无要求" ? "" : value, status, quote, source: "用户" };
			const key = `${slot}|${aspect}|${row.value}`;
			if (keys.has(key)) continue;
			keys.add(key);
			rows.push(row);
		}
		for (const row of await profileCues(rows)) {
			const key = `${row.slot}|${row.aspect}|${row.value}`;
			if (keys.has(key)) continue;
			keys.add(key);
			rows.push(row);
		}
		rows.forEach((r, i) => (r.id = `R${i + 1}`));
		reqs = rows;
		const att = out.attitude ?? {};
		const status = String(att.status ?? "");
		const valid = snapshot.length > 0 && ATTITUDES.includes(status) && quotedFrom(String(att.quote ?? ""), lastAnswer);
		attitude = { status: valid ? status : "未表态", product: valid ? String(att.product ?? "").trim() : "", quote: valid ? String(att.quote) : "", n: snapshot.length };
		if (dropped.length) log("state_log.jsonl", { kind: "requirements_dropped", turn: shop.turn(), dropped });
	}

	/** 画像分级：按用户已经说过的要求，从画像里挑这次用得上的条目；每条都要能在画像原文里找到。 */
	async function profileCues(userRows: Req[]): Promise<Req[]> {
		if (!profile) return [];
		const table = userRows.length ? userRows.map((r) => `${r.aspect}：${reqValue(r)}`).join("\n") : "（还没有）";
		const recent = profile.recent.length ? profile.recent.map((s) => `「${s}」`).join("") : "（没有）";
		const input = `用户在对话里说过的要求（需求表）：\n${table}\n\n用户的个人资料：\n最近的搜索词：${recent}\n平时的偏好：${profile.prefs}`;
		const out = await chatJson("profile", PROFILE_SYSTEM, input, 1500, shop.turn());
		const found: Req[] = [];
		const dropped: unknown[] = [];
		for (const raw of Array.isArray(out?.cues) ? out.cues : []) {
			const source = String(raw?.source ?? "");
			const quote = String(raw?.quote ?? "").trim();
			const value = String(raw?.value ?? "").trim();
			const from = source === "画像搜索" ? recentText : source === "画像偏好" ? profile.prefs : "";
			if (!from || !value || !quotedFrom(quote, from)) {
				dropped.push(raw);
				continue;
			}
			const slot = SLOTS.includes(String(raw?.slot)) ? String(raw.slot) : "其他";
			found.push({ id: "", slot, aspect: short(String(raw?.aspect ?? "").trim() || slot, 8), value, status: "已确认", quote, source });
		}
		if (dropped.length) log("state_log.jsonl", { kind: "profile_dropped", turn: shop.turn(), dropped });
		const prefs = PROFILE_ABLATE === "search_only" ? [] : found.filter((r) => r.source === "画像偏好").slice(0, MAX_PREF_CUES);
		return [...found.filter((r) => r.source === "画像搜索"), ...prefs];
	}

	function scheduleRequirements(): void {
		reqJob = reqJob.then(extractRequirements).catch((err) => {
			log("state_log.jsonl", { kind: "requirements_error", turn: shop.turn(), error: String(err) });
		});
	}

	const unknownSlots = () => BASIC_SLOTS.filter((s) => !reqs.some((r) => r.slot === s));
	const reqKey = () =>
		reqs
			.filter((r) => r.status === "已确认")
			.map((r) => `${r.aspect}|${r.value}`)
			.sort()
			.join("\n");

	function reqValue(r: Req): string {
		if (r.status === "无要求") return "用户说无所谓";
		if (r.status === "待补充") return `用户说有要求、还没说具体（${r.value || r.aspect}）`;
		return r.value;
	}

	// ---- 候选单 ----

	function judgedPool(): Item[] {
		const items = [...pool.values()];
		const opened = items.filter((i) => i.opened).sort((a, b) => b.seen - a.seen);
		const rest = items.filter((i) => !i.opened).sort((a, b) => b.seen - a.seen);
		return [...opened, ...rest].slice(0, POOL_LIMIT);
	}

	async function buildSheet(reason: string): Promise<Pick[]> {
		await reqJob;
		sheetKey = reqKey();
		const items = judgedPool();
		if (!items.length) {
			lastSheet = [];
			return lastSheet;
		}
		const confirmed = reqs.filter((r) => r.status === "已确认" && !isReference(r));
		const relaxed = reqs.filter((r) => r.status === "无要求");
		const prefs = reqs.filter((r) => r.status === "已确认" && isReference(r));
		const from = (r: Req) => (r.source === "画像搜索" ? "（来自用户最近的搜索）" : r.source === "画像偏好" ? "（来自用户平时的偏好）" : "");
		const listing = confirmed.map((r) => `${r.id} ${r.aspect}：${r.value}${from(r)}`);
		if (relaxed.length) listing.push(`（用户说无所谓的方面：${relaxed.map((r) => r.aspect).join("、")}）`);
		const prefLine = prefs.length ? `\n\n用户平时的偏好（不是需求，只在几件商品其他方面差不多时参考）：${prefs.map((r) => `${r.aspect}：${r.value}`).join("；")}` : "";
		const byAsin = new Map(items.map((i) => [i.asin, i]));
		const input =
			`用户的需求单：\n${listing.join("\n") || "（还没有确定的需求）"}${prefLine}\n\n` +
			`看过的商品（共 ${items.length} 件）：\n${items.map((i) => `[${i.asin}] ${factsOf(i)}`).join("\n")}`;
		const out = await chatJson("sheet", prefs.length ? SHEET_SYSTEM + SHEET_PREF_NOTE : SHEET_SYSTEM, input, 2000, shop.turn());
		const picks: Pick[] = [];
		const downgraded: unknown[] = [];
		for (const raw of Array.isArray(out?.top) ? out.top : []) {
			const it = byAsin.get(String(raw?.asin ?? ""));
			if (!it || picks.some((p) => p.item === it)) continue;
			const facts = factsOf(it);
			const checks = new Map<string, { verdict: Verdict; evidence: string }>();
			for (const c of Array.isArray(raw.checks) ? raw.checks : []) {
				const row = confirmed.find((r) => r.id === String(c?.id ?? ""));
				let verdict = String(c?.verdict ?? "");
				const evidence = String(c?.evidence ?? "").trim();
				if (!row || !["符合", "不符合", "页面没写"].includes(verdict)) continue;
				if (verdict !== "页面没写" && !evidenceFrom(evidence, facts)) {
					downgraded.push({ asin: it.asin, ...c });
					verdict = "页面没写";
				}
				checks.set(row.id, { verdict: verdict as Verdict, evidence: verdict === "页面没写" ? "" : evidence });
			}
			// 预算能换成价格区间时按规则判，不用小模型
			for (const row of confirmed.filter((r) => r.slot === "预算")) {
				const budget = parseBudget(row.value);
				if (!budget || !it.prices.length) continue;
				checks.set(row.id, { verdict: inBudget(it.prices, budget) ? "符合" : "不符合", evidence: `价格 ${it.price}` });
			}
			const wanted: Record<string, string[]> = {};
			if (it.options) {
				for (const o of Array.isArray(raw.options) ? raw.options : []) {
					const axis = Object.keys(it.options).find((a) => looseKey(a) === looseKey(o?.axis));
					if (!axis) continue;
					const values = it.options[axis];
					const hits = (Array.isArray(o.wanted) ? o.wanted : [])
						.map((w: unknown) => values.find((v) => looseKey(v) === looseKey(w)))
						.filter((v: string | undefined): v is string => Boolean(v));
					wanted[axis] = hits.length ? [...new Set<string>(hits)] : values;
				}
			}
			picks.push({ item: it, checks, wanted });
			if (picks.length >= SHEET_SIZE) break;
		}
		// 排名模型调用失败时，退回最近打开过的几件
		if (!picks.length) for (const it of items.slice(0, SHEET_SIZE)) picks.push({ item: it, checks: new Map(), wanted: {} });
		lastSheet = picks;
		const card = sheetText(picks);
		log("state_cards.jsonl", { turn: shop.turn(), reason, card });
		logEvent({ type: "roles_sheet", turn: shop.turn(), reason, pool: pool.size, top: picks.map((p) => p.item.asin) });
		if (downgraded.length) log("state_log.jsonl", { kind: "sheet_downgraded", turn: shop.turn(), downgraded });
		return picks;
	}

	function specLines(p: Pick): string[] {
		const it = p.item;
		if (!it.options) return ["     规格：还没打开看过（需要的话让买手打开这件）"];
		if (!Object.keys(it.options).length) return ["     规格：无"];
		return Object.entries(it.options).map(([axis, values]) => {
			const w = p.wanted[axis] ?? values;
			const want =
				values.length === 1
					? `只有“${values[0]}”一个选项（下单时自动点上）`
					: w.length === 1
						? `用户要“${w[0]}”`
						: `用户还没定（可选 ${quoteValues(w, 8)}）`;
			return `     规格「${axis}」：${want}`;
		});
	}

	/** 需求表一行的来源：用户原话，或者画像里的哪一部分。 */
	function sourceNote(r: Req): string {
		if (r.source === "画像搜索") return `画像：用户最近搜过「${short(r.quote, 40)}」（用户没亲口说过）`;
		if (r.source === "画像偏好") return `画像：平时的偏好「${short(r.quote, 30)}」${isReference(r) ? "（只做参考，不是要求）" : "（用户没亲口说过）"}`;
		return `原话「${short(r.quote, 40)}」`;
	}

	/** 候选单的文字。需求表和商品行的格式和第三版的状态卡一致，评测器第 8 个面板可以直接读。 */
	function sheetText(picks: Pick[]): string {
		const L = [`【候选单】系统把看过的 ${pool.size} 件商品按需求单排了名（只认页面上写着的字）。`, "需求表："];
		if (!reqs.length) L.push("  （还没有）");
		for (const r of reqs) L.push(`  ${r.id} ${r.aspect}：${reqValue(r)}｜${sourceNote(r)}`);
		const unknown = unknownSlots();
		if (unknown.length) L.push(`  还不知道：${unknown.join("、")}`);
		if (!picks.length) {
			L.push("候选：（还没看过商品）");
			return L.join("\n");
		}
		const confirmed = reqs.filter((r) => r.status === "已确认" && !isReference(r));
		L.push("候选（按符合程度排序）：");
		picks.forEach((p, i) => {
			const it = p.item;
			L.push(`  ${i + 1}. [${it.asin}] ${short(it.title, 40)}｜价格 ${it.price}｜店铺 ${short(it.brand, 16)}｜在搜索“${short(it.query, 20)}”第 ${it.page} 页`);
			const by = (v: Verdict) => confirmed.filter((r) => (p.checks.get(r.id)?.verdict ?? "页面没写") === v);
			const bad = by("不符合").map((r) => `${r.id}（页面：${short(p.checks.get(r.id)?.evidence, 20)}）`);
			const ids = (rows: Req[]) => (rows.length ? rows.map((r) => r.id).join(" ") : "无");
			if (confirmed.length) L.push(`     符合 ${ids(by("符合"))}｜不符合 ${bad.join(" ") || "无"}｜页面没写 ${ids(by("页面没写"))}`);
			L.push(...specLines(p));
			L.push(`     详情页：${subpageNote(it)}`);
		});
		const nowhere = confirmed.filter((r) => pool.size >= 5 && picks.every((p) => (p.checks.get(r.id)?.verdict ?? "页面没写") === "页面没写"));
		if (nowhere.length) {
			L.push(`候选里每件的页面上都没写：${nowhere.map((r) => `${r.id} ${r.aspect}`).join("、")}（多半查不到，不用再让买手去找；用户问到时如实说明）`);
		}
		return L.join("\n");
	}

	// ---- 收尾 ----

	function closingNote(): string {
		if (shop.isDone()) return "";
		const left = Math.min(turnsLeft(), stepsLeft());
		if (mustClose()) {
			const why = userSaidBye ? "用户已经结束对话" : `只剩 ${left} 个回合`;
			// 规则原文和 v2 的收尾一样，来自论文官方 agent 提示
			const rule =
				"按这个任务的规则，回合用完前必须买下一件商品，否则任务失败；对话结束时用户仍不满意的，就买最接近用户需求的那一件，不需要用户同意（这是任务要求，不算违背用户）。";
			logEvent({ type: "roles_close_hard", turn: shop.turn(), left, bye: userSaidBye, candidates: lastSheet.length });
			closeAnnounced = true;
			if (!lastSheet.length) return `【现在必须下单】${why}。${rule}还没有候选：马上派买手找一次（只找一两件），再调用 buy 下单。`;
			const list = lastSheet.map((p, i) => {
				const it = p.item;
				const specs = it.options
					? Object.entries(it.options)
							.filter(([axis, values]) => values.length > 1 && (p.wanted[axis] ?? values).length === 1)
							.map(([axis]) => `${axis}=“${p.wanted[axis][0]}”`)
					: [];
				return `${i + 1}. [${it.asin}] ${short(it.title, 30)}｜价格 ${it.price}${specs.length ? `｜用户要的规格：${specs.join("，")}` : ""}`;
			});
			return (
				`【现在必须下单】${why}。${rule}不要再问用户，也不要再派买手去找：从候选单里选用户最倾向的那件（没有就选第 1 件），` +
				`直接调用 buy，规格写用户要的那个（用户没定的选最接近需求的）。\n${list.join("\n")}`
			);
		}
		if (left <= CLOSING_SOFT) {
			logEvent({ type: "roles_close_soft", turn: shop.turn(), left });
			return `【还剩 ${left} 个回合】对用户说的话和买手的每个动作都算回合，回合用完还没下单这一单就算失败。请尽快请用户确认并下单。`;
		}
		return "";
	}

	const withNote = (text: string) => {
		const note = closingNote();
		return note ? `${text}\n——\n${note}` : text;
	};

	/**
	 * 导购连着在内部打转（看候选单、被拦下的下单）时，要求它先对用户说话或派买手。这个提醒只给一次：之后再打转、
	 * 而它在这条消息里写了要对用户说的话（和工具调用写在一起，本来不会发出去），系统就替它转给用户，对话接着往下走。
	 */
	async function internalGate(): Promise<string | null> {
		if (shop.turn() !== lastExternalTurn) {
			lastExternalTurn = shop.turn();
			internalCalls = 0;
		}
		internalCalls += 1;
		if (internalCalls <= MAX_INTERNAL) return null;
		const text = lastAssistantText.trim();
		if (gateWarned && text && !shop.isDone() && shop.turn() < shop.maxTurns) {
			logEvent({ type: "roles_relay_text", turn: shop.turn(), calls: internalCalls, text: short(text, 200) });
			const answer = await shop.say(text);
			return withNote(
				shop.withTurn(`（你刚才和工具调用写在一起的那段话，系统已经替你转给了用户。以后要对用户说话，只回复文字，不要同时调用工具。）\n用户回答：${answer}`),
			);
		}
		gateWarned = true;
		logEvent({ type: "roles_internal_block", turn: shop.turn(), calls: internalCalls });
		return `连着几次都没有对用户说话、也没有派买手。请先对用户说话，或者派买手去找。${TEXT_ONLY}`;
	}

	// ---- 买手 ----

	function buyerTask(request: string, budget: number): string {
		const L = [`找货任务（导购交给你的）：${request}`, "", profile ? "用户的需求单（系统按用户原话和用户画像整理）：" : "用户的需求单（系统按用户原话整理）："];
		if (!reqs.length) L.push("  （还没有）");
		const tag = (r: Req) =>
			r.source === "画像搜索" ? "（来自用户最近的搜索）" : r.source !== "画像偏好" ? "" : isReference(r) ? "（用户平时的偏好，只做参考）" : "（来自用户平时的偏好）";
		for (const r of reqs) L.push(`  ${r.id} ${r.aspect}：${reqValue(r)}${tag(r)}`);
		const unknown = unknownSlots();
		if (unknown.length) L.push(`  还不知道：${unknown.join("、")}`);
		if (profile) {
			// 用户自己最近搜过的词往往就是最准的搜索词
			const used = profile.recent.filter((s) => reqs.some((r) => r.source === "画像搜索" && quotedFrom(r.quote, s)));
			if (used.length) L.push(`用户最近在商城里搜过：${used.map((s) => `「${s}」`).join("")}（和这次要买的有关，可以直接拿来当搜索词）`);
			if (reqs.some(isReference)) {
				L.push("用户平时的偏好不是要求：不要只拿偏好里的品牌去搜，也不要因为不是这个牌子就放过更符合需求的商品。");
			}
		}
		if (searches.size) L.push("", `已经搜过：${[...searches.entries()].map(([q, p]) => `「${q}」（看到第 ${p} 页）`).join("；")}`);
		const recent = judgedPool().slice(0, 12);
		if (recent.length) {
			L.push("已经看过的商品：");
			for (const it of recent) {
				const subs = Object.entries(it.subpages).map(([name, text]) => `${name}${text.startsWith("（") ? text : `：${short(text, 30)}`}`);
				const where = it.opened ? `打开过${subs.length ? `，看过的详情页：${subs.join("、")}` : ""}` : "只在搜索结果里见过";
				L.push(`  [${it.asin}] ${short(it.title, 36)}｜价格 ${it.price}｜${where}`);
			}
		}
		L.push("", `当前页面：\n${renderPage(cur)}`);
		L.push("", `这次最多做 ${budget} 个动作（每个动作都算用户那边的一个回合），够用就提前停下。做完用一两句话说明找到了什么、还缺什么。`);
		return L.join("\n");
	}

	async function runBuyer(request: string, budget: number, mainModel: any): Promise<string> {
		if (!buyerPrompt) buyerPrompt = BUYER_PROMPT_FILE ? readFileSync(BUYER_PROMPT_FILE, "utf-8").trim() : "你是网店的买手，在商店里替导购找商品。";
		const agentDir = getAgentDir();
		runtime ??= await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
		// 买手和导购用同一个模型：优先取导购当前的模型，取不到时用 runner 传来的 provider/model
		const [provider, ...rest] = String(process.env.SHOP_BUYER_MODEL ?? "").split("/");
		const ref = mainModel?.provider && mainModel?.id ? { provider: mainModel.provider, id: mainModel.id } : { provider, id: rest.join("/") };
		const model = runtime.getModel(ref.provider, ref.id);
		if (!model) throw new Error(`找不到买手用的模型 ${ref.provider}/${ref.id}`);
		let used = 0;
		let perMessage = 0;
		let over = 0;
		let calls = 0;
		const say = (text: string, terminate = false) => ({ content: [{ type: "text" as const, text }], details: {}, ...(terminate ? { terminate: true } : {}) });
		const SUBPAGE: Record<string, string> = { view_description: "Description", view_features: "Features", view_reviews: "Reviews" };
		// 在商品页或详情子页时：打开最近一页搜索结果里的另一件，要先返回搜索结果；看别的详情页、选规格，要先从子页回到商品页
		const backSteps = (tool: string, params: Record<string, string>): number => {
			const page = cur?.page_type;
			if (tool === "open_product") {
				const asin = String(params.asin ?? "");
				const listed = lastResults.has(asin) || [...lastResults].some((a) => looseKey(a) === looseKey(asin));
				if (!listed) return 0;
				return page === "information_subpage" ? 2 : page === "product_detail" ? 1 : 0;
			}
			return page === "information_subpage" && (tool in SUBPAGE || tool === "select_option") ? 1 : 0;
		};
		const NAV_NOTE: Record<string, string> = {
			open_product: "（在商品页或详情子页时，系统会先替你返回搜索结果，再打开）",
			select_option: "（在详情子页时，系统会先替你回到商品页）",
			view_description: "（在详情子页时，系统会先替你回到商品页）",
			view_features: "（在详情子页时，系统会先替你回到商品页）",
			view_reviews: "（在详情子页时，系统会先替你回到商品页）",
		};
		const tools = BUYER_TOOLS.map((t) =>
			defineTool({
				name: t.name,
				label: t.label,
				description: t.description + (NAV_NOTE[t.name] ?? ""),
				parameters: t.parameters,
				executionMode: "sequential",
				execute: async (_id: string, raw: any) => {
					const params: Record<string, string> = { ...(raw ?? {}) };
					perMessage += 1;
					if (perMessage > 1) return say("每次只能做一个动作，这条没有执行。请先看上一个动作返回的页面，再决定下一步。");
					if (used >= budget || shop.isDone() || shop.turn() >= shop.maxTurns) {
						over += 1;
						return say("这次找货的动作用完了，请停下，用一两句话说明找到了什么、还缺什么。", over > 1);
					}
					used += 1;
					// 同一件商品的详情页看过一次就记下了（内容不会变），再看只是白花回合
					const sub = SUBPAGE[t.name];
					const onItem = cur?.page_type === "product_detail" || cur?.page_type === "information_subpage";
					const seenText = sub && onItem ? pool.get(String(cur.product?.asin ?? ""))?.subpages[sub] : undefined;
					if (seenText !== undefined) {
						const turn = shop.countBlocked();
						const reason = `这件商品的 ${sub} 页刚才看过了，内容：${seenText}。不用再看。`;
						logEvent({ type: "guard_block", turn, tool: t.name, args: params, page: cur?.page_type, reason, who: "买手" });
						return say(reason);
					}
					const backs = backSteps(t.name, params);
					for (let i = 0; i < backs; i++) {
						logEvent({ type: "roles_auto_back", turn: shop.turn(), tool: t.name, args: params, page: cur?.page_type });
						const r = await shop.act("prev_page", {}, "买手");
						used += 1;
						if (r.refused || r.kind !== "ok") return say(r.text, true);
					}
					const verdict = guardCheck(cur, t.name, params);
					if (verdict.block) {
						const turn = shop.countBlocked();
						logEvent({ type: "guard_block", turn, tool: t.name, args: params, page: cur?.page_type, reason: verdict.block, who: "买手" });
						return say(verdict.block);
					}
					if (verdict.rewrite) {
						logEvent({ type: "guard_rewrite", turn: shop.turn(), tool: t.name, from: params[verdict.rewrite.field], to: verdict.rewrite.value, who: "买手" });
						params[verdict.rewrite.field] = verdict.rewrite.value;
					}
					const r = await shop.act(t.name, params, "买手");
					if (r.refused || r.kind !== "ok") return say(r.text, true);
					return say(`${r.text}\n（这次找货还能做 ${budget - used} 个动作）`);
				},
			}),
		);
		const { session } = await createAgentSession({
			cwd: process.cwd(),
			agentDir,
			model,
			thinkingLevel: "off",
			modelRuntime: runtime,
			resourceLoader: minimalLoader(buyerPrompt),
			tools: tools.map((t) => t.name),
			customTools: tools,
			sessionManager: RUN_DIR ? SessionManager.create(process.cwd(), join(RUN_DIR, "buyer_sessions")) : SessionManager.inMemory(),
			settingsManager: SettingsManager.inMemory({
				compaction: { enabled: false },
				retry: { enabled: true, maxRetries: 3, baseDelayMs: 2000, provider: { timeoutMs: 120000, maxRetries: 0 } },
			}),
		});
		const unsubscribe = session.subscribe((event: any) => {
			if (event?.type !== "turn_start") return;
			perMessage = 0;
			calls += 1;
			// 买手不停地调用（比如一直在说话或重复被拦的动作）时强行收工
			if (calls > budget + 4) void session.abort();
		});
		try {
			await session.prompt(buyerTask(request, budget));
		} catch (err) {
			logEvent({ type: "roles_buyer_error", turn: shop.turn(), error: String(err) });
		} finally {
			unsubscribe();
		}
		const last = [...session.messages].reverse().find((m: any) => m?.role === "assistant" && textOf(m.content)) as any;
		session.dispose();
		return last ? short(textOf(last.content), 300) : "";
	}

	// ---- 下单：回到商品页、点好规格、购买 ----

	async function step(name: string, params: Record<string, string>): Promise<ActResult | null> {
		const verdict = guardCheck(cur, name, params);
		if (verdict.block) {
			logEvent({ type: "roles_buy_step_blocked", turn: shop.turn(), tool: name, args: params, reason: verdict.block });
			return null;
		}
		if (verdict.rewrite) params[verdict.rewrite.field] = verdict.rewrite.value;
		const r = await shop.act(name, params, "下单");
		return r.refused || r.kind === "error" ? null : r;
	}

	const onResults = (asin: string) => cur?.page_type === "search_results" && (cur.products ?? []).some((p: any) => String(p.asin) === asin);
	const atProduct = (asin: string) => cur?.page_type === "product_detail" && String(cur.product?.asin ?? "") === asin;

	async function gotoProduct(it: Item): Promise<boolean> {
		if (cur?.page_type === "information_subpage" && String(cur.product?.asin ?? "") === it.asin) await step("prev_page", {});
		if (atProduct(it.asin)) return true;
		if (!onResults(it.asin)) {
			if (!(await step("search_products", { query: it.query || it.title }))) return false;
			for (let p = 1; p < it.page && cur?.page_type === "search_results" && !onResults(it.asin); p++) {
				if (!(await step("next_page", {}))) break;
			}
		}
		if (!onResults(it.asin) && !shop.isDone()) {
			if (!(await step("search_products", { query: it.title }))) return false;
		}
		if (!onResults(it.asin)) return false;
		if (!(await step("open_product", { asin: it.asin }))) return false;
		return atProduct(it.asin);
	}

	/** 系统替导购下单时，没定的规格项按需求单里对得上的选项选（画像偏好最后看），都对不上就选第一个。 */
	function guessOption(axis: string, values: string[]): string {
		const confirmed = reqs.filter((x) => x.status === "已确认");
		for (const r of [...confirmed.filter((x) => x.source !== "画像偏好"), ...confirmed.filter((x) => x.source === "画像偏好")]) {
			const hit = matchOption({ [axis]: values }, r.value);
			if (hit) return hit[1];
		}
		return values[0];
	}

	/** forced：必须下单的阶段（规格没定时不再让导购去问用户）；fill：系统替导购下单，没定的规格自己选。 */
	async function purchase(it: Item, wanted: string[], how: { forced?: boolean; fill?: boolean } = {}): Promise<{ done: boolean; text: string }> {
		if (!(await gotoProduct(it))) {
			return { done: shop.isDone(), text: `没能回到商品 [${it.asin}] 的页面（回合或环境步数可能不够了）。` };
		}
		const axes = Object.fromEntries(
			Object.entries((cur.available_options ?? {}) as Record<string, unknown[]>).map(([k, vs]) => [k, (vs ?? []).map(String)]),
		);
		const plan: [string, string][] = [];
		const bad: string[] = [];
		for (const w of wanted) {
			const hit = matchOption(axes, w);
			if (!hit) bad.push(w);
			else if (!plan.some(([a]) => a === hit[0])) plan.push(hit);
		}
		if (bad.length && !how.fill) {
			return { done: false, text: `「${bad.join("」「")}」不是这件商品的规格选项。可选规格：${JSON.stringify(axes)}。请照抄选项原文再调用 buy。` };
		}
		const open: string[] = [];
		for (const [axis, values] of Object.entries(axes)) {
			if (plan.some(([a]) => a === axis)) continue;
			if (values.length === 1) plan.push([axis, values[0]]);
			else if (how.fill) plan.push([axis, guessOption(axis, values)]);
			else open.push(`「${axis}」（可选 ${quoteValues(values, 8)}）`);
		}
		if (open.length) {
			const next = how.forced
				? "现在必须下单，不要再问用户：每项选最接近用户需求的那个（用户没说过的选第一个），再调用 buy 并在 options 里写上。"
				: "请先问用户要哪个，再调用 buy 并在 options 里写上。";
			return { done: false, text: `还有规格没定：${open.join("；")}。${next}` };
		}
		for (const [axis, value] of plan) {
			if (looseKey(cur.selected_options?.[axis]) === looseKey(value)) continue;
			if (!(await step("select_option", { value }))) return { done: shop.isDone(), text: `规格“${value}”没能点上。` };
		}
		const r = await step("buy_now", {});
		return { done: shop.isDone(), text: r ? r.text : "下单没有成功。" };
	}

	function consentFor(asin: string): boolean {
		if (!dialogue.length || attitude.n !== dialogue.length || attitude.status !== "同意购买") return false;
		const named = attitude.product;
		return !named || named === asin || !pool.has(named);
	}

	// ---- 事件 ----

	pi.on("message_end", async (event: any) => {
		if (event?.message?.role === "assistant") lastAssistantText = textOf(event.message.content);
		return undefined;
	});

	pi.on("before_agent_start", async (event) => {
		if (!opening) {
			opening = String(event.prompt ?? "");
			scheduleRequirements();
		}
		return undefined;
	});

	pi.events.on("shop:state", (data: any) => {
		cur = data?.state ?? {};
		record(cur);
	});

	pi.events.on("shop:ask", (data: any) => {
		const answer = String(data?.answer ?? "");
		dialogue.push({ question: String(data?.question ?? ""), answer, turn: shop.turn() });
		if (BYE.test(answer)) userSaidBye = true;
		// 系统已经要求必须下单之后，导购又对用户说了一次话
		if (closeAnnounced) closeIgnored += 1;
		scheduleRequirements();
	});

	// 收尾阶段导购还是不下单（比如认为用户说了不买就不能买）：提示过必须下单后它又对用户说了话，
	// 或者回合只剩 3 个，系统在它下一次调用模型前按官方规则替它下单。用户刚刚同意买的，先让导购自己下单。
	pi.on("context", async () => {
		if (shop.isDone() || !mustClose() || autoTried) return undefined;
		await reqJob;
		const agreed = dialogue.length > 0 && attitude.n === dialogue.length && attitude.status === "同意购买";
		if (turnsLeft() > 3 && (closeIgnored < 1 || (agreed && closeIgnored < 2))) return undefined;
		autoTried = true;
		await autoBuy();
		return undefined;
	});

	/** 用户最倾向的那件：用户最后一次表态针对的商品（不是“不要这件”），否则候选单第 1 件。 */
	function closestPick(): Pick | undefined {
		const named = attitude.n === dialogue.length ? pool.get(attitude.product) : undefined;
		if (named && attitude.status !== "不要这件") return lastSheet.find((p) => p.item === named) ?? { item: named, checks: new Map(), wanted: {} };
		return lastSheet.find((p) => p.item !== named) ?? lastSheet[0];
	}

	async function autoBuy(): Promise<void> {
		if (!lastSheet.length) await buildSheet("auto");
		const pick = closestPick();
		if (!pick || shop.isDone()) return;
		const options = Object.entries(pick.item.options ?? {})
			.filter(([, values]) => values.length > 1)
			.map(([axis, values]) => (pick.wanted[axis] ?? values)[0]);
		logEvent({ type: "roles_auto_buy", turn: shop.turn(), asin: pick.item.asin, options, ignored: closeIgnored, left: turnsLeft(), attitude: attitude.status });
		const res = await purchase(pick.item, options, { forced: true, fill: true });
		logEvent({ type: "roles_buy", turn: shop.turn(), asin: pick.item.asin, options, consent: false, forced: true, auto: true, done: res.done, note: res.done ? "" : short(res.text, 120) });
	}

	pi.events.on("shop:relayed", (data: any) => {
		// 适配层同步收集附加在转回消息后面的提示
		const note = closingNote();
		if (note && Array.isArray(data?.notes)) data.notes.push(note);
	});

	pi.on("tool_result", async (event) => {
		if (event.toolName !== "ask_shopper" || event.isError || shop.isDone()) return undefined;
		const note = closingNote();
		return note ? { content: [...event.content, { type: "text" as const, text: `\n——\n${note}` }] } : undefined;
	});

	// ---- 导购的工具 ----

	pi.registerTool({
		name: "find_products",
		label: "派买手去找",
		description:
			"派买手去商店里找商品或查看商品。request 写清楚这次要找什么、看什么（比如“找单瓶装的蟑螂药，预算 30 元左右”“打开候选 2 看规格和价格”）。" +
			"买手做完后交回候选单：最接近需求的几件，每件对照需求单标出符合、不符合、页面没写。买手在商店里的每个动作都算一个回合。",
		parameters: Type.Object({ request: Type.String({ description: "这次找货的任务" }) }),
		async execute(_toolCallId: string, params: any, _signal: unknown, _onUpdate: unknown, ctx: any) {
			const request = String(params?.request ?? "").trim();
			if (!request) return reply("找货任务不能为空。");
			const urgent = mustClose();
			if (urgent && lastSheet.length) return reply(shop.withTurn(closingNote()));
			const budget = urgent ? Math.max(1, turnsLeft() - 4) : Math.min(BUYER_STEPS, turnsLeft() - RESERVE, stepsLeft() - RESERVE);
			if (budget <= 0) {
				const gate = await internalGate();
				return reply(gate ?? withNote(shop.withTurn(`回合不多了（还剩 ${turnsLeft()} 个），不能再派买手去找。请从候选单里请用户挑一件。`)));
			}
			await reqJob;
			// 连着两次找货候选单第 1 件都没变，按最新的需求单它也没有和用户要求冲突的地方：剩下的只是页面没写的信息，
			// 商店里多半没有更合适的了，再找只会多花回合
			if (!urgent && staleFinds >= 2 && lastSheet.length) {
				if (reqKey() !== sheetKey) await buildSheet("recheck");
				const top = lastSheet[0];
				const conflict = reqs.some((r) => r.status === "已确认" && top?.checks.get(r.id)?.verdict === "不符合");
				if (top && !conflict) {
					const gate = await internalGate();
					logEvent({ type: "roles_find_stale", turn: shop.turn(), top: top.item.asin, request });
					return reply(
						gate ??
							withNote(
								shop.withTurn(
									`【不用再找了】最近两次找货，候选单第 1 件都没变；按最新的需求单，第 1 件 [${top.item.asin}] 没有和用户的要求冲突的地方，` +
										"页面没写的信息再找也多半找不到。请把这件推荐给用户，说清楚页面上写了什么、和用户的要求怎么对上，请用户确认；" +
										`用户要的信息页面上确实没有，就如实说明，让用户决定。\n${sheetText(lastSheet)}`,
								),
							),
					);
				}
			}
			const before = shop.turn();
			let summary = "";
			try {
				summary = await runBuyer(request, Math.min(budget, stepsLeft()), ctx?.model);
			} catch (err) {
				logEvent({ type: "roles_buyer_error", turn: shop.turn(), error: String(err) });
				summary = "（买手出错了）";
			}
			if (shop.isDone()) return { ...reply("【任务已结束】"), terminate: true };
			const picks = await buildSheet("find");
			const used = shop.turn() - before;
			const top = picks[0]?.item.asin ?? "";
			staleFinds = top && top === lastTop ? staleFinds + 1 : 1;
			lastTop = top;
			logEvent({ type: "roles_find", turn: shop.turn(), request, budget, actions: used, top: picks.map((p) => p.item.asin), stale: staleFinds });
			return reply(withNote(shop.withTurn(`${sheetText(picks)}\n买手说：${summary || "（没有说明）"}\n（这次买手做了 ${used} 个动作）`)));
		},
	});

	pi.registerTool({
		name: "show_candidates",
		label: "看候选单",
		description:
			"不操作商店，不花回合。不填 asin：按最新的需求单把看过的所有商品重新排名，交回候选单（用户补充或改了需求以后用）。" +
			"填 asin：交回这件商品页面上记下的全部内容（标题、属性、规格、看过的详情页），用户问某件商品的细节时先用它查。",
		parameters: Type.Object({
			asin: Type.Optional(Type.String({ description: "只看某一件商品时填它的商品ID" })),
		}),
		async execute(_toolCallId: string, params: any) {
			const gate = await internalGate();
			if (gate) return reply(gate);
			const asin = String(params?.asin ?? "").trim();
			if (asin) {
				const it = pool.get(asin);
				logEvent({ type: "roles_show_item", turn: shop.turn(), asin, found: Boolean(it) });
				if (!it) return reply(`没看过商品 ${asin}，请照抄候选单上方括号里的商品ID。`);
				return reply(withNote(shop.withTurn(itemCard(it))));
			}
			const picks = await buildSheet("show");
			return reply(withNote(shop.withTurn(sheetText(picks))));
		},
	});

	pi.registerTool({
		name: "buy",
		label: "下单",
		description:
			"下单购买候选单上的一件商品，下单后任务结束。asin 照抄候选单上方括号里的商品ID；options 写用户选定的规格值，照抄候选单上的原文" +
			"（只有一个选项的规格会自动点上）。只有用户在最后一句话里明确同意买这件以后才能下单。",
		parameters: Type.Object({
			asin: Type.String({ description: "商品ID" }),
			options: Type.Optional(Type.Array(Type.String(), { description: "要选的规格值，照抄候选单上的原文" })),
		}),
		async execute(_toolCallId: string, params: any) {
			const asin = String(params?.asin ?? "").trim();
			const wanted = (Array.isArray(params?.options) ? params.options : []).map(String).filter(Boolean);
			const it = pool.get(asin);
			if (!it) {
				const gate = await internalGate();
				return reply(gate ?? `候选单里没有商品 ${asin}，请照抄候选单上方括号里的商品ID。`);
			}
			await reqJob;
			const forced = mustClose();
			const agreed = consentFor(asin);
			if (!agreed && !forced) {
				const gate = await internalGate();
				if (gate) return reply(gate);
				// 拦截审计用：放行的话会选哪些规格（只有一个选项的会自动点上）
				const planned = [...wanted, ...Object.values(it.options ?? {}).filter((vs) => vs.length === 1).map((vs) => vs[0])];
				logEvent({
					type: "roles_consent_block",
					turn: shop.turn(),
					asin,
					selected: Object.fromEntries(planned.map((v, i) => [`选项${i + 1}`, v])),
					attitude: attitude.status,
					quote: attitude.quote,
				});
				const last = dialogue.length ? dialogue[dialogue.length - 1].answer : opening;
				const refused = attitude.n === dialogue.length && attitude.status === "不要这件";
				const next = refused
					? "用户明确说了不要，不要再劝用户买这件：问清用户的顾虑，看看候选单里有没有更合适的；用户要的信息页面上确实没有，如实说明后让用户决定。"
					: "先把这件商品、规格和价格报给用户，问用户要不要买；用户点头后再调用 buy。页面上没写的信息如实告诉用户，让用户自己决定。";
				const timing = `现在还没到对话结束：只有用户说了再见、或者回合快用完时（系统会提示【现在必须下单】），才能不经用户同意下单，现在还剩 ${turnsLeft()} 个回合。`;
				return reply(withNote(shop.withTurn(`【还不能下单】用户最后一句话没有明确同意买这件（用户说：「${short(last, 80)}」）。${timing}${next}${TEXT_ONLY}`)));
			}
			const res = await purchase(it, wanted, { forced });
			logEvent({ type: "roles_buy", turn: shop.turn(), asin, options: wanted, consent: agreed, forced, done: res.done, note: res.done ? "" : short(res.text, 120) });
			if (res.done) return { ...reply("【已下单，任务结束】"), terminate: true };
			return reply(withNote(shop.withTurn(res.text)));
		},
	});
}
