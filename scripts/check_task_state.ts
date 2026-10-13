// 任务状态层里不依赖 pi 的小函数的自查：node scripts/check_task_state.ts
import { evidenceFrom, parseBudget, quotedFrom } from "../pi-shop/task_state.ts";
const cases: [string, any][] = [
  ["115-125元", { lo: 115, hi: 125 }], ["100块上下", { lo: 80, hi: 120 }], ["两百以内", { lo: 0, hi: 200 }],
  ["不超过1300元", { lo: 0, hi: 1300 }], ["200出头", { lo: 200, hi: 240 }], ["65元左右", { lo: 52, hi: 78 }],
  ["140元到150元之间", { lo: 140, hi: 150 }], ["70元以内", { lo: 0, hi: 70 }], ["500以上", { lo: 500, hi: Infinity }],
  ["预算不高", null], ["100", { lo: 80, hi: 120 }], ["一百块左右", { lo: 80, hi: 120 }],
];
for (const [t, want] of cases) {
  const got = parseBudget(t);
  const ok = JSON.stringify(got && { lo: Math.round(got.lo), hi: got.hi === Infinity ? "inf" : Math.round(got.hi) }) === JSON.stringify(want && { lo: want.lo, hi: want.hi === Infinity ? "inf" : want.hi });
  console.log(ok ? "ok  " : "FAIL", t, JSON.stringify(got));
}
const facts = "标题：28度黑茶粉速溶正宗湖南安化黑茶无糖金花茶饮品大餐救星散20支装\n属性：黑茶粉；速溶；正宗；无糖；大餐救星\n价格：120.0";
console.log(evidenceFrom("属性：速溶", facts), evidenceFrom("速溶；无糖", facts), evidenceFrom("120元", facts), evidenceFrom("属性：冲泡方便", facts));
console.log(quotedFrom("我想要的是**象牙白**的色号", "我想要的是**象牙白**的色号。另外"), quotedFrom("预算一百", "我的预算是两百"));
