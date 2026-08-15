"""
route_a.py  ——  临床规则引擎（安全底座）
硬约束：不可绕过，所有推荐必须先经此层
参考标准：
  GB 29922-2013  特殊医学用途配方食品通则
  GB 25596-2010  特殊医学用途婴儿配方食品通则
  ESPGHAN 2022   牛奶蛋白过敏（CMPA）管理指南
  ESPEN 2019     临床营养实践指南
"""
from __future__ import annotations
import re, sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════
# 临床知识库（Clinical Knowledge Base）
# 这是整个推荐系统唯一的"医学事实"存储点
# ══════════════════════════════════════════════════════════════

@dataclass
class ClinicalRule:
    """单条临床规则"""
    condition     : str           # 病症/需求
    preferred     : list[str]     # 首选产品类别（有序）
    alternative   : list[str]     # 次选产品类别（有序）
    contraindicated: list[str]    # 禁用产品类别（绝对禁用）
    age_group     : str           # 适用年龄段
    evidence      : str           # 循证依据
    notes         : str = ""      # 临床补充说明
    combination   : str = ""      # 组合用药提示


CLINICAL_KB: dict[str, ClinicalRule] = {

    # ── 婴儿专用 ──────────────────────────────────────────────
    "食物蛋白过敏": ClinicalRule(
        condition      = "食物蛋白过敏",
        preferred      = ["氨基酸配方"],
        alternative    = ["乳蛋白深度水解配方"],
        contraindicated= ["乳蛋白部分水解配方"],
        age_group      = "婴儿",
        evidence       = (
            "ESPGHAN 2022: IgE介导或严重CMPA首选氨基酸配方（AAF），"
            "游离氨基酸分子量<300Da，免疫原性为零；"
            "中重度确诊患者可用深度水解配方（eHF），肽链<1500Da；"
            "部分水解配方（pHF）仅用于高风险婴儿预防，禁用于确诊患者"
        ),
        notes          = "严重过敏（全身症状/生长迟缓）直接跳至AAF，勿尝试eHF",
    ),

    "乳蛋白过敏高风险": ClinicalRule(
        condition      = "乳蛋白过敏高风险",
        preferred      = ["乳蛋白部分水解配方"],
        alternative    = [],
        contraindicated= [],
        age_group      = "婴儿",
        evidence       = (
            "ESPGHAN 2022: 有一级亲属（父母/兄弟姐妹）"
            "过敏史的高风险婴儿，预防性使用pHF可降低过敏发生率"
        ),
    ),

    "乳糖不耐受_婴儿": ClinicalRule(
        condition      = "乳糖不耐受_婴儿",
        preferred      = ["无乳糖配方"],
        alternative    = ["低乳糖配方"],
        contraindicated= [],
        age_group      = "婴儿",
        evidence       = (
            "GB 25596-2010: 先天性乳糖酶缺乏或继发性乳糖不耐受，"
            "完全无乳糖配方（<10mg/100kcal）；"
            "轻度不耐受可用低乳糖配方（≤2g/100kcal）"
        ),
    ),

    "早产": ClinicalRule(
        condition      = "早产",
        preferred      = ["早产/低出生体重婴儿配方"],
        alternative    = ["母乳营养补充剂"],
        contraindicated= [],
        age_group      = "婴儿",
        evidence       = (
            "GB 25596-2010: 早产儿（<37周）或低出生体重儿（<2500g）"
            "需要高能量密度（≥80kcal/100mL）、高蛋白（2.4-3.6g/100kcal）；"
            "母乳喂养者添加母乳营养补充剂（HMF）强化"
        ),
        combination    = "早产儿配方与母乳营养补充剂可联合使用（母乳+HMF）",
    ),

    "苯丙酮尿症_婴儿": ClinicalRule(
        condition      = "苯丙酮尿症_婴儿",
        preferred      = ["氨基酸代谢障碍配方"],
        alternative    = [],
        contraindicated= ["氨基酸配方", "乳蛋白深度水解配方",
                          "乳蛋白部分水解配方"],
        age_group      = "婴儿",
        evidence       = (
            "GB 25596-2010: PKU婴儿需无苯丙氨酸氨基酸配方，"
            "普通氨基酸配方含苯丙氨酸，禁用"
        ),
    ),

    # ── 1岁以上专用 ────────────────────────────────────────────
    "补充蛋白质": ClinicalRule(
        condition      = "补充蛋白质",
        preferred      = ["蛋白质（氨基酸）组件"],
        alternative    = ["非全营养配方食品"],  # 其中蛋白质类
        contraindicated= [],
        age_group      = "1岁以上",
        evidence       = (
            "ESPEN 2019: 蛋白质-能量营养不良患者优先补充高纯蛋白组件，"
            "含量4.8-5.9g/100mL，碳水极低（0.01-0.56g/100mL），"
            "天然兼容乳糖不耐受；"
            "非全营养，需与全营养配方联合使用"
        ),
        combination    = (
            "蛋白质组件为非全营养产品，须与全营养配方食品联合使用，"
            "建议按蛋白质缺口计算补充剂量，全营养配方提供整体营养基础"
        ),
        notes          = "乳糖不耐受患者可直接使用蛋白质组件（碳水化合物极低）",
    ),

    "消化吸收障碍": ClinicalRule(
        condition      = "消化吸收障碍",
        preferred      = ["全营养配方食品"],
        alternative    = ["非全营养配方食品"],
        contraindicated= [],
        age_group      = "1岁以上",
        evidence       = (
            "GB 29922-2013 / ESPEN 2019: 进食受限、消化吸收障碍、"
            "代谢紊乱患者，全营养配方可作为单一营养来源，"
            "1kcal/mL能量密度，平衡宏量与微量营养素"
        ),
    ),

    "肿瘤": ClinicalRule(
        condition      = "肿瘤",
        preferred      = ["特定全营养配方食品"],
        alternative    = ["全营养配方食品"],
        contraindicated= [],
        age_group      = "1岁以上",
        evidence       = (
            "ESPEN 2017肿瘤营养指南: 肿瘤患者存在营养风险，"
            "特定全营养配方（速熠素）蛋白质含量≥1.45g/100mL，"
            "能量密度高，ω-3脂肪酸调节炎症反应；"
            "当特定全营养不可及时，高蛋白全营养配方为替代"
        ),
    ),

    "腹泻脱水": ClinicalRule(
        condition      = "腹泻脱水",
        preferred      = ["电解质配方"],
        alternative    = ["非全营养配方食品"],
        contraindicated= [],
        age_group      = "1岁以上",
        evidence       = (
            "WHO ORS标准: 轻中度腹泻脱水首选口服补液盐，"
            "纠正水电解质失衡；电解质配方符合WHO ORS渗透压标准"
        ),
    ),

    "吞咽障碍": ClinicalRule(
        condition      = "吞咽障碍",
        preferred      = ["增稠组件"],
        alternative    = ["全营养配方食品"],
        contraindicated= [],
        age_group      = "1岁以上",
        evidence       = (
            "IDDSI（国际吞咽障碍食物标准化协作组）框架: "
            "调节液体稠度（Level 0-4）降低误吸风险；"
            "增稠组件单独使用或加入全营养配方中"
        ),
        combination    = "增稠组件须配合全营养配方使用以满足营养需求",
    ),

    "苯丙酮尿症": ClinicalRule(
        condition      = "苯丙酮尿症",
        preferred      = ["非全营养配方食品"],  # PKU配方在非全营养分类下
        alternative    = [],
        contraindicated= ["氨基酸配方"],        # 含苯丙氨酸
        age_group      = "1岁以上",
        evidence       = (
            "GB 29922-2013: 1岁以上PKU患者使用低苯丙氨酸"
            "氨基酸特殊配方，严格控制苯丙氨酸摄入"
        ),
    ),
}


# ══════════════════════════════════════════════════════════════
# 年龄解析（将自然语言年龄描述转为标准分组）
# ══════════════════════════════════════════════════════════════
AGE_PATTERNS = [
    # 婴儿（0-12月龄）
    (r'婴儿|月龄|0\s*[~～-]\s*12|新生儿|infant',       '婴儿'),
    # 1-10岁儿童
    (r'[1-9]\s*岁|10\s*岁|儿童|幼儿|小孩|孩子|child',  '1-10岁'),
    # 10岁以上成人
    (r'成人|成年|老人|老年|adult|青少年|teen',           '10岁以上'),
]

def parse_age_group(text: str) -> Optional[str]:
    """从自然语言文本中识别年龄段"""
    text = text.lower()
    for pattern, group in AGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return group
    return None


# ══════════════════════════════════════════════════════════════
# 用户查询解析
# ══════════════════════════════════════════════════════════════
@dataclass
class ParsedQuery:
    age_group : str
    conditions: list[str]
    raw_input : dict


def parse_query(client: dict) -> ParsedQuery:
    """
    将客户输入字典解析为结构化查询
    处理边界情况：
      - 年龄未指定 → 从 query 字段推断
      - 乳糖不耐受 → 区分婴儿/成人（成人无专用产品，需特殊处理）
      - 多病症 → 按临床优先级排序
    """
    # 1. 年龄段
    age = client.get('age_group')
    if not age:
        age = parse_age_group(client.get('query', ''))
    if not age:
        age = '1岁以上'   # 默认值

    # 2. 病症列表（去重 + 归一化）
    conditions = list(dict.fromkeys(client.get('diseases', [])))

    # 3. 乳糖不耐受：区分年龄段
    if '乳糖不耐受' in conditions:
        conditions.remove('乳糖不耐受')
        if age == '婴儿':
            conditions.insert(0, '乳糖不耐受_婴儿')
        # 成人乳糖不耐受：无专用产品，在 notes 中说明

    # 4. PKU：区分年龄段
    if '苯丙酮尿症' in conditions:
        conditions.remove('苯丙酮尿症')
        if age == '婴儿':
            conditions.insert(0, '苯丙酮尿症_婴儿')
        else:
            conditions.append('苯丙酮尿症')

    logger.debug(f"解析结果: 年龄={age}, 病症={conditions}")
    return ParsedQuery(age_group=age, conditions=conditions, raw_input=client)


# ══════════════════════════════════════════════════════════════
# 硬约束过滤器
# ══════════════════════════════════════════════════════════════
@dataclass
class FilterResult:
    """Route A 输出"""
    pool        : pd.DataFrame             # 候选产品集
    tiers       : dict                     # {condition: {S:[], A:[], ban:[]}}
    active_rules: dict[str, ClinicalRule]  # 命中的规则
    warnings    : list[str]                # 安全警告
    notes       : list[str]                # 临床提示
    combination : list[str]                # 组合用药建议


class RouteA:
    """
    临床规则引擎
    Step 1: 年龄硬过滤（人群类别）
    Step 2: 年龄内精确匹配（适用人群文本）
    Step 3: 每种病症输出首选/次选/禁用分层
    """
    AGE_TO_POP = {
        '婴儿':   '特医婴配食品',
        '1-10岁': '1岁以上特医食品',
        '10岁以上':'1岁以上特医食品',
        '1岁以上':'1岁以上特医食品',
    }
    # 年龄段 → 适用人群文本关键词
    AGE_TEXT_FILTER = {
        '1-10岁': [r'1.*10岁', r'1\s*~\s*10', r'1\s*～\s*10',
                   r'1岁以上', r'10岁以上'],
    }

    def run(self, df: pd.DataFrame, query: ParsedQuery) -> FilterResult:
        warnings, notes, combinations = [], [], []

        # ── Step 1: 人群类别硬过滤 ──────────────────────────
        pop_cat = self.AGE_TO_POP.get(query.age_group, '1岁以上特医食品')
        pool    = df[df['适用人群类别'] == pop_cat].copy()
        logger.info(f"[A] 年龄过滤 '{query.age_group}' → '{pop_cat}': "
                    f"{len(pool)}款")

        # ── Step 2: 年龄文本精确过滤（1-10岁细分）──────────
        if query.age_group == '1-10岁':
            patterns = self.AGE_TEXT_FILTER['1-10岁']
            mask = pool['适用人群'].fillna('').apply(
                lambda t: any(re.search(p, t) for p in patterns)
            )
            pool = pool[mask].copy()
            logger.info(f"[A] 年龄文本细化过滤后: {len(pool)}款")

        # ── Step 3: 每种病症规则匹配 ─────────────────────────
        tiers       = {}
        active_rules= {}

        for cond in query.conditions:
            rule = CLINICAL_KB.get(cond)
            if rule is None:
                warnings.append(f"病症 '{cond}' 暂无临床规则，跳过")
                continue
            active_rules[cond] = rule

            # 收集各优先级产品索引
            tier = {'S': [], 'A': [], 'ban': []}

            for idx, row in pool.iterrows():
                cat = str(row.get('产品类别', ''))
                # 禁用：立即排除
                if any(bc in cat for bc in rule.contraindicated):
                    tier['ban'].append(idx)
                    continue
                # 首选
                if any(pc in cat or cat in pc
                       for pc in rule.preferred):
                    tier['S'].append(idx)
                # 次选（且不在首选）
                elif any(ac in cat or cat in ac
                         for ac in rule.alternative):
                    tier['A'].append(idx)

            tiers[cond] = tier
            logger.info(f"[A] {cond}: "
                        f"首选{len(tier['S'])}款, "
                        f"次选{len(tier['A'])}款, "
                        f"禁用{len(tier['ban'])}款")

            # 收集临床提示
            if rule.notes:
                notes.append(f"[{cond}] {rule.notes}")
            if rule.combination:
                combinations.append(f"[{cond}] {rule.combination}")

        # ── 成人乳糖不耐受特别处理 ────────────────────────────
        if (query.age_group != '婴儿' and
                '乳糖不耐受' in query.raw_input.get('diseases', [])):
            notes.append(
                "【乳糖不耐受 · 成人】1岁以上特医食品中无乳糖不耐受专用产品。"
                "蛋白质（氨基酸）组件碳水化合物含量极低（0.01–0.56g/100mL），"
                "天然兼容乳糖不耐受，建议优先选用。"
            )

        # ── 关键修复：从候选池中彻底删除禁用产品 ───────────────
        # Route B 不应看到禁用产品，否则高 CBF 得分可能使其进入 Top-8
        all_banned = set()
        for cond_tier in tiers.values():
            all_banned.update(cond_tier.get('ban', []))
        if all_banned:
            pool = pool.drop(index=list(all_banned & set(pool.index)),
                             errors='ignore')
            logger.info(f"[A] 剔除禁用产品后候选池: {len(pool)}款 "
                        f"(移除{len(all_banned)}款)")

        return FilterResult(
            pool         = pool,
            tiers        = tiers,
            active_rules = active_rules,
            warnings     = warnings,
            notes        = notes,
            combination  = combinations,
        )


# ══════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from schema import load_existing_data
    df = load_existing_data(
        '/mnt/user-data/uploads/1779784591591_result1.xlsx',
        '/mnt/user-data/uploads/1779784591592_result2.xlsx',
    )
    engine = RouteA()

    for client in [
        {'name':'客户1','age_group':'婴儿',
         'diseases':['食物蛋白过敏'],'query':'婴儿蛋白质过敏'},
        {'name':'客户2','age_group':'1-10岁',
         'diseases':['补充蛋白质'],'query':'10岁 补蛋白 乳糖不耐受',
         'extra_diseases_note':['乳糖不耐受']},
    ]:
        q = parse_query(client)
        # 注入 raw 中的 diseases 包含乳糖不耐受
        if 'extra_diseases_note' in client:
            q.raw_input['diseases'] = (
                client['diseases'] + client['extra_diseases_note'])
        r = engine.run(df, q)
        print(f"\n{'='*55}")
        print(f"  {client['name']}: 候选{len(r.pool)}款")
        for cond, t in r.tiers.items():
            print(f"  [{cond}] S={len(t['S'])} A={len(t['A'])} ban={len(t['ban'])}")
        for n in r.notes: print(f"  💡 {n}")
        for c in r.combination: print(f"  🔗 {c}")
        if r.warnings: print(f"  ⚠  {r.warnings}")
