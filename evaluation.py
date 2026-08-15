"""
evaluation.py  ——  消融实验评估框架
基于 outputs/ground_truth.csv 对三路融合推荐系统进行量化评估

指标：
  ① Precision@K / Recall@K / NDCG@K   （核心推荐准确率）
  ② 安全违规率                          （Route A 安全性验证）
  ③ Kendall's τ                        （排序质量，Route B 边际贡献）
  ④ LLM-as-Judge                       （解释质量，Route C 边际贡献）

消融配置：
  A       ——  仅临床规则（tier 排序 + 蛋白质兜底）
  A+B     ——  规则 + CBF 量化得分（需 MindSpore）
  A+B+C   ——  规则 + CBF + LLM 解释（需 MindSpore + API）
"""
from __future__ import annotations

import sys, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config       import LLM_CFG, REC_CFG, RESULT1_PATH, RESULT2_PATH
from schema       import load_existing_data
from route_a      import RouteA, parse_query, CLINICAL_KB
from llm_providers import get_provider

# ── MindSpore 可选加载 ────────────────────────────────────────────
HAS_MS = False
try:
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE)
    from route_b_mindspore import RouteB_CBF, FeatureEngine
    HAS_MS = True
    logger.info("MindSpore 可用 — Route B 已启用")
except Exception as _e:
    logger.warning(f"MindSpore 不可用（{_e}），仅运行 Route A")

# ── 常量 ──────────────────────────────────────────────────────────
K = 5    # 评估截止位置，所有 @K 指标均使用此值
GT_PATH = ROOT / "outputs" / "ground_truth.csv"
OUT_DIR = ROOT / "outputs"

# ══════════════════════════════════════════════════════════════════
# Ground Truth 加载
# ══════════════════════════════════════════════════════════════════
def load_ground_truth() -> dict[str, dict[str, set[str]]]:
    """
    Returns:
        {condition: {"S": set(注册证号), "A": set(), "ban": set()}}
    """
    df = pd.read_csv(GT_PATH, encoding="utf-8-sig")
    gt: dict = {}
    for cond, grp in df.groupby("condition"):
        gt[cond] = {
            "S":   set(grp[grp["label"] == "S"]["注册证号"]),
            "A":   set(grp[grp["label"] == "A"]["注册证号"]),
            "ban": set(grp[grp["label"] == "ban"]["注册证号"]),
        }
    return gt


# ══════════════════════════════════════════════════════════════════
# 测试案例（11 个病症，覆盖所有 GT 条目）
# ══════════════════════════════════════════════════════════════════
TEST_CASES: list[dict] = [
    # ── 婴儿 ──────────────────────────────────────────────────────
    {"name": "食物蛋白过敏",     "age_group": "婴儿",    "diseases": ["食物蛋白过敏"],
     "query": "婴儿确诊牛奶蛋白过敏（IgE介导），需特医配方"},
    {"name": "乳蛋白过敏高风险", "age_group": "婴儿",    "diseases": ["乳蛋白过敏高风险"],
     "query": "婴儿父母有过敏史，乳蛋白过敏高风险预防"},
    {"name": "乳糖不耐受_婴儿",  "age_group": "婴儿",    "diseases": ["乳糖不耐受"],
     "query": "婴儿乳糖不耐受，腹泻腹胀"},
    {"name": "早产",             "age_group": "婴儿",    "diseases": ["早产"],
     "query": "32周早产婴儿，出生体重1.8kg，需高能量配方"},
    {"name": "苯丙酮尿症_婴儿",  "age_group": "婴儿",    "diseases": ["苯丙酮尿症"],
     "query": "婴儿PKU确诊，需无苯丙氨酸配方"},
    # ── 1岁以上 ───────────────────────────────────────────────────
    {"name": "补充蛋白质",       "age_group": "1岁以上", "diseases": ["补充蛋白质"],
     "query": "成人蛋白质-能量营养不良，术后蛋白质补充"},
    {"name": "消化吸收障碍",     "age_group": "1岁以上", "diseases": ["消化吸收障碍"],
     "query": "成人消化吸收功能障碍，需全营养支持"},
    {"name": "肿瘤",             "age_group": "1岁以上", "diseases": ["肿瘤"],
     "query": "消化道肿瘤患者，化疗期间营养支持"},
    {"name": "腹泻脱水",         "age_group": "1岁以上", "diseases": ["腹泻脱水"],
     "query": "急性腹泻脱水，需口服补液"},
    {"name": "吞咽障碍",         "age_group": "1岁以上", "diseases": ["吞咽障碍"],
     "query": "脑卒中后吞咽障碍，需增稠液体"},
    {"name": "苯丙酮尿症",       "age_group": "1岁以上", "diseases": ["苯丙酮尿症"],
     "query": "成人PKU，需长期低苯丙氨酸饮食管理"},
]


# ══════════════════════════════════════════════════════════════════
# ① 核心推荐准确率指标
# ══════════════════════════════════════════════════════════════════
def precision_at_k(recommended: list[str], s_set: set[str], k: int = K) -> float:
    """top-K 中 S 级产品占比"""
    return sum(1 for p in recommended[:k] if p in s_set) / k if k else 0.0


def recall_at_k(recommended: list[str], s_set: set[str], k: int = K) -> float:
    """top-K 覆盖了多少比例的 S 级产品（上限由 K 决定）"""
    if not s_set:
        return 1.0
    return sum(1 for p in recommended[:k] if p in s_set) / len(s_set)


def ndcg_at_k(recommended: list[str],
              s_set: set[str], a_set: set[str],
              k: int = K) -> float:
    """
    Graded NDCG：S=2, A=1, 其他=0
    DCG = Σ rel(i) / log2(i+2)
    """
    def rel(p: str) -> int:
        if p in s_set: return 2
        if p in a_set: return 1
        return 0

    dcg  = sum(rel(p) / np.log2(i + 2) for i, p in enumerate(recommended[:k]))
    # Ideal：最多 K 个，按最高 rel 排列
    ideal = sorted((rel(p) for p in (s_set | a_set)), reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# ② 安全违规率（Route A 核心价值验证）
# ══════════════════════════════════════════════════════════════════
def safety_violation_rate(recommended: list[str], ban_set: set[str]) -> float:
    """
    推荐列表中出现禁用产品的比例（Route A 正常工作时应始终为 0）
    """
    if not recommended:
        return 0.0
    return len([p for p in recommended if p in ban_set]) / len(recommended)


# ══════════════════════════════════════════════════════════════════
# ③ Kendall's τ（排序质量，体现 Route B 价值）
# ══════════════════════════════════════════════════════════════════
def kendall_tau_score(recommended: list[str],
                      s_set: set[str], a_set: set[str]) -> Optional[float]:
    """
    只对推荐列表中属于 S∪A 的产品计算 τ：
    · 系统排名：产品在 recommended 中的顺序（越前越好）
    · 理想排名：S 产品均排在 A 产品之前
    · τ = 1  → 完全正确（所有 S 在 A 前面）
    · τ = -1 → 完全颠倒
    """
    from scipy.stats import kendalltau

    relevant = [p for p in recommended if p in s_set or p in a_set]
    if len(relevant) < 2:
        return None

    sys_rank   = list(range(len(relevant)))
    # 理想顺序：S 优先（0），A 其次（1），同级内保持系统顺序
    ideal_order = sorted(relevant, key=lambda p: (0 if p in s_set else 1,
                                                   relevant.index(p)))
    ideal_rank  = [ideal_order.index(p) for p in relevant]

    τ, _ = kendalltau(sys_rank, ideal_rank)
    return float(τ)


# ══════════════════════════════════════════════════════════════════
# 推荐执行：Route A（纯规则排序）
# ══════════════════════════════════════════════════════════════════
def run_route_a(df: pd.DataFrame, tc: dict, k: int = K) -> list[str]:
    """
    Returns: 注册证号列表，按 S→A→C + 蛋白质含量 降序排列
    """
    route_a = RouteA()
    query   = parse_query(tc)
    query.raw_input['diseases'] = tc.get('diseases', [])
    fr      = route_a.run(df, query)

    if len(fr.pool) == 0:
        return []

    pool = fr.pool.copy()
    pool['_tier'] = 2   # C 默认
    for cond, tier in fr.tiers.items():
        for idx in tier.get('S', []):
            if idx in pool.index: pool.loc[idx, '_tier'] = 0
        for idx in tier.get('A', []):
            if idx in pool.index and pool.loc[idx, '_tier'] != 0:
                pool.loc[idx, '_tier'] = 1

    pool['蛋白质_g'] = pool.get('蛋白质_g', pd.Series(0.0, index=pool.index)).fillna(0)
    pool = pool.sort_values(['_tier', '蛋白质_g'], ascending=[True, False])
    return pool['注册证号'].tolist()[:k]


# ══════════════════════════════════════════════════════════════════
# 推荐执行：Route A+B（CBF量化排序，需MindSpore）
# ══════════════════════════════════════════════════════════════════
def run_route_ab(df: pd.DataFrame, tc: dict,
                 route_b: "RouteB_CBF", k: int = K) -> list[str]:
    """
    Returns: 注册证号列表，按 A级别×CBF得分 降序排列
    """
    route_a = RouteA()
    query   = parse_query(tc)
    query.raw_input['diseases'] = tc.get('diseases', [])
    fr      = route_a.run(df, query)

    if len(fr.pool) == 0:
        return []

    pool_indices = [df.index.get_loc(i) for i in fr.pool.index if i in df.index]
    cbf_scores   = route_b.score(pool_indices, query.conditions)

    pool = fr.pool.copy()
    pool['_tier'] = 2
    for cond, tier in fr.tiers.items():
        for idx in tier.get('S', []):
            if idx in pool.index: pool.loc[idx, '_tier'] = 0
        for idx in tier.get('A', []):
            if idx in pool.index and pool.loc[idx, '_tier'] != 0:
                pool.loc[idx, '_tier'] = 1

    pool['_cbf'] = 0.0
    for li, idx in enumerate(fr.pool.index):
        if li < len(cbf_scores):
            pool.loc[idx, '_cbf'] = float(cbf_scores[li])

    tier_w = REC_CFG.tier_weights
    tier_map = {0: tier_w['S'], 1: tier_w['A'], 2: tier_w.get('C', 0.5)}
    pool['_score'] = pool['_cbf'] * pool['_tier'].map(tier_map)
    pool = pool.sort_values('_score', ascending=False)
    return pool['注册证号'].tolist()[:k]


# ══════════════════════════════════════════════════════════════════
# 推荐执行：Route A+B+C（LLM 解释，返回推荐列表 + 解释文本）
# ══════════════════════════════════════════════════════════════════
def run_route_abc(df: pd.DataFrame, tc: dict,
                  route_b: "RouteB_CBF", k: int = K) -> tuple[list[str], str]:
    """
    Returns: (注册证号列表, explanation_text)
    Route C 对 A+B Top-10 候选做 LLM 重排序，返回新顺序前 K 个
    """
    from route_c import RouteC

    route_a = RouteA()
    query   = parse_query(tc)
    query.raw_input['diseases'] = tc.get('diseases', [])
    fr      = route_a.run(df, query)

    if len(fr.pool) == 0:
        return [], ""

    pool_indices = [df.index.get_loc(i) for i in fr.pool.index if i in df.index]
    cbf_scores   = route_b.score(pool_indices, query.conditions)

    pool = fr.pool.copy()
    pool['_tier'] = 2
    pool['A级别'] = 'C'
    for cond, tier in fr.tiers.items():
        for idx in tier.get('S', []):
            if idx in pool.index:
                pool.loc[idx, '_tier'] = 0
                pool.loc[idx, 'A级别'] = 'S'
        for idx in tier.get('A', []):
            if idx in pool.index and pool.loc[idx, '_tier'] != 0:
                pool.loc[idx, '_tier'] = 1
                pool.loc[idx, 'A级别'] = 'A'

    pool['CBF得分'] = 0.0
    for li, idx in enumerate(fr.pool.index):
        if li < len(cbf_scores):
            pool.loc[idx, 'CBF得分'] = float(cbf_scores[li])

    tier_w = REC_CFG.tier_weights
    tier_map = {0: tier_w['S'], 1: tier_w['A'], 2: tier_w.get('C', 0.5)}
    pool['综合得分'] = pool['CBF得分'] * pool['_tier'].map(tier_map)
    ranked = pool.sort_values('综合得分', ascending=False).reset_index(drop=True)
    ranked.index += 1

    route_c = RouteC(llm_cfg=LLM_CFG)

    # ── LLM 重排序：Route C 核心价值——改变产品顺序 ──────────────
    reranked_ids = route_c.rerank(tc, ranked, fr, top_n=min(10, len(ranked)))

    # ── 生成解释文本（供 LLM-as-Judge 评分）─────────────────────
    llm_out = route_c.run(tc, ranked, fr)
    explanation = ""
    if llm_out:
        parts    = [f"推荐产品：{r.name}（{r.reason}）"
                    for r in getattr(llm_out, 'recommendations', [])]
        warnings = getattr(llm_out, 'clinical_warnings', [])
        explanation = "\n".join(parts + warnings)

    return reranked_ids[:k], explanation


# ══════════════════════════════════════════════════════════════════
# ④ LLM-as-Judge（解释质量评分）
# ══════════════════════════════════════════════════════════════════
JUDGE_SYSTEM = """\
你是一位临床营养学评审专家。
请对推荐系统给出的解释文本进行质量评分，严格按 JSON 格式输出，不要输出任何其他文字。"""

def llm_judge(explanation: str, tc: dict) -> Optional[dict]:
    """
    对 Route C 的解释文本进行 LLM-as-Judge 评分
    Returns: {"accuracy": 1-5, "relevance": 1-5, "completeness": 1-5, "professionalism": 1-5}
    """
    if not explanation or not LLM_CFG.api_key:
        return None

    user_prompt = f"""\
客户信息：
  病症：{tc.get('diseases')}
  年龄：{tc.get('age_group')}
  描述：{tc.get('query', '')}

系统推荐解释：
{explanation}

请从以下 4 个维度为该解释打分（每项 1~5 分，5 分最优）：
1. accuracy       — 临床依据引用的准确性（是否引用了正确标准，有无编造）
2. relevance      — 推荐理由与病症的相关性（是否针对该病症给出了有意义的理由）
3. completeness   — 警示/禁忌信息的完整性（是否提及了关键禁忌）
4. professionalism — 语言专业性（是否使用规范临床术语）

输出格式（纯 JSON，无 markdown）：
{{"accuracy": N, "relevance": N, "completeness": N, "professionalism": N, "comment": "简短总评（≤30字）"}}"""

    try:
        provider = get_provider(LLM_CFG)
        raw      = provider.generate_text(JUDGE_SYSTEM, user_prompt)
        if not raw:
            return None
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"LLM Judge 调用失败: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# 消融评估主流程
# ══════════════════════════════════════════════════════════════════
@dataclass
class CaseMetrics:
    condition   : str
    config      : str
    precision   : float
    recall      : float
    ndcg        : float
    safety_rate : float
    kendall_tau : Optional[float]
    # LLM judge（仅 A+B+C 配置）
    judge_accuracy      : Optional[float] = None
    judge_relevance     : Optional[float] = None
    judge_completeness  : Optional[float] = None
    judge_professionalism: Optional[float] = None
    judge_comment       : Optional[str]  = None


def run_ablation(k: int = K) -> pd.DataFrame:
    """
    对所有测试案例运行三路消融，返回指标 DataFrame
    """
    df = load_existing_data(str(RESULT1_PATH), str(RESULT2_PATH))
    gt = load_ground_truth()

    # 若 MindSpore 可用，提前构建 Route B（特征矩阵只需一次）
    route_b = None
    if HAS_MS:
        try:
            kb_dict = {k: v.__dict__ for k, v in CLINICAL_KB.items()}
            route_b = RouteB_CBF(
                df_full     = df,
                clinical_kb = kb_dict,
                mode        = 'base',
                cfg         = REC_CFG,
            )
        except Exception as e:
            logger.warning(f"Route B 初始化失败: {e}")

    rows: list[CaseMetrics] = []

    for tc in TEST_CASES:
        cond_key = tc["name"]
        if cond_key not in gt:
            logger.warning(f"GT 中找不到: {cond_key}，跳过")
            continue

        s_set   = gt[cond_key]["S"]
        a_set   = gt[cond_key]["A"]
        ban_set = gt[cond_key]["ban"]

        logger.info(f"\n{'─'*52}")
        logger.info(f"病症: {cond_key}  |  GT: {len(s_set)}S  {len(a_set)}A  {len(ban_set)}ban")

        # ── Route A ──────────────────────────────────────────
        recs_a = run_route_a(df, tc, k=k)
        rows.append(CaseMetrics(
            condition   = cond_key,
            config      = "A",
            precision   = precision_at_k(recs_a, s_set, k),
            recall      = recall_at_k(recs_a, s_set, k),
            ndcg        = ndcg_at_k(recs_a, s_set, a_set, k),
            safety_rate = safety_violation_rate(recs_a, ban_set),
            kendall_tau = kendall_tau_score(recs_a, s_set, a_set),
        ))
        logger.info(f"  [A]     P@{k}={rows[-1].precision:.3f}  "
                    f"R@{k}={rows[-1].recall:.3f}  "
                    f"NDCG@{k}={rows[-1].ndcg:.3f}  "
                    f"Safety={rows[-1].safety_rate:.3f}  "
                    f"τ={rows[-1].kendall_tau}")

        # ── Route A+B ─────────────────────────────────────────
        if route_b is not None:
            recs_ab = run_route_ab(df, tc, route_b, k=k)
            rows.append(CaseMetrics(
                condition   = cond_key,
                config      = "A+B",
                precision   = precision_at_k(recs_ab, s_set, k),
                recall      = recall_at_k(recs_ab, s_set, k),
                ndcg        = ndcg_at_k(recs_ab, s_set, a_set, k),
                safety_rate = safety_violation_rate(recs_ab, ban_set),
                kendall_tau = kendall_tau_score(recs_ab, s_set, a_set),
            ))
            logger.info(f"  [A+B]   P@{k}={rows[-1].precision:.3f}  "
                        f"R@{k}={rows[-1].recall:.3f}  "
                        f"NDCG@{k}={rows[-1].ndcg:.3f}  "
                        f"Safety={rows[-1].safety_rate:.3f}  "
                        f"τ={rows[-1].kendall_tau}")

        # ── Route A+B+C ───────────────────────────────────────
        if route_b is not None and LLM_CFG.api_key:
            try:
                recs_abc, explanation = run_route_abc(df, tc, route_b, k=k)
            except Exception as e:
                logger.warning(f"Route C 失败: {e}")
                recs_abc, explanation = recs_ab if route_b else recs_a, ""

            judge = llm_judge(explanation, tc)
            rows.append(CaseMetrics(
                condition              = cond_key,
                config                 = "A+B+C",
                precision              = precision_at_k(recs_abc, s_set, k),
                recall                 = recall_at_k(recs_abc, s_set, k),
                ndcg                   = ndcg_at_k(recs_abc, s_set, a_set, k),
                safety_rate            = safety_violation_rate(recs_abc, ban_set),
                kendall_tau            = kendall_tau_score(recs_abc, s_set, a_set),
                judge_accuracy         = judge.get("accuracy")       if judge else None,
                judge_relevance        = judge.get("relevance")      if judge else None,
                judge_completeness     = judge.get("completeness")   if judge else None,
                judge_professionalism  = judge.get("professionalism") if judge else None,
                judge_comment          = judge.get("comment")         if judge else None,
            ))
            logger.info(f"  [A+B+C] P@{k}={rows[-1].precision:.3f}  "
                        f"R@{k}={rows[-1].recall:.3f}  "
                        f"NDCG@{k}={rows[-1].ndcg:.3f}  "
                        f"Judge={judge}")

    result_df = pd.DataFrame([asdict(r) for r in rows])
    # 保存原始结果
    result_df.to_csv(OUT_DIR / "ablation_results.csv",
                     index=False, encoding="utf-8-sig")
    logger.info(f"\n原始结果已保存 → outputs/ablation_results.csv")
    return result_df


# ══════════════════════════════════════════════════════════════════
# 汇总报告打印
# ══════════════════════════════════════════════════════════════════
def print_summary(result_df: pd.DataFrame, k: int = K) -> None:
    if result_df.empty:
        print("无评估结果")
        return

    configs = result_df["config"].unique()
    metrics = ["precision", "recall", "ndcg", "safety_rate", "kendall_tau"]

    print("\n" + "═" * 72)
    print(f"  消融实验汇总报告  (K={k}, 病症数={result_df['condition'].nunique()})")
    print("═" * 72)

    # ── 指标均值对比表 ─────────────────────────────────────────────
    print(f"\n{'配置':<10} {'P@K':>8} {'R@K':>8} {'NDCG@K':>8} "
          f"{'安全违规':>10} {'Kendall τ':>12}")
    print("─" * 62)
    for cfg in configs:
        sub = result_df[result_df["config"] == cfg]
        p   = sub["precision"].mean()
        r   = sub["recall"].mean()
        n   = sub["ndcg"].mean()
        s   = sub["safety_rate"].mean()
        τ   = sub["kendall_tau"].dropna().mean()
        print(f"  {cfg:<8}  {p:>7.3f}  {r:>7.3f}  {n:>7.3f}  "
              f"{s:>9.3f}  {τ:>11.3f}")
    print("─" * 62)

    # ── LLM Judge 汇总（仅 A+B+C）────────────────────────────────
    abc = result_df[result_df["config"] == "A+B+C"]
    if not abc.empty and abc["judge_accuracy"].notna().any():
        print("\nLLM-as-Judge（Route C 解释质量，均分 / 满分5分）：")
        for dim in ["judge_accuracy", "judge_relevance",
                    "judge_completeness", "judge_professionalism"]:
            val = abc[dim].dropna().mean()
            label = dim.replace("judge_", "").capitalize()
            print(f"  {label:<18} {val:.2f}")

    # ── 各病症明细 ────────────────────────────────────────────────
    print("\n各病症明细：")
    for cond in result_df["condition"].unique():
        sub = result_df[result_df["condition"] == cond]
        print(f"\n  {cond}:")
        for _, row in sub.iterrows():
            cfg = row["config"]
            print(f"    [{cfg:<5}]  "
                  f"P={row['precision']:.2f}  R={row['recall']:.2f}  "
                  f"NDCG={row['ndcg']:.2f}  "
                  f"Safety={row['safety_rate']:.2f}  "
                  f"τ={row['kendall_tau'] if pd.notna(row['kendall_tau']) else 'N/A'}")

    print("\n" + "═" * 72)

    # 保存汇总
    summary = result_df.groupby("config")[metrics].mean().round(3)
    summary.to_csv(OUT_DIR / "ablation_summary.csv", encoding="utf-8-sig")
    logger.info("汇总结果已保存 → outputs/ablation_summary.csv")


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result_df = run_ablation(k=K)
    print_summary(result_df, k=K)
