"""
compare_models.py — 4 个 LLM 横向对比评估
指标：NDCG@5（客观）/ LLM-as-Judge 四维度（主观）/ 置信度 / 延迟
结果不互相覆盖：每模型单独存 compare_{model}.csv
汇总：outputs/model_comparison_summary.csv
图表：outputs/model_comparison_table.png
"""
from __future__ import annotations
import sys, time, json, re, os, warnings
from pathlib import Path
from typing import Optional
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from loguru import logger

from config import REC_CFG, LLMConfig, RESULT1_PATH, RESULT2_PATH
from schema import load_existing_data
from route_a import RouteA, parse_query, CLINICAL_KB
from llm_providers import get_provider
from route_c import RouteC
from evaluation import (
    load_ground_truth, TEST_CASES, K,
    ndcg_at_k, precision_at_k, recall_at_k,
    JUDGE_SYSTEM,
)

import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device('CPU')
from route_b_mindspore import RouteB_CBF

OUT_DIR = Path(__file__).parent / "outputs"

# ══════════════════════════════════════════════════════════════════
# 待评估模型（均使用硅基流动）
# ══════════════════════════════════════════════════════════════════
_API_KEY  = "sk-bfebkmzrimcietonbudesdcicfobxjdytszbrlcxzyymiwzd"
_BASE_URL = "https://api.siliconflow.cn/v1"

MODELS_TO_EVAL = [
    {"label": "DeepSeek-V4-Pro", "text_model": "deepseek-ai/DeepSeek-V4-Pro"},
    {"label": "GLM-5.1",         "text_model": "Pro/zai-org/GLM-5.1"},
    {"label": "MiniMax-M2.5",    "text_model": "MiniMaxAI/MiniMax-M2.5"},
    {"label": "Qwen2.5-72B",     "text_model": "Qwen/Qwen2.5-72B-Instruct"},
]

# 固定裁判模型（不参与被评测，保持评分一致性）
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _make_cfg(text_model: str) -> LLMConfig:
    return LLMConfig(
        provider    = "siliconflow",
        text_model  = text_model,
        model_name  = text_model,
        api_key     = _API_KEY,
        base_url    = _BASE_URL,
        max_tokens  = 2000,
        temperature = 0.1,
        timeout     = 120,
        retry_max   = 2,
        retry_delay = 5,
    )


# ══════════════════════════════════════════════════════════════════
# 预计算 A+B 排序（所有模型共用，只算一次）
# ══════════════════════════════════════════════════════════════════
def compute_ab_ranking(df, tc, route_b):
    route_a = RouteA()
    query = parse_query(tc)
    query.raw_input['diseases'] = tc.get('diseases', [])
    fr = route_a.run(df, query)

    if len(fr.pool) == 0:
        return None, fr

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

    tier_w   = REC_CFG.tier_weights
    tier_map = {0: tier_w['S'], 1: tier_w['A'], 2: tier_w.get('C', 0.5)}
    pool['综合得分'] = pool['CBF得分'] * pool['_tier'].map(tier_map)
    ranked = pool.sort_values('综合得分', ascending=False).reset_index(drop=True)
    ranked.index += 1
    return ranked, fr


# ══════════════════════════════════════════════════════════════════
# 固定裁判（在所有模型之间保持一致）
# ══════════════════════════════════════════════════════════════════
_judge_provider = get_provider(_make_cfg(JUDGE_MODEL))


def llm_judge(explanation: str, tc: dict) -> Optional[dict]:
    if not explanation:
        return None
    user_prompt = (
        f"客户信息：\n"
        f"  病症：{tc.get('diseases')}\n"
        f"  年龄：{tc.get('age_group')}\n"
        f"  描述：{tc.get('query', '')}\n\n"
        f"系统推荐解释：\n{explanation}\n\n"
        f"请从以下 4 个维度打分（每项 1~5 分，5 分最优）：\n"
        f"1. accuracy       — 临床依据引用的准确性\n"
        f"2. relevance      — 推荐理由与病症的相关性\n"
        f"3. completeness   — 警示/禁忌信息的完整性\n"
        f"4. professionalism — 语言专业性\n\n"
        f'输出格式（纯 JSON，无 markdown）：\n'
        f'{{"accuracy": N, "relevance": N, "completeness": N, '
        f'"professionalism": N, "comment": "简短总评（≤30字）"}}'
    )
    try:
        raw = _judge_provider.generate_text(JUDGE_SYSTEM, user_prompt)
        if not raw:
            return None
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Judge 失败: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# 单模型评估
# ══════════════════════════════════════════════════════════════════
def evaluate_model(model_info: dict, gt: dict,
                   ab_cache: dict) -> pd.DataFrame:
    label      = model_info["label"]
    text_model = model_info["text_model"]
    logger.info(f"\n{'═'*60}")
    logger.info(f"  评估模型: {label}  ({text_model})")
    logger.info(f"{'═'*60}")

    cfg     = _make_cfg(text_model)
    route_c = RouteC(llm_cfg=cfg)
    rows    = []

    for tc in TEST_CASES:
        cond_key = tc["name"]
        if cond_key not in gt:
            logger.warning(f"GT 中找不到 {cond_key}，跳过")
            continue

        s_set   = gt[cond_key]["S"]
        a_set   = gt[cond_key]["A"]
        ranked, fr = ab_cache[cond_key]
        if ranked is None:
            continue

        # ── LLM 重排序 + 解释（计时包含两次 API 调用）─────────
        t0 = time.perf_counter()
        try:
            reranked_ids = route_c.rerank(
                tc, ranked, fr, top_n=min(10, len(ranked)))
            llm_out = route_c.run(tc, ranked, fr)
        except Exception as e:
            logger.warning(f"[{label}] {cond_key} API 失败: {e}")
            reranked_ids = ranked['注册证号'].tolist()[:K]
            llm_out = None
        elapsed = time.perf_counter() - t0

        # ── 客观指标（对比 ground truth）──────────────────────
        recs_k = reranked_ids[:K]
        ndcg   = ndcg_at_k(recs_k, s_set, a_set, K)
        prec   = precision_at_k(recs_k, s_set, K)
        recall = recall_at_k(recs_k, s_set, K)

        # ── 解释文本 + 置信度 ──────────────────────────────────
        explanation = ""
        confidence  = "N/A"
        if llm_out:
            parts       = [f"推荐产品：{r.name}（{r.reason}）"
                           for r in getattr(llm_out, 'recommendations', [])]
            warns       = getattr(llm_out, 'clinical_warnings', [])
            explanation = "\n".join(parts + warns)
            confidence  = getattr(llm_out, 'confidence', 'N/A')

        # ── LLM-as-Judge（裁判固定为 Qwen2.5-72B）────────────
        judge = llm_judge(explanation, tc)

        rows.append({
            "model":           label,
            "condition":       cond_key,
            "ndcg":            round(ndcg,   4),
            "precision":       round(prec,   4),
            "recall":          round(recall, 4),
            "accuracy":        judge.get("accuracy")        if judge else None,
            "relevance":       judge.get("relevance")       if judge else None,
            "completeness":    judge.get("completeness")    if judge else None,
            "professionalism": judge.get("professionalism") if judge else None,
            "judge_comment":   judge.get("comment")         if judge else None,
            "confidence":      confidence,
            "elapsed_s":       round(elapsed, 2),
        })
        logger.info(
            f"  [{cond_key:12s}] "
            f"NDCG={ndcg:.3f}  "
            f"acc={rows[-1]['accuracy']}  rel={rows[-1]['relevance']}  "
            f"conf={confidence}  t={elapsed:.1f}s"
        )

    df_model = pd.DataFrame(rows)
    tag      = text_model.replace("/", "_").replace(".", "-")
    out_path = OUT_DIR / f"compare_{tag}.csv"
    df_model.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"  ✓ 逐案例结果 → {out_path.name}")
    return df_model


# ══════════════════════════════════════════════════════════════════
# 汇总表可视化
# ══════════════════════════════════════════════════════════════════
def plot_comparison_table(summary: pd.DataFrame):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    def fp(s=10, b=False):
        if font_path:
            return fm.FontProperties(fname=font_path, size=s,
                                     weight='bold' if b else 'normal')
        return fm.FontProperties(size=s, weight='bold' if b else 'normal')

    matplotlib.rcParams.update({
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
    })

    col_labels = ['模型', 'NDCG@5', '准确性', '相关性', '完整性', '专业性', '置信度', '延迟(s)']
    judge_dims = ['accuracy', 'relevance', 'completeness', 'professionalism']

    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row['model'],
            f"{row['ndcg']:.3f}",
            f"{row['accuracy']:.2f}"       if pd.notna(row.get('accuracy'))       else "N/A",
            f"{row['relevance']:.2f}"      if pd.notna(row.get('relevance'))      else "N/A",
            f"{row['completeness']:.2f}"   if pd.notna(row.get('completeness'))   else "N/A",
            f"{row['professionalism']:.2f}" if pd.notna(row.get('professionalism')) else "N/A",
            str(row.get('confidence', 'N/A')),
            f"{row['elapsed_s']:.1f}",
        ])

    n_rows = len(table_data)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(15, 2.5 + n_rows * 1.0))
    ax.axis('off')

    tbl = ax.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.5)

    # 表头
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor('#471365')
        cell.set_text_props(color='white', fontproperties=fp(11, True))

    # 找最佳 NDCG 行
    best_idx = int(summary['ndcg'].idxmax())

    for i in range(n_rows):
        bg = '#F0EBF8' if i % 2 == 0 else 'white'
        if i == best_idx:
            bg = '#D4EDDA'
        for j in range(n_cols):
            tbl[i + 1, j].set_facecolor(bg)
            tbl[i + 1, j].set_text_props(fontproperties=fp(10.5))

    ax.set_title(
        f'多模型横向对比  ·  NDCG@5（客观）+ LLM-as-Judge（主观，裁判: {JUDGE_MODEL}）\n'
        f'11个病症均值  |  绿色行 = NDCG最高  |  主观分满分5分',
        fontproperties=fp(12, True), pad=18
    )

    out = OUT_DIR / "model_comparison_table.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 对比图 → {out.name}")


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("加载数据...")
    df = load_existing_data(str(RESULT1_PATH), str(RESULT2_PATH))
    gt = load_ground_truth()

    logger.info("构建 Route B（特征矩阵，只算一次）...")
    kb_dict = {k: v.__dict__ for k, v in CLINICAL_KB.items()}
    route_b = RouteB_CBF(df_full=df, clinical_kb=kb_dict, mode='base', cfg=REC_CFG)

    logger.info("预计算所有病症的 A+B 排序（各模型共用）...")
    ab_cache: dict = {}
    for tc in TEST_CASES:
        ranked, fr = compute_ab_ranking(df, tc, route_b)
        ab_cache[tc["name"]] = (ranked, fr)
    logger.info(f"✓ 已缓存 {len(ab_cache)} 个病症")

    # ── 逐模型评估 ────────────────────────────────────────────────
    all_dfs = []
    for model_info in MODELS_TO_EVAL:
        df_m = evaluate_model(model_info, gt, ab_cache)
        all_dfs.append(df_m)

    # ── 合并全部逐案例结果 ────────────────────────────────────────
    all_results = pd.concat(all_dfs, ignore_index=True)
    all_results.to_csv(OUT_DIR / "model_comparison_all.csv",
                       index=False, encoding="utf-8-sig")
    logger.info("✓ 全部逐案例结果 → model_comparison_all.csv")

    # ── 汇总均值 ─────────────────────────────────────────────────
    num_cols = ["ndcg", "precision", "recall",
                "accuracy", "relevance", "completeness", "professionalism", "elapsed_s"]
    summary  = (all_results.groupby("model")[num_cols]
                .mean().round(3).reset_index())

    # 置信度取众数
    conf_mode = (all_results.groupby("model")["confidence"]
                 .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A")
                 .reset_index())
    summary = summary.merge(conf_mode, on="model")

    # 保持输入顺序
    order_map = {m["label"]: i for i, m in enumerate(MODELS_TO_EVAL)}
    summary["_o"] = summary["model"].map(order_map)
    summary = summary.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    summary.to_csv(OUT_DIR / "model_comparison_summary.csv",
                   index=False, encoding="utf-8-sig")

    # ── 打印汇总表 ─────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  多模型横向对比汇总（11个病症均值）")
    print("═" * 80)
    print(summary[["model", "ndcg", "accuracy", "relevance",
                    "completeness", "professionalism", "confidence", "elapsed_s"]]
          .to_string(index=False))
    print("═" * 80)

    # ── 可视化 ────────────────────────────────────────────────────
    plot_comparison_table(summary)

    logger.info("\n✓ 全部完成！")
    logger.info(f"  逐案例: outputs/compare_{{model}}.csv  (4个文件，不覆盖)")
    logger.info(f"  汇总:   outputs/model_comparison_summary.csv")
    logger.info(f"  图表:   outputs/model_comparison_table.png")
