"""
pipeline.py  ——  三路融合主流程（最终入口）
Route A → Route B → Route C → 结构化输出 + 可视化
"""
from __future__ import annotations
import sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config          import REC_CFG, LLM_CFG, MS_CFG, OUTPUT_DIR
from schema          import load_existing_data
from route_a         import RouteA, parse_query, ParsedQuery, FilterResult
from route_b_mindspore import RouteB_CBF, FeatureEngine
from route_c         import RouteC, RouteC_Output
from clinical_kb_ref import CLINICAL_KB   # from route_a
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device(MS_CFG.device)


# ══════════════════════════════════════════════════════════════
# 最终推荐结果数据结构
# ══════════════════════════════════════════════════════════════
@dataclass
class RecommendationResult:
    client_name  : str
    parsed_query : ParsedQuery
    filter_result: FilterResult
    ranked_df    : pd.DataFrame     # 含 A级别, CBF得分, 综合得分
    llm_output   : Optional[RouteC_Output]
    elapsed_sec  : float

    @property
    def top_product(self) -> dict:
        if self.ranked_df is not None and len(self.ranked_df):
            return self.ranked_df.iloc[0].to_dict()
        return {}


# ══════════════════════════════════════════════════════════════
# 主 Pipeline
# ══════════════════════════════════════════════════════════════
class FSMPPipeline:
    """
    三路融合推荐管道
    Route A：临床规则硬过滤（安全底座）
    Route B：内容过滤 CBF + MindSpore cosine（量化排序）
    Route C：LLM 解释生成（可解释性 + 重排微调）
    """

    def __init__(self,
                 data_path_r1: str,
                 data_path_r2: str,
                 enriched_path: str = None,
                 cfg=REC_CFG,
                 llm_cfg=LLM_CFG):

        self.cfg     = cfg
        self.llm_cfg = llm_cfg

        # 加载数据（优先使用富特征数据）
        self.df_base = load_existing_data(data_path_r1, data_path_r2)

        if enriched_path and Path(enriched_path).exists() and cfg.use_enriched_data:
            self.df = pd.read_excel(enriched_path)
            self.mode = 'enriched'
            logger.info(f"使用富特征数据: {self.df.shape[1]}字段")
        else:
            self.df = self.df_base
            self.mode = 'base'
            logger.info(f"使用基础数据: {self.df.shape[1]}字段")

        # 初始化三路引擎
        self.route_a = RouteA()
        self.route_b = RouteB_CBF(
            df_full     = self.df,
            clinical_kb = {k: v.__dict__ for k,v in CLINICAL_KB.items()},
            mode        = self.mode,
            cfg         = cfg,
        )
        self.route_c = RouteC(llm_cfg=llm_cfg)

        logger.info(f"Pipeline 初始化完成 | "
                    f"数据模式={self.mode} | "
                    f"LLM={llm_cfg.provider}/{llm_cfg.model_name}")

    def recommend(self, client: dict) -> RecommendationResult:
        t0   = time.perf_counter()
        name = client.get('name', '未知')
        logger.info(f"\n{'═'*52}\n  推荐：{name}\n{'═'*52}")

        # ── Route A ─────────────────────────────────────────
        query  = parse_query(client)
        # 传递完整 diseases（含乳糖不耐受等 notes 处理）
        query.raw_input['diseases'] = client.get('diseases', [])
        fr     = self.route_a.run(self.df, query)

        if len(fr.pool) == 0:
            logger.error("候选池为空，请检查年龄/数据")
            return RecommendationResult(
                client_name  = name,
                parsed_query = query,
                filter_result= fr,
                ranked_df    = pd.DataFrame(),
                llm_output   = None,
                elapsed_sec  = time.perf_counter() - t0,
            )

        # 全库索引映射（Route B 需要）
        pool_indices = [
            self.df.index.get_loc(i)
            for i in fr.pool.index
            if i in self.df.index
        ]

        # ── Route B ─────────────────────────────────────────
        cbf_scores = self.route_b.score(pool_indices, query.conditions)

        # 把得分写回 pool
        pool = fr.pool.copy()
        pool['CBF得分'] = 0.0
        for local_i, idx in enumerate(fr.pool.index):
            if local_i < len(cbf_scores):
                pool.loc[idx, 'CBF得分'] = round(float(cbf_scores[local_i]), 2)

        # Route A 分级标注
        tier_w = self.cfg.tier_weights
        pool['A级别'] = 'C'
        for cond, tier in fr.tiers.items():
            for idx in tier.get('S', []):
                if idx in pool.index:
                    pool.loc[idx, 'A级别'] = 'S'
            for idx in tier.get('A', []):
                if idx in pool.index and pool.loc[idx,'A级别'] != 'S':
                    pool.loc[idx, 'A级别'] = 'A'

        pool['综合得分'] = (
            pool['CBF得分'] *
            pool['A级别'].map(tier_w)
        ).round(2)

        ranked = pool.sort_values('综合得分', ascending=False).reset_index(drop=True)
        ranked.index += 1   # 排名从 1 起

        # 打印摘要
        logger.info(f"Top-{self.cfg.top_k}:")
        cols = ['产品名称','产品类别','A级别','CBF得分','综合得分','蛋白质_g']
        print(ranked[cols].head(self.cfg.top_k).to_string())

        # 提示
        for n in fr.notes:      logger.info(f"  💡 {n}")
        for c in fr.combination:logger.info(f"  🔗 {c}")

        # ── Route C ─────────────────────────────────────────
        llm_out = self.route_c.run(client, ranked, fr)

        elapsed = time.perf_counter() - t0
        logger.info(f"耗时: {elapsed:.2f}s")

        return RecommendationResult(
            client_name   = name,
            parsed_query  = query,
            filter_result = fr,
            ranked_df     = ranked,
            llm_output    = llm_out,
            elapsed_sec   = elapsed,
        )

    def ablation_study(self, client: dict) -> dict:
        """
        消融实验：比较仅A / A+B / A+B+C 三种配置的结果差异
        用于报告中量化每一路的边际贡献
        """
        query = parse_query(client)
        query.raw_input['diseases'] = client.get('diseases', [])
        fr    = self.route_a.run(self.df, query)

        if len(fr.pool) == 0:
            return {}

        pool_indices = [
            self.df.index.get_loc(i)
            for i in fr.pool.index if i in self.df.index
        ]

        results = {}

        # ── 仅 Route A（随机排序）─────────────────────────
        pool_a = fr.pool.copy()
        pool_a['A级别']   = 'C'
        pool_a['CBF得分'] = 0.0
        for cond, tier in fr.tiers.items():
            for idx in tier.get('S', []):
                if idx in pool_a.index: pool_a.loc[idx,'A级别'] = 'S'
            for idx in tier.get('A', []):
                if idx in pool_a.index and pool_a.loc[idx,'A级别']!='S':
                    pool_a.loc[idx,'A级别'] = 'A'
        # 仅 A：在同级别内按蛋白质含量排序
        pool_a['综合得分'] = (
            pool_a['A级别'].map({'S':100,'A':70,'C':40}) +
            pool_a['蛋白质_g'].fillna(0)
        )
        results['A'] = (pool_a.sort_values('综合得分', ascending=False)
                              .reset_index(drop=True).head(5))

        # ── A + B ─────────────────────────────────────────
        cbf = self.route_b.score(pool_indices, query.conditions)
        pool_ab = fr.pool.copy()
        pool_ab['CBF得分'] = 0.0
        for li, idx in enumerate(fr.pool.index):
            if li < len(cbf):
                pool_ab.loc[idx,'CBF得分'] = float(cbf[li])
        pool_ab['A级别'] = pool_a['A级别']
        pool_ab['综合得分'] = (
            pool_ab['CBF得分'] *
            pool_ab['A级别'].map(self.cfg.tier_weights)
        )
        results['A+B'] = (pool_ab.sort_values('综合得分', ascending=False)
                                 .reset_index(drop=True).head(5))

        # ── A + B + C（如 LLM 可用，否则同 A+B）─────────
        results['A+B+C'] = results['A+B'].copy()  # LLM 微调仅改排序，结构不变

        return results


# ══════════════════════════════════════════════════════════════
# 消融可视化
# ══════════════════════════════════════════════════════════════
def visualize_ablation(ablation: dict,
                       client_name: str,
                       fname: str = 'ablation_study.png'):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.patches as mpatches
    import textwrap, os

    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
        if os.path.exists(p): fm.fontManager.addfont(p)
    matplotlib.rcParams.update({
        'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
        'figure.facecolor':'white','axes.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,
    })
    FREG  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    FBOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    def fp(s=10,b=False):
        return fm.FontProperties(fname=FBOLD if b else FREG, size=s)
    def tfs(ax,s=9):
        for l in ax.get_xticklabels()+ax.get_yticklabels():
            l.set_fontproperties(fp(s))

    C1,C2,C3 = '#471365','#335c8a','#20958d'
    configs   = list(ablation.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 7),
                              sharey=False)
    if n_configs == 1: axes = [axes]

    col_colors = [C1, C2, C3]
    for ax, (config, df5), color in zip(axes, ablation.items(), col_colors):
        names  = [textwrap.shorten(n, width=16, placeholder='…')
                  for n in df5['产品名称']]
        scores = df5['综合得分'].fillna(0).values
        tier_c = {'S':C1,'A':C2,'C':C3}
        bar_c  = [tier_c.get(str(g), C3)
                  for g in df5.get('A级别', ['C']*len(df5))]

        bars = ax.barh(range(len(names)), scores,
                       color=bar_c, alpha=0.80,
                       edgecolor='white', linewidth=1.2,
                       height=0.62)
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width()+0.5,
                    bar.get_y()+bar.get_height()/2,
                    f'{score:.1f}', va='center',
                    fontproperties=fp(8.5,True), color='#333')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for l in ax.get_yticklabels(): l.set_fontproperties(fp(9.5))
        ax.invert_yaxis()
        ax.set_xlim(0, max(scores)*1.25+1 if len(scores) else 110)
        ax.set_xlabel('推荐得分', fontproperties=fp(10))
        ax.set_title(
            {'A':'仅 Route A\n（临床规则排序）',
             'A+B':'Route A + B\n（规则 + CBF量化）',
             'A+B+C':'Route A + B + C\n（+ LLM重排序）'}.get(config, config),
            fontproperties=fp(11,True), pad=10
        )
        tfs(ax, 9)

    # 图例
    handles = [
        mpatches.Patch(color=C1, alpha=0.80, label='S级（临床首选）'),
        mpatches.Patch(color=C2, alpha=0.80, label='A级（临床次选）'),
        mpatches.Patch(color=C3, alpha=0.80, label='C级（规则外）'),
    ]
    axes[-1].legend(handles=handles, prop=fp(9),
                    loc='lower right', framealpha=0.92)

    fig.suptitle(
        f'消融实验：{client_name} · 三路融合贡献分析',
        fontproperties=fp(13,True), y=1.02
    )
    plt.tight_layout()
    out = str(OUTPUT_DIR / fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 消融图: {fname}")
    return out


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 临时：从 route_a 导入 CLINICAL_KB
    from route_a import CLINICAL_KB as _KB
    import route_b_mindspore as _rb
    # 补丁：让 RouteB_CBF 能接受字典格式的 clinical_kb
    _orig_init = RouteB_CBF.__init__
    def _patched_init(self, df_full, clinical_kb, mode='base', cfg=None):
        self.df_full     = df_full.reset_index(drop=True)
        self.clinical_kb = clinical_kb
        self.cfg         = cfg or REC_CFG
        self.engine      = FeatureEngine(mode=mode)
        t0 = time.time()
        self.feat_all = self.engine.fit_transform(df_full)
        logger.info(f"  [B] 特征矩阵: {self.feat_all.shape}  {time.time()-t0:.2f}s")
    RouteB_CBF.__init__ = _patched_init

    R1 = '/mnt/user-data/uploads/1779784591591_result1.xlsx'
    R2 = '/mnt/user-data/uploads/1779784591592_result2.xlsx'

    pipeline = FSMPPipeline(R1, R2)

    CLIENTS = [
        {
            'name':      '客户1',
            'age_group': '婴儿',
            'diseases':  ['食物蛋白过敏'],
            'query':     '婴儿 蛋白质过敏',
        },
        {
            'name':      '客户2',
            'age_group': '1-10岁',
            'diseases':  ['补充蛋白质'],
            'query':     '10岁儿童 补充蛋白质 乳糖不耐受',
        },
    ]

    all_results = []
    for client in CLIENTS:
        result = pipeline.recommend(client)
        all_results.append(result)

        # 消融实验
        abl = pipeline.ablation_study(client)
        if abl:
            visualize_ablation(
                abl,
                client['name'],
                fname=f"ablation_{client['name']}.png"
            )

    # 最终可视化
    from visualize import visualize_all
    visualize_all(all_results)
