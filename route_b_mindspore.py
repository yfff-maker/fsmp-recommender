"""
route_b_mindspore.py  ——  Person A 核心任务
内容过滤（CBF）推荐引擎，MindSpore 张量加速版
对标华为 ModelArts / Atlas 部署标准
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops

sys.path.insert(0, str(Path(__file__).parent))
from config import MS_CFG, REC_CFG

# ── MindSpore 环境初始化 ──────────────────────────────────────
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device(MS_CFG.device)
print(f"[MindSpore] 版本: {ms.__version__}  设备: {MS_CFG.device}")


# ══════════════════════════════════════════════════════════════
# 特征工程
# ══════════════════════════════════════════════════════════════
class FeatureEngine:
    """
    将产品结构化数据转换为特征矩阵
    支持两种数据模式：
      mode='base'      仅用 result1/2 原始 8 个营养字段
      mode='enriched'  用 PDF 提取后的 50+ 字段（效果更好）
    """

    # ── 基础营养字段（result1）──────────────────────────────
    BASE_NUT_COLS = [
        '能量_kJ','蛋白质_g','脂肪_g','碳水_g',
        '钠_mg','氯_mg','钾_mg','磷_mg'
    ]

    # ── 富特征字段（PDF 提取后）──────────────────────────────
    ENRICHED_NUT_COLS = BASE_NUT_COLS + [
        'nut_linoleic_acid_g','nut_ala_mg',
        'nut_calcium_mg','nut_iron_mg','nut_zinc_mg',
        'nut_vit_a_ug','nut_vit_d_ug','nut_vit_c_mg',
        'nut_choline_mg','nut_taurine_mg','nut_carnitine_mg',
        'nut_dietary_fiber_g',
    ]

    # ── 类别特征权重（临床重要性） ──────────────────────────
    CAT_WEIGHT  = 2.5    # 类别 one-hot
    NUT_WEIGHT  = 1.5    # 营养成分
    POP_WEIGHT  = 2.0    # 适用人群类别
    SRC_WEIGHT  = 0.3    # 产品来源（弱特征）

    def __init__(self, mode: str = 'base'):
        self.mode    = mode
        self.nut_cols= self.ENRICHED_NUT_COLS if mode == 'enriched' \
                       else self.BASE_NUT_COLS
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tensor:
        """
        构建产品特征矩阵
        返回：MindSpore Tensor [N_products × N_features]
        """
        # ── 1. 营养成分（Z-score 标准化）──────────────────
        available = [c for c in self.nut_cols if c in df.columns]
        nut_np = df[available].fillna(0).values.astype(np.float32)
        nut_t  = Tensor(nut_np)
        # MindSpore Z-score
        mean   = nut_t.mean(axis=0, keep_dims=True)
        std = nut_t.std(axis=0, keepdims=True) + 1e-8
        nut_z  = (nut_t - mean) / std

        # ── 2. 产品类别 one-hot ────────────────────────────
        cat_dummies = pd.get_dummies(df['产品类别'], prefix='cat').astype(np.float32)
        cat_t  = Tensor(cat_dummies.values)

        # ── 3. 适用人群类别 one-hot ────────────────────────
        pop_dummies = pd.get_dummies(df['适用人群类别'], prefix='pop').astype(np.float32)
        pop_t  = Tensor(pop_dummies.values)

        # ── 4. 产品来源 one-hot ────────────────────────────
        src_dummies = pd.get_dummies(df['产品来源'], prefix='src').astype(np.float32)
        src_t  = Tensor(src_dummies.values)

        # ── 5. 是否无乳糖（PDF 提取后可用）────────────────
        if 'lactose_free' in df.columns and self.mode == 'enriched':
            lf = df['lactose_free'].fillna(False).astype(np.float32).values
            lf_t = Tensor(lf.reshape(-1, 1)) * 2.0  # 强特征
        else:
            lf_t = Tensor(np.zeros((len(df), 1), dtype=np.float32))

        # ── 6. 可作为单一营养来源（PDF 提取后可用）────────
        if 'single_source_ok' in df.columns and self.mode == 'enriched':
            ss = df['single_source_ok'].fillna(False).astype(np.float32).values
            ss_t = Tensor(ss.reshape(-1, 1)) * 1.5
        else:
            ss_t = Tensor(np.zeros((len(df), 1), dtype=np.float32))

        # ── 7. 加权拼接 ────────────────────────────────────
        feat = ops.cat([
            nut_z  * self.NUT_WEIGHT,
            cat_t  * self.CAT_WEIGHT,
            pop_t  * self.POP_WEIGHT,
            src_t  * self.SRC_WEIGHT,
            lf_t,
            ss_t,
        ], axis=1)

        # 保存元数据（用于 query 向量构建）
        self._cat_cols = cat_dummies.columns.tolist()
        self._pop_cols = pop_dummies.columns.tolist()
        self._src_cols = src_dummies.columns.tolist()
        self._nut_cols = available
        self._feat_dim = feat.shape[1]
        self._fitted   = True

        print(f"  [FeatureEngine] 特征维度: {feat.shape[1]} "
              f"(营养{len(available)} + 类别{len(self._cat_cols)} "
              f"+ 人群{len(self._pop_cols)} + 其他{self._feat_dim - len(available) - len(self._cat_cols) - len(self._pop_cols)})")
        return feat

    def build_query_vector(self,
                           diseases: list,
                           clinical_kb: dict) -> Tensor:
        """
        将用户需求（病症列表）转为查询特征向量
        与产品矩阵同维度，可直接做 cosine similarity
        """
        assert self._fitted, "请先调用 fit_transform()"

        # 病症 → 营养方向（基于临床知识）
        NUT_DIRECTION = {
            '食物蛋白过敏':  {'蛋白质_g': 0.0,  '碳水_g': 0.0, '能量_kJ': 1.5},
            '乳糖不耐受':    {'碳水_g': -2.0},
            '早产':          {'能量_kJ': 2.0, '蛋白质_g': 2.0, '脂肪_g': 1.5},
            '补充蛋白质':    {'蛋白质_g': 2.5, '碳水_g': -1.5, '脂肪_g': -1.0},
            '消化吸收障碍':  {'能量_kJ': 1.5, '蛋白质_g': 1.5},
            '肿瘤':          {'能量_kJ': 2.0, '蛋白质_g': 2.0},
            '腹泻脱水':      {'钠_mg': 2.0, '钾_mg': 2.0, '磷_mg': 1.0},
            '吞咽障碍':      {'能量_kJ': 1.0},
        }
        # 病症 → 首选类别
        CAT_PRIORITY = {
            '食物蛋白过敏':  ['氨基酸配方'],
            '乳糖不耐受':    ['无乳糖配方', '低乳糖配方'],
            '早产':          ['早产/低出生体重婴儿配方'],
            '补充蛋白质':    ['蛋白质（氨基酸）组件'],
            '消化吸收障碍':  ['全营养配方食品'],
            '肿瘤':          ['特定全营养配方食品', '全营养配方食品'],
            '腹泻脱水':      ['电解质配方'],
            '吞咽障碍':      ['增稠组件'],
        }

        q_np = np.zeros(self._feat_dim, dtype=np.float32)
        nut_dim  = len(self._nut_cols)
        cat_dim  = len(self._cat_cols)
        pop_dim  = len(self._pop_cols)
        src_dim  = len(self._src_cols)

        for dis in diseases:
            # 营养方向
            nd = NUT_DIRECTION.get(dis, {})
            for i, col in enumerate(self._nut_cols):
                base = col.replace('nut_', '')  # 对齐字段名
                q_np[i] += nd.get(base, 0.0) * self.NUT_WEIGHT

            # 类别偏好
            pref_cats = CAT_PRIORITY.get(dis, [])
            for j, col in enumerate(self._cat_cols):
                for pc in pref_cats:
                    if pc in col or col.endswith(pc.replace('/','_')):
                        q_np[nut_dim + j] += 3.0 * self.CAT_WEIGHT

            # 人群类别
            age = clinical_kb.get(dis, {}).get('年龄', '')
            for j, col in enumerate(self._pop_cols):
                if (age == '婴儿' and '婴' in col) or \
                   (age in ('1岁以上','1-10岁') and '以上' in col):
                    q_np[nut_dim + cat_dim + j] += 2.0 * self.POP_WEIGHT

        q_t = Tensor(q_np.reshape(1, -1))
        return q_t


# ══════════════════════════════════════════════════════════════
# MindSpore Cosine Similarity（替代 sklearn）
# ══════════════════════════════════════════════════════════════
def ms_cosine_similarity(query: Tensor, matrix: Tensor) -> Tensor:
    """
    MindSpore 实现的 cosine similarity
    query:  [1 × D]
    matrix: [N × D]
    return: [N]
    """
    # L2 归一化
    q_norm = query / (ops.norm(query, dim=1, keepdim=True) + 1e-8)
    m_norm = matrix / (ops.norm(matrix, dim=1, keepdim=True) + 1e-8)
    # 矩阵乘法：[1×D] @ [D×N] → [1×N] → [N]
    sims = ops.matmul(q_norm, m_norm.T).squeeze(0)
    return sims


# ══════════════════════════════════════════════════════════════
# CBF 推荐核心
# ══════════════════════════════════════════════════════════════
class RouteB_CBF:
    """
    内容过滤推荐引擎（MindSpore 加速）
    在 Route A 候选集内做细粒度排序
    """
    def __init__(self, df_full: pd.DataFrame,
                 clinical_kb: dict,
                 mode: str = 'base',
                 cfg=REC_CFG):
        self.df_full     = df_full.reset_index(drop=True)
        self.clinical_kb = clinical_kb
        self.cfg         = cfg
        self.engine      = FeatureEngine(mode=mode)

        # 预计算全库特征矩阵
        t0 = time.time()
        self.feat_all = self.engine.fit_transform(df_full)
        print(f"  [B] 特征矩阵构建耗时: {time.time()-t0:.2f}s")

    def score(self, pool_indices: list,
              diseases: list) -> np.ndarray:
        """
        对候选集打分
        pool_indices: Route A 返回的产品索引列表
        diseases:     用户病症列表
        返回：normalized 得分数组 [0, 100]
        """
        if not pool_indices:
            return np.array([])

        # 构建查询向量
        q_vec  = self.engine.build_query_vector(diseases, self.clinical_kb)

        # 候选产品特征矩阵（从全库切片）
        feat_pool = self.feat_all[pool_indices]

        # MindSpore cosine similarity
        sims = ms_cosine_similarity(q_vec, feat_pool)
        sims_np = sims.asnumpy()

        # 归一化到 [0, 100]
        s_min, s_max = sims_np.min(), sims_np.max()
        if s_max > s_min:
            sims_norm = (sims_np - s_min) / (s_max - s_min) * 100
        else:
            sims_norm = np.full_like(sims_np, 50.0)

        return sims_norm.astype(float)


# ══════════════════════════════════════════════════════════════
# 消融实验：base vs enriched 对比
# ══════════════════════════════════════════════════════════════
def ablation_study(df_base: pd.DataFrame,
                   df_enriched: pd.DataFrame,
                   clinical_kb: dict,
                   test_disease: str = '补充蛋白质'):
    """
    对比两种特征模式的推荐差异
    用于报告中展示 PDF 提取的价值
    """
    from config import REC_CFG
    print(f"\n{'='*50}")
    print(f"消融实验: '{test_disease}'")
    print(f"{'='*50}")

    for mode, df in [('base(8字段)', df_base),
                     ('enriched(50+字段)', df_enriched)]:
        try:
            engine = FeatureEngine(mode=mode.split('(')[0])
            feat   = engine.fit_transform(df)
            q_vec  = engine.build_query_vector(
                [test_disease], clinical_kb)
            sims   = ms_cosine_similarity(q_vec, feat)
            sims_np = sims.asnumpy()

            # 找 Top-5
            top5_idx = np.argsort(sims_np)[::-1][:5]
            print(f"\n  [{mode}] Top-5:")
            for rank, idx in enumerate(top5_idx, 1):
                row = df.iloc[idx]
                print(f"    #{rank} {row.get('产品名称','?')[:22]:22s} "
                      f"sim={sims_np[idx]:.4f} "
                      f"类别={row.get('产品类别','?')[:10]}")
        except Exception as e:
            print(f"  [{mode}] 跳过（数据不足）: {e}")


# ══════════════════════════════════════════════════════════════
# MindSpore vs NumPy vs sklearn Benchmark
# ══════════════════════════════════════════════════════════════
def benchmark_cosine(n_products: int = 182, n_features: int = 50):
    """
    对比 MindSpore / NumPy / sklearn 的 cosine similarity 性能
    在报告中展示 MindSpore 的工程价值
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import os

    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
        if os.path.exists(p): fm.fontManager.addfont(p)
    matplotlib.rcParams.update({'font.family': 'Noto Sans CJK JP',
                                'axes.unicode_minus': False})
    FREG  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    FBOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    def fp(s=10, b=False):
        return fm.FontProperties(fname=FBOLD if b else FREG, size=s)

    np.random.seed(42)
    rng     = np.random
    N_REPS  = 200   # 重复次数（模拟高并发查询）
    results = {}

    mat_np  = rng.randn(n_products, n_features).astype(np.float32)
    q_np    = rng.randn(1, n_features).astype(np.float32)
    mat_ms  = Tensor(mat_np)
    q_ms    = Tensor(q_np)

    # ── NumPy ─────────────────────────────────────────────
    times_np = []
    for _ in range(N_REPS):
        t = time.perf_counter()
        q_n = q_np / (np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-8)
        m_n = mat_np / (np.linalg.norm(mat_np, axis=1, keepdims=True) + 1e-8)
        _ = q_n @ m_n.T
        times_np.append((time.perf_counter() - t) * 1000)

    # ── sklearn ───────────────────────────────────────────
    times_sk = []
    for _ in range(N_REPS):
        t = time.perf_counter()
        _ = sk_cos(q_np, mat_np)
        times_sk.append((time.perf_counter() - t) * 1000)

    # ── MindSpore ─────────────────────────────────────────
    # 预热
    for _ in range(10):
        _ = ms_cosine_similarity(q_ms, mat_ms)

    times_ms = []
    for _ in range(N_REPS):
        t = time.perf_counter()
        _ = ms_cosine_similarity(q_ms, mat_ms)
        times_ms.append((time.perf_counter() - t) * 1000)

    results = {
        'NumPy':      times_np,
        'sklearn':    times_sk,
        'MindSpore\n(CPU)': times_ms,
    }

    # ── 可视化 ─────────────────────────────────────────────
    C1,C2,C3 = '#471365','#335c8a','#20958d'
    colors   = [C1, C2, C3]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    # 箱线图
    bp = axes[0].boxplot(
        list(results.values()),
        labels=list(results.keys()),
        patch_artist=True,
        notch=False, widths=0.45,
        medianprops=dict(color='white', lw=2.2),
        whiskerprops=dict(lw=1.3, color='#555'),
        capprops=dict(lw=1.8, color='#555'),
        flierprops=dict(marker='o', ms=3.5, alpha=0.3,
                        linestyle='none', mec='none'),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)

    axes[0].set_ylabel('单次推理耗时（毫秒）',
                        fontproperties=fp(10))
    axes[0].set_title(f'Cosine Similarity 性能对比\n'
                       f'(N={n_products}产品 × D={n_features}特征，{N_REPS}次)',
                       fontproperties=fp(11, True), pad=10)
    for lbl in axes[0].get_xticklabels():
        lbl.set_fontproperties(fp(9.5))
    for lbl in axes[0].get_yticklabels():
        lbl.set_fontproperties(fp(9))

    # 均值柱状图 + 标注
    means  = [np.mean(v) for v in results.values()]
    labels = list(results.keys())
    bars   = axes[1].bar(range(len(means)), means,
                          color=colors, alpha=0.78,
                          edgecolor='white', linewidth=1.3,
                          width=0.55)
    for bar, m, std in zip(bars, means,
                            [np.std(v) for v in results.values()]):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f'{m:.3f} ms\n±{std:.3f}',
            ha='center', va='bottom',
            fontproperties=fp(9, True), color='#333',
            linespacing=1.4
        )
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    for lbl in axes[1].get_xticklabels():
        lbl.set_fontproperties(fp(9.5))
    for lbl in axes[1].get_yticklabels():
        lbl.set_fontproperties(fp(9))
    axes[1].set_ylabel('平均推理耗时（毫秒）',
                        fontproperties=fp(10))
    axes[1].set_title('各框架平均耗时对比',
                       fontproperties=fp(11, True), pad=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # 注脚
    fig.text(0.5, -0.02,
        f'测试环境：{MS_CFG.device}  ·  MindSpore {ms.__version__}  ·  '
        f'产品规模 N={n_products}  ·  特征维度 D={n_features}',
        ha='center', fontproperties=fp(8), color='#888888')
    fig.suptitle('Route B 核心算子性能基准测试  ·  MindSpore vs NumPy vs sklearn',
                 fontproperties=fp(13, True), y=1.02)

    plt.tight_layout()
    out = '/mnt/user-data/outputs/benchmark_mindspore.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  NumPy    均值: {np.mean(times_np):.4f} ms")
    print(f"  sklearn  均值: {np.mean(times_sk):.4f} ms")
    print(f"  MindSpore均值: {np.mean(times_ms):.4f} ms")
    print(f"  ✓ Benchmark 图: benchmark_mindspore.png")
    return results


if __name__ == '__main__':
    print("Route B MindSpore 模块测试")
    benchmark_cosine(n_products=182, n_features=50)
