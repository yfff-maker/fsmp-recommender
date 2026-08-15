"""
run_with_llm.py  ——  完整三路管道（含真实 LLM 调用）
Route A → Route B → Route C(Qwen) → 可视化
"""
import sys, os, time, warnings, textwrap, datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fsmp_project')

import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import mindspore as ms
from mindspore import Tensor
ms.set_context(mode=ms.PYNATIVE_MODE); ms.set_device('CPU')

# ── 字体：跨平台自动检测 ─────────────────────────────────────
def _find_cjk_font():
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',                                       # Windows 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',                                     # Windows 宋体
        'C:/Windows/Fonts/simhei.ttf',                                     # Windows 黑体
        '/System/Library/Fonts/PingFang.ttc',                              # Mac
        '/System/Library/Fonts/STHeiti Light.ttc',                         # Mac
        '/System/Library/Fonts/Hiragino Sans GB.ttc',                      # Mac
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',          # Linux
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',                  # Linux
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            return p
    return None

FONT_PATH = _find_cjk_font()
if FONT_PATH:
    matplotlib.rcParams['font.family'] = \
        fm.FontProperties(fname=FONT_PATH).get_name()
FREG  = FONT_PATH
FBOLD = FONT_PATH

def fp(s=10, b=False):
    if FONT_PATH:
        return fm.FontProperties(fname=FONT_PATH, size=s,
                                 weight='bold' if b else 'normal')
    return fm.FontProperties(size=s, weight='bold' if b else 'normal')

def tfs(ax, s=9):
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontproperties(fp(s))


C1,C2,C3,C4='#471365','#335c8a','#20958d','#9ed030'
TIER_C={'S':C1,'A':C2,'C':'#999999'}
OUT='outputs'

from schema import load_existing_data
from route_a import RouteA, parse_query, CLINICAL_KB
from route_b_mindspore import RouteB_CBF
from route_c import RouteC, RouteCOutput
from config import REC_CFG, LLM_CFG, DATA_DIR

# ── 加载数据 ──────────────────────────────────────────────────
def _find_excel(name):
    candidates = [
        DATA_DIR / name,
        Path(__file__).parent / 'data' / name,
        Path(__file__).parent / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"找不到 {name}，请放到 data/ 目录下")

df = load_existing_data(
    _find_excel('result1.xlsx'),
    _find_excel('result2.xlsx'))

route_a = RouteA()
kb_dict = {k: v.__dict__ for k,v in CLINICAL_KB.items()}
route_b = RouteB_CBF(df, kb_dict, 'base', REC_CFG)
route_c = RouteC(LLM_CFG)

def run_pipeline(client: dict):
    print(f"\n{'='*55}")
    print(f"  {client['name']}  |  {client['diseases']}")
    print(f"{'='*55}")

    # Route A
    t0    = time.perf_counter()
    query = parse_query(client)
    query.raw_input['diseases'] = client.get('diseases', [])
    fr    = route_a.run(df, query)

    # Route B
    pidx = [df.index.get_loc(i) for i in fr.pool.index if i in df.index]
    cbf  = route_b.score(pidx, query.conditions)
    pool = fr.pool.copy()
    pool['CBF得分'] = 0.0
    for li, idx in enumerate(fr.pool.index):
        if li < len(cbf): pool.loc[idx,'CBF得分'] = round(float(cbf[li]),2)
    pool['A级别'] = 'C'
    for cond, tier in fr.tiers.items():
        for idx in tier.get('S',[]):
            if idx in pool.index: pool.loc[idx,'A级别']='S'
        for idx in tier.get('A',[]):
            if idx in pool.index and pool.loc[idx,'A级别']!='S':
                pool.loc[idx,'A级别']='A'
    pool['综合得分'] = (pool['CBF得分'] *
                       pool['A级别'].map(REC_CFG.tier_weights)).round(2)
    ranked = pool.sort_values('综合得分',ascending=False).reset_index(drop=True)
    ranked.index += 1

    print(f"\n  Top-5（Route A+B）:")
    for i, row in ranked.head(5).iterrows():
        print(f"  #{i} {row['产品名称'][:22]:22s} [{row['A级别']}] "
              f"综合={row['综合得分']:.1f} 蛋白={row['蛋白质_g']:.2f}g")

    # Route C（真实 LLM 调用）
    print(f"\n  [C] 调用 {LLM_CFG.provider} / {LLM_CFG.text_model}...")
    llm = route_c.run(client, ranked, fr)
    elapsed = time.perf_counter() - t0

    if llm:
        print(route_c.format_text(llm, client['name']))
    else:
        print("  （LLM未调用，检查 API key）")

    return ranked, fr, llm, query, elapsed


CLIENTS = [
    {'name':    '客户1',
     'age_group':'婴儿',
     'diseases': ['食物蛋白过敏'],
     'query':    '婴儿 蛋白质过敏'},
    {'name':    '客户2',
     'age_group':'1-10岁',
     'diseases': ['补充蛋白质'],
     'query':    '10岁儿童 补充蛋白质 乳糖不耐受'},
]

all_res = [run_pipeline(c) for c in CLIENTS]

# ══════════════════════════════════════════════════════════════
# 可视化：含 LLM 文本的完整报告图
# ══════════════════════════════════════════════════════════════
def make_report(all_res, clients):
    n   = len(clients)
    fig = plt.figure(figsize=(22, 8*n))
    gs  = gridspec.GridSpec(n, 3, hspace=0.55, wspace=0.38,
                            width_ratios=[1.4, 0.9, 1.1])

    for ri, ((ranked, fr, llm, query, elapsed), client) in \
            enumerate(zip(all_res, clients)):
        top8 = ranked.head(8).reset_index(drop=True)
        name = client['name']

        # ── 条形图 ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[ri, 0])
        bc  = [TIER_C.get(str(g),'#999') for g in top8['A级别']]
        ns  = [textwrap.shorten(n, width=17, placeholder='…')
               for n in top8['产品名称']]
        bars= ax1.barh(range(len(ns)), top8['综合得分'], color=bc,
                       alpha=0.82, edgecolor='white', linewidth=1.2, height=0.60)
        for i, (bar, row) in enumerate(zip(bars, top8.itertuples())):
            ax1.text(bar.get_width()+0.3,
                     bar.get_y()+bar.get_height()/2,
                     f'{row.综合得分:.0f}[{row.A级别}]',
                     va='center', fontproperties=fp(8,True), color='#333')
        ax1.set_yticks(range(len(ns))); ax1.set_yticklabels(ns)
        for l in ax1.get_yticklabels(): l.set_fontproperties(fp(9.5))
        ax1.invert_yaxis(); ax1.set_xlim(0, top8['综合得分'].max()*1.28+1)
        ax1.set_title(f'{name} · Top-8 综合推荐得分',
                      fontproperties=fp(11,True), pad=8)
        ax1.set_xlabel('综合得分（Route A × B）', fontproperties=fp(9.5))
        tfs(ax1, 9)

        # ── 营养散点 ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[ri, 1])
        sc2 = [TIER_C.get(str(g),'#999') for g in top8['A级别']]
        ax2.scatter(top8['碳水_g'], top8['蛋白质_g'],
                    c=sc2, s=80, alpha=0.78,
                    edgecolors='white', linewidths=1.0, zorder=3)
        for i, row in top8.head(3).iterrows():
            ax2.annotate(f'#{i+1}', (row['碳水_g'], row['蛋白质_g']),
                         xytext=(3,3), textcoords='offset points',
                         fontproperties=fp(8,True), color=C1)
        ax2.set_xlabel('碳水化合物（g/100mL）', fontproperties=fp(9))
        ax2.set_ylabel('蛋白质（g/100mL）', fontproperties=fp(9))
        ax2.set_title('营养分布', fontproperties=fp(11,True), pad=8)
        tfs(ax2, 8.5)

        # ── LLM 推荐文本框 ────────────────────────────────────
        ax3 = fig.add_subplot(gs[ri, 2]); ax3.axis('off')

        if llm and llm.recommendations:
            lines = [
                f"◆ LLM推荐说明",
                f"  模型: {llm.provider_used.split('/')[-1]}",
                f"  置信度: {llm.confidence}",
                "─"*30, "",
            ]
            for rec in llm.recommendations[:3]:
                lines += [
                    f"#{rec.rank}  {rec.name[:18]}",
                    textwrap.fill(f"    理由: {rec.reason}", width=30),
                    f"    优势: {rec.key_benefit}",
                    "",
                ]
            if llm.combination_advice:
                lines += ["💡 组合方案:",
                          textwrap.fill(llm.combination_advice, width=30), ""]
            if llm.clinical_warnings:
                lines.append("⚠ 临床警示:")
                for w in llm.clinical_warnings[:2]:
                    lines.append(f"  · {textwrap.fill(w, width=28)}")
                lines.append("")
            if llm.monitoring:
                lines.append("📋 监测:")
                for m in llm.monitoring[:2]:
                    lines.append(f"  · {m[:25]}")
            if llm.reviewer_note:
                lines += ["","🔍 纠错:",
                          textwrap.fill(llm.reviewer_note[:60], width=28)]
        else:
            top1 = ranked.iloc[0]
            lines = [
                "◆ 推荐摘要", "─"*28,
                f"年龄: {query.age_group}",
                f"病症: {'、'.join(query.conditions)}", "",
                "🥇 第1推荐",
                textwrap.shorten(top1['产品名称'], width=20, placeholder='…'),
                f"  [{top1['A级别']}] {top1['综合得分']:.1f}分",
                f"  蛋白: {top1['蛋白质_g']:.3f}g/100mL", "",
            ]
            for n2 in fr.notes[:2]:
                lines.append(textwrap.fill(f"💡 {n2[:50]}", width=28))
            lines += ["─"*28,
                      "Route C: 配置 API key 后",
                      "将在此显示 LLM 推荐说明"]

        ax3.text(0.03, 0.97, '\n'.join(lines),
                 transform=ax3.transAxes, va='top', ha='left',
                 fontproperties=fp(8.8), linespacing=1.55,
                 bbox=dict(boxstyle='round,pad=0.65', fc='#F8F6FF',
                           ec=C1, lw=1.5, alpha=0.97))

    handles = [mpatches.Patch(color=C1, alpha=0.82, label='S级—临床首选'),
               mpatches.Patch(color=C2, alpha=0.82, label='A级—临床次选'),
               mpatches.Patch(color='#999', alpha=0.82, label='C级—规则外')]
    fig.legend(handles=handles, loc='lower center', prop=fp(10),
               ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, -0.01))

    has_llm = any(r[2] is not None for r in all_res)
    llm_tag = f"（Route C: {LLM_CFG.text_model}）" if has_llm else "（Route C 未激活）"
    fig.suptitle(f'特医食品智能推荐系统  ·  完整三路报告 {llm_tag}',
                 fontproperties=fp(13,True), y=1.01)

    # ── 文件名带模型名和时间戳，不覆盖历史记录 ──────────────
    model_tag = LLM_CFG.text_model.replace('/', '_').replace('.', '-')
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    out = f'{OUT}/report_{model_tag}_{timestamp}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 报告图已保存: {out}")


make_report(all_res, CLIENTS)
print("\n✓ 全部完成")