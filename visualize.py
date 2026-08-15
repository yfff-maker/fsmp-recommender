"""
visualize.py  ——  统一可视化输出
接收 pipeline 结果，生成报告级图表
"""
from __future__ import annotations
import sys, textwrap, os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR

for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
          '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if os.path.exists(p): fm.fontManager.addfont(p)
matplotlib.rcParams.update({
    'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.8,'axes.grid':True,
    'grid.alpha':0.22,'grid.linestyle':'--',
})
FREG  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FBOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def fp(s=10,b=False):
    return fm.FontProperties(fname=FBOLD if b else FREG, size=s)
def tfs(ax,s=9):
    for l in ax.get_xticklabels()+ax.get_yticklabels():
        l.set_fontproperties(fp(s))

C1,C2,C3,C4,C5='#471365','#335c8a','#20958d','#9ed030','#d4e600'
TIER_C = {'S':C1,'A':C2,'B':C3,'C':'#999999'}
PAL = [C1,C2,C3,C4,C5]


def _single_report(result, ax_row):
    """单客户推荐结果图（一行 3 个子图）"""
    ax1, ax2, ax3 = ax_row
    ranked = result.ranked_df
    name   = result.client_name
    top8   = ranked.head(8).reset_index(drop=True)

    # ── 子图①：综合得分横向条形 ──────────────────────────
    bar_c = [TIER_C.get(str(g),'#999') for g in top8['A级别']]
    names = [textwrap.shorten(n, width=17, placeholder='…')
             for n in top8['产品名称']]
    bars  = ax1.barh(range(len(names)), top8['综合得分'],
                     color=bar_c, alpha=0.82,
                     edgecolor='white', linewidth=1.2, height=0.60)
    for i,(bar,row) in enumerate(zip(bars,top8.itertuples())):
        ax1.text(bar.get_width()+0.5,
                 bar.get_y()+bar.get_height()/2,
                 f'{row.综合得分:.0f} [{row.A级别}]',
                 va='center', fontproperties=fp(8,True), color='#333')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    for l in ax1.get_yticklabels(): l.set_fontproperties(fp(9))
    ax1.invert_yaxis()
    ax1.set_xlim(0, top8['综合得分'].max()*1.28)
    ax1.set_title(f'{name}\n综合推荐得分 Top-8',
                  fontproperties=fp(11,True), pad=8)
    ax1.set_xlabel('综合得分（Route A×B）', fontproperties=fp(9.5))
    tfs(ax1, 8.5)

    # ── 子图②：营养散点（蛋白质 vs 碳水）────────────────
    sc_c = [TIER_C.get(str(g),'#999') for g in top8['A级别']]
    ax2.scatter(top8['碳水_g'], top8['蛋白质_g'],
                c=sc_c, s=70, alpha=0.75,
                edgecolors='white', linewidths=1.0, zorder=3)
    # 标注 Top3
    for i,row in top8.head(3).iterrows():
        ax2.annotate(f'#{i+1}', (row['碳水_g'], row['蛋白质_g']),
                     xytext=(3,3), textcoords='offset points',
                     fontproperties=fp(8,True), color=C1)
    ax2.set_xlabel('碳水化合物（g/100mL）', fontproperties=fp(9.5))
    ax2.set_ylabel('蛋白质（g/100mL）',    fontproperties=fp(9.5))
    ax2.set_title('营养分布\n（气泡=推荐排名）',
                  fontproperties=fp(11,True), pad=8)
    tfs(ax2, 8.5)

    # ── 子图③：推荐摘要文本框 ────────────────────────────
    ax3.axis('off')
    top1 = ranked.iloc[0]
    llm  = result.llm_output

    lines = [
        f"推荐摘要 — {name}",
        "─"*28,
        f"年龄: {result.parsed_query.age_group}",
        f"病症: {'、'.join(result.parsed_query.conditions)}",
        f"候选池: {len(result.filter_result.pool)}款",
        f"耗时: {result.elapsed_sec:.2f}s",
        "",
        "🥇 第1推荐",
        top1['产品名称'][:20],
        f"  类别: {top1['产品类别'][:12]}",
        f"  得分: {top1['综合得分']:.1f} [{top1['A级别']}]",
        f"  蛋白: {top1['蛋白质_g']:.2f}g/100mL",
        "",
    ]

    if llm and llm.recommendations:
        lines += ["◆ LLM推荐理由（" + llm.provider_used + "）"]
        for rec in llm.recommendations[:2]:
            lines.append(textwrap.fill(f"#{rec.rank} {rec.reason}", width=26))
        if llm.combination_advice:
            lines += ["", "🔗 " + textwrap.fill(llm.combination_advice, 24)]
    else:
        for n in result.filter_result.notes[:2]:
            lines.append(textwrap.fill(n[:55], width=26))
        for c in result.filter_result.combination[:1]:
            lines.append(textwrap.fill(c[:55], width=26))

    ax3.text(0.04, 0.97, '\n'.join(lines),
             transform=ax3.transAxes, va='top', ha='left',
             fontproperties=fp(8.8), linespacing=1.55,
             bbox=dict(boxstyle='round,pad=0.6', fc='#F8F6FF',
                       ec=C1, lw=1.4, alpha=0.96))


def visualize_all(results: list, fname='final_report.png'):
    n = len(results)
    fig = plt.figure(figsize=(20, 7.5*n))
    gs  = gridspec.GridSpec(n, 3, figure=fig,
                            hspace=0.55, wspace=0.38)

    for i, result in enumerate(results):
        _single_report(result,
                       [fig.add_subplot(gs[i,0]),
                        fig.add_subplot(gs[i,1]),
                        fig.add_subplot(gs[i,2])])

    # 图例
    handles = [
        mpatches.Patch(color=C1, alpha=0.82, label='S级 — 临床首选'),
        mpatches.Patch(color=C2, alpha=0.82, label='A级 — 临床次选'),
        mpatches.Patch(color='#999', alpha=0.82, label='C级 — 规则外'),
    ]
    fig.legend(handles=handles, loc='lower center',
               prop=fp(10), ncol=3, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        '特医食品智能推荐系统  ·  Route A × B × C  全流程推荐报告',
        fontproperties=fp(14,True), y=1.01
    )
    out = str(OUTPUT_DIR / fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 最终报告: {fname}")
    return out
