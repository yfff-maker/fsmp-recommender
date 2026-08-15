"""
merge_data.py  ——  合并官方数据与 PDF 提取结果
输入：result1.xlsx + result2.xlsx + outputs/products_enriched.xlsx
输出：data/products_final.xlsx
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR

print("=" * 58)
print("  数据合并：result1/2 × products_enriched → products_final")
print("=" * 58)

# ── 1. 官方数据（权威来源）─────────────────────────────────────
r1 = pd.read_excel(DATA_DIR / 'result1.xlsx')
r2 = pd.read_excel(DATA_DIR / 'result2.xlsx')
r1.rename(columns={
    '能量(kJ)':'能量_kJ','蛋白质(g)':'蛋白质_g','脂肪(g)':'脂肪_g',
    '碳水化合物(g)':'碳水_g','钠(mg)':'钠_mg','氯(mg)':'氯_mg',
    '钾(mg)':'钾_mg','磷(mg)':'磷_mg'}, inplace=True)
base = pd.merge(r2, r1, on='注册证号', how='left')
base.rename(columns={'产品名称_x':'产品名称'}, inplace=True)
base.drop(columns=[c for c in base.columns if c.endswith('_y')],
          inplace=True, errors='ignore')
base['注册证号'] = base['注册证号'].str.strip()
print(f"\n  官方基础数据: {len(base)} 款 × {len(base.columns)} 字段")

# ── 2. PDF 提取结果 ───────────────────────────────────────────
enriched_path = OUTPUT_DIR / 'products_enriched.xlsx'
if not enriched_path.exists():
    print(f"✗ 找不到 {enriched_path}")
    print("  请先运行 pdf_extractor.py 生成该文件")
    sys.exit(1)
enriched = pd.read_excel(enriched_path)
print(f"  PDF提取数据:  {len(enriched)} 款 × {len(enriched.columns)} 字段")

# 对齐注册证号（PDF提取的字段名是 reg_no）
if 'reg_no' in enriched.columns:
    enriched['注册证号'] = enriched['reg_no'].fillna('').str.strip()
elif '注册证号' in enriched.columns:
    enriched['注册证号'] = enriched['注册证号'].fillna('').str.strip()

# ── 3. 只取 PDF 新增字段（不覆盖官方权威字段）──────────────
NEW_INFO = [
    'net_weight','shelf_life_months','admin_route','mixing_ratio',
    'energy_density_kcal_ml','protein_source','whey_ratio_pct',
    'fat_source','fiber_source','has_mcfa','lactose_free',
    'single_source_ok','formula_feature','special_additions',
    'contraindications',
]
NUT_COLS = [c for c in enriched.columns if c.startswith('nut_')]
take_cols = ['注册证号'] + \
            [c for c in NEW_INFO if c in enriched.columns] + \
            NUT_COLS
enriched_slim = enriched[take_cols].copy()

merged = pd.merge(base, enriched_slim, on='注册证号', how='left')
print(f"\n  合并结果: {len(merged)} 款 × {len(merged.columns)} 字段")

# ── 4. result1 回填核心营养（修复 4 款不完整产品）─────────────
FILL_MAP = {
    'nut_energy_kJ':     '能量_kJ',
    'nut_protein_g':     '蛋白质_g',
    'nut_fat_g':         '脂肪_g',
    'nut_carbohydrate_g':'碳水_g',
    'nut_sodium_mg':     '钠_mg',
    'nut_chloride_mg':   '氯_mg',
    'nut_potassium_mg':  '钾_mg',
    'nut_phosphorus_mg': '磷_mg',
}
filled = 0
for nut_col, base_col in FILL_MAP.items():
    if nut_col in merged.columns and base_col in merged.columns:
        mask = merged[nut_col].isna() & merged[base_col].notna()
        merged.loc[mask, nut_col] = merged.loc[mask, base_col]
        filled += mask.sum()
print(f"  result1 回填核心营养: {filled} 处（修复不完整产品）")

# ── 5. 最终完整性检查 ─────────────────────────────────────────
core_ok = merged[['nut_energy_kJ','nut_protein_g',
                   'nut_fat_g','nut_carbohydrate_g']].notna().all(axis=1).sum()
nut_total = len([c for c in merged.columns if c.startswith('nut_')])
nut_rich  = len([c for c in merged.columns
                 if c.startswith('nut_') and
                 merged[c].notna().mean() > 0.5])

print(f"\n  核心营养完整率: {core_ok}/182 款 ({core_ok/182*100:.1f}%)")
print(f"  nut_ 字段总数:  {nut_total} 个")
print(f"  覆盖率>50%的:   {nut_rich} 个（可用于 Route B 特征）")

# ── 6. 保存 ──────────────────────────────────────────────────
out = DATA_DIR / 'products_final.xlsx'
merged.to_excel(out, index=False)
print(f"\n  ✓ 保存: {out}")
print(f"    {len(merged)} 款 × {len(merged.columns)} 字段")

# ── 7. 展示增幅 ───────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"  原始 result1+2:      {len(base.columns):3d} 字段")
print(f"  PDF 提取新增:        {len(merged.columns)-len(base.columns):3d} 字段")
print(f"  最终 products_final: {len(merged.columns):3d} 字段  (+{(len(merged.columns)/len(base.columns)-1)*100:.0f}%)")
print(f"\n  Route B 特征维度:")
print(f"    基础版  (result1):  8 个营养特征")
print(f"    富特征版(enriched): {nut_rich} 个营养特征 + 3个布尔特征")
print(f"    提升幅度:           +{nut_rich-8} 个特征 (+{(nut_rich-8)/8*100:.0f}%)")
print(f"{'─'*40}")
