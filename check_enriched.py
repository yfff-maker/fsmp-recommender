"""
check_enriched.py v2  ——  从缓存重建 Excel，然后做质量报告
"""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, CACHE_DIR

# ── 先从 cache 重建完整 Excel（确保包含 retry 的结果）────────
cache_path = CACHE_DIR / "extraction_progress.json"
if cache_path.exists():
    prog = json.loads(cache_path.read_text(encoding='utf-8'))
    done = prog.get('done', {})
    rows = []
    for data in done.values():
        row = {k: v for k, v in data.items() if k != 'nutrition'}
        for k, v in (data.get('nutrition') or {}).items():
            row[f'nut_{k}'] = v
        rows.append(row)
    df_cache = pd.DataFrame(rows)
    out = OUTPUT_DIR / "products_enriched.xlsx"
    df_cache.to_excel(out, index=False)
    print(f"已从缓存重建 Excel: {len(df_cache)} 款产品")
    df = df_cache
else:
    out = OUTPUT_DIR / "products_enriched.xlsx"
    if not out.exists():
        print("✗ 找不到数据文件"); sys.exit(1)
    df = pd.read_excel(out)

print(f"\n{'='*58}")
print(f"  提取结果质量报告")
print(f"{'='*58}")
print(f"  产品总数:  {len(df)} / 182 款  "
      f"（覆盖率 {len(df)/182*100:.1f}%）")
print(f"  字段总数:  {len(df.columns)} 列")

# ── 基本字段 ─────────────────────────────────────────────────
print(f"\n  【基本信息字段】")
basic = ['reg_no','product_name','category','population',
         'net_weight','shelf_life_months','admin_route',
         'protein_source','lactose_free','single_source_ok']
for c in basic:
    if c in df.columns:
        rate = df[c].notna().mean()*100
        flag = '✓' if rate > 90 else '⚠' if rate > 60 else '✗'
        print(f"    {flag} {c:28s}: {rate:5.1f}%")

# ── 核心营养字段 ─────────────────────────────────────────────
print(f"\n  【核心营养字段（每100mL）】")
core_nut = ['nut_energy_kJ','nut_protein_g','nut_fat_g',
            'nut_carbohydrate_g','nut_sodium_mg','nut_potassium_mg',
            'nut_phosphorus_mg','nut_calcium_mg','nut_iron_mg','nut_zinc_mg']
for c in core_nut:
    if c in df.columns:
        rate = df[c].notna().mean()*100
        flag = '✓' if rate > 85 else '⚠' if rate > 60 else '✗'
        print(f"    {flag} {c:32s}: {rate:5.1f}%")

# ── 维生素字段 ───────────────────────────────────────────────
print(f"\n  【维生素字段（新增）】")
vit = ['nut_vit_a_ug','nut_vit_d_ug','nut_vit_e_mg',
       'nut_vit_c_mg','nut_folate_ug','nut_vit_b12_ug']
for c in vit:
    if c in df.columns:
        rate = df[c].notna().mean()*100
        flag = '✓' if rate > 50 else '⚠' if rate > 30 else '✗'
        print(f"    {flag} {c:32s}: {rate:5.1f}%")

# ── 特殊成分（低覆盖率是正常的）─────────────────────────────
print(f"\n  【特殊功能成分（非必含，低覆盖属正常）】")
special = ['nut_choline_mg','nut_taurine_mg','nut_carnitine_mg',
           'nut_dietary_fiber_g','nut_dha_mg']
for c in special:
    if c in df.columns:
        rate = df[c].notna().mean()*100
        n    = df[c].notna().sum()
        print(f"    · {c:32s}: {rate:4.1f}%  ({n}款含有)")

# ── 不完整记录 ───────────────────────────────────────────────
if '_incomplete' in df.columns:
    inc = df[df['_incomplete']==True]
    print(f"\n  核心字段不完整: {len(inc)} 款")
    for _, row in inc.iterrows():
        print(f"    · {row.get('reg_no','')}  "
              f"缺失: {row.get('_missing','?')}")

# ── 与原始数据对比 ───────────────────────────────────────────
nut_cols_new = [c for c in df.columns if c.startswith('nut_')]
print(f"\n  【与原始数据对比】")
print(f"    result1 原有字段:  8 个营养字段")
print(f"    PDF 新增字段:      {len(nut_cols_new)} 个营养字段")
print(f"    总字段增幅:        {len(nut_cols_new)-8} 个（+{(len(nut_cols_new)-8)/8*100:.0f}%）")
print(f"\n  文件: {out}")
print(f"{'='*58}")