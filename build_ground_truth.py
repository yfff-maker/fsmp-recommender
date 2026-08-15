"""
build_ground_truth.py
从 CLINICAL_KB 反推 ground truth，输出两份文件：
  outputs/ground_truth.json  —— 完整结构（含产品名称，便于人工审核）
  outputs/ground_truth.csv   —— 展开为行记录（便于指标计算）

CSV 结构:
  condition, age_group, 注册证号, 产品名称, 产品类别, label
  label ∈ {S, A, ban}
"""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

DATA_PATH  = Path(__file__).parent / "data" / "result2.xlsx"
OUT_DIR    = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── 年龄段 → 适用人群类别 ───────────────────────────────────────
AGE_TO_POP = {
    "婴儿":    "特医婴配食品",
    "1岁以上": "1岁以上特医食品",
}

# ── 直接从 CLINICAL_KB 提取的规则（preferred/alternative/contraindicated）──
RULES: dict[str, tuple[str, list, list, list]] = {
    "食物蛋白过敏":    ("婴儿",    ["氨基酸配方"],              ["乳蛋白深度水解配方"],  ["乳蛋白部分水解配方"]),
    "乳蛋白过敏高风险":("婴儿",    ["乳蛋白部分水解配方"],      [],                      []),
    "乳糖不耐受_婴儿": ("婴儿",    ["无乳糖配方"],              ["低乳糖配方"],           []),
    "早产":            ("婴儿",    ["早产/低出生体重婴儿配方"], ["母乳营养补充剂"],        []),
    "苯丙酮尿症_婴儿": ("婴儿",    ["氨基酸代谢障碍配方"],      [],                      ["氨基酸配方", "乳蛋白深度水解配方", "乳蛋白部分水解配方"]),
    "补充蛋白质":      ("1岁以上", ["蛋白质（氨基酸）组件"],    ["非全营养配方食品"],      []),
    "消化吸收障碍":    ("1岁以上", ["全营养配方食品"],          ["非全营养配方食品"],      []),
    "肿瘤":            ("1岁以上", ["特定全营养配方食品"],       ["全营养配方食品"],        []),
    "腹泻脱水":        ("1岁以上", ["电解质配方"],              ["非全营养配方食品"],      []),
    "吞咽障碍":        ("1岁以上", ["增稠组件"],                ["全营养配方食品"],        []),
    "苯丙酮尿症":      ("1岁以上", ["非全营养配方食品"],        [],                      ["氨基酸配方"]),
}


def build(data_path: Path = DATA_PATH):
    df = pd.read_excel(data_path)
    keep_cols = ["注册证号", "产品名称", "产品类别"]

    gt_json  = {}
    rows     = []

    for cond, (age, preferred, alternative, contraindicated) in RULES.items():
        pop = AGE_TO_POP[age]
        sub = df[df["适用人群类别"] == pop]

        s   = sub[sub["产品类别"].isin(preferred)]
        a   = sub[sub["产品类别"].isin(alternative)]
        ban = sub[sub["产品类别"].isin(contraindicated)]

        gt_json[cond] = {
            "age_group": age,
            "pop_cat":   pop,
            "S":   s[keep_cols].to_dict("records"),
            "A":   a[keep_cols].to_dict("records"),
            "ban": ban[keep_cols].to_dict("records"),
        }

        for label, subset in [("S", s), ("A", a), ("ban", ban)]:
            for _, row in subset.iterrows():
                rows.append({
                    "condition":  cond,
                    "age_group":  age,
                    "注册证号":   row["注册证号"],
                    "产品名称":   row["产品名称"],
                    "产品类别":   row["产品类别"],
                    "label":      label,
                })

    # ── 保存 JSON ───────────────────────────────────────────────
    json_path = OUT_DIR / "ground_truth.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_json, f, ensure_ascii=False, indent=2)

    # ── 保存 CSV ────────────────────────────────────────────────
    csv_path = OUT_DIR / "ground_truth.csv"
    gt_df = pd.DataFrame(rows)
    gt_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── 打印摘要 ────────────────────────────────────────────────
    print("=" * 60)
    print("Ground Truth 构建完成")
    print("=" * 60)
    print(f"{'病症':<20} {'年龄段':<8} {'S':>4} {'A':>4} {'ban':>5} {'小计':>5}")
    print("-" * 60)
    total = 0
    for cond, v in gt_json.items():
        s_n, a_n, ban_n = len(v["S"]), len(v["A"]), len(v["ban"])
        sub_total = s_n + a_n + ban_n
        total += sub_total
        print(f"{cond:<20} {v['age_group']:<8} {s_n:>4} {a_n:>4} {ban_n:>5} {sub_total:>5}")
    print("-" * 60)
    print(f"{'合计':<20} {'':8} {'':>4} {'':>4} {'':>5} {total:>5}")
    print()
    print(f"病症数:  {len(gt_json)}")
    print(f"总条数:  {total}  (product-condition pairs)")
    print(f"输出路径:")
    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")

    return gt_df


if __name__ == "__main__":
    build()
