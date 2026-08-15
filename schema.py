"""
schema.py  ——  产品数据模型（提取目标字段的唯一定义）
基于 GB 29922-2013 + GB 25596-2010 标准字段体系
使用 Pydantic 做类型校验，确保提取质量
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pandas as pd

# ══════════════════════════════════════════════════════════════
# 核心产品 Schema（从 PDF 说明书提取）
# ══════════════════════════════════════════════════════════════
class NutritionPer100mL(BaseModel):
    """每 100mL 营养成分（标准冲调液）"""
    # ── result1 已有（8项）──────────────────────────────────
    energy_kJ         : Optional[float] = Field(None, description="能量 kJ")
    protein_g         : Optional[float] = Field(None, description="蛋白质 g")
    fat_g             : Optional[float] = Field(None, description="脂肪 g")
    carbohydrate_g    : Optional[float] = Field(None, description="碳水化合物 g")
    sodium_mg         : Optional[float] = Field(None, description="钠 mg")
    chloride_mg       : Optional[float] = Field(None, description="氯 mg")
    potassium_mg      : Optional[float] = Field(None, description="钾 mg")
    phosphorus_mg     : Optional[float] = Field(None, description="磷 mg")

    # ── PDF 新增（脂肪酸）───────────────────────────────────
    linoleic_acid_g   : Optional[float] = Field(None, description="亚油酸 g (n-6)")
    ala_mg            : Optional[float] = Field(None, description="α-亚麻酸 mg (n-3)")
    dha_mg            : Optional[float] = Field(None, description="DHA mg（若含）")

    # ── PDF 新增（维生素，14项）────────────────────────────
    vit_a_ug          : Optional[float] = Field(None, description="维生素A μg RE")
    vit_d_ug          : Optional[float] = Field(None, description="维生素D μg")
    vit_e_mg          : Optional[float] = Field(None, description="维生素E mg αTE")
    vit_k1_ug         : Optional[float] = Field(None, description="维生素K1 μg")
    vit_b1_mg         : Optional[float] = Field(None, description="维生素B1 mg")
    vit_b2_mg         : Optional[float] = Field(None, description="维生素B2 mg")
    vit_b6_mg         : Optional[float] = Field(None, description="维生素B6 mg")
    vit_b12_ug        : Optional[float] = Field(None, description="维生素B12 μg")
    niacin_mg         : Optional[float] = Field(None, description="烟酸 mg")
    folate_ug         : Optional[float] = Field(None, description="叶酸 μg")
    pantothenic_mg    : Optional[float] = Field(None, description="泛酸 mg")
    vit_c_mg          : Optional[float] = Field(None, description="维生素C mg")
    biotin_ug         : Optional[float] = Field(None, description="生物素 μg")

    # ── PDF 新增（矿物质，10项）────────────────────────────
    calcium_mg        : Optional[float] = Field(None, description="钙 mg")
    magnesium_mg      : Optional[float] = Field(None, description="镁 mg")
    iron_mg           : Optional[float] = Field(None, description="铁 mg")
    zinc_mg           : Optional[float] = Field(None, description="锌 mg")
    copper_ug         : Optional[float] = Field(None, description="铜 μg")
    manganese_ug      : Optional[float] = Field(None, description="锰 μg")
    iodine_ug         : Optional[float] = Field(None, description="碘 μg")
    selenium_ug       : Optional[float] = Field(None, description="硒 μg")
    chromium_ug       : Optional[float] = Field(None, description="铬 μg")
    molybdenum_ug     : Optional[float] = Field(None, description="钼 μg")

    # ── PDF 新增（特殊功能成分）────────────────────────────
    choline_mg        : Optional[float] = Field(None, description="胆碱 mg")
    taurine_mg        : Optional[float] = Field(None, description="牛磺酸 mg")
    carnitine_mg      : Optional[float] = Field(None, description="左旋肉碱 mg")
    dietary_fiber_g   : Optional[float] = Field(None, description="膳食纤维 g")
    inositol_mg       : Optional[float] = Field(None, description="肌醇 mg（若含）")


class FSMPProduct(BaseModel):
    """
    特医食品完整产品模型（result1/2 + PDF 提取）
    字段来源注明，便于追溯
    """
    # ── result2 已有 ────────────────────────────────────────
    reg_no            : str   = Field(...,  description="注册证号（主键）")
    product_name      : str   = Field(...,  description="产品名称")
    company           : Optional[str] = Field(None, description="企业名称")
    category          : Optional[str] = Field(None, description="产品类别")
    population        : Optional[str] = Field(None, description="适用人群（全文）")
    population_class  : Optional[str] = Field(None, description="适用人群类别")
    source            : Optional[str] = Field(None, description="产品来源 国产/进口")
    reg_year          : Optional[int] = Field(None, description="登记年份")
    texture           : Optional[str] = Field(None, description="组织状态 粉状/液态")

    # ── PDF 新增（产品信息）─────────────────────────────────
    net_weight        : Optional[str] = Field(None, description="净含量规格 如 400g/1000mL")
    shelf_life_months : Optional[int] = Field(None, description="保质期（月）")
    admin_route       : Optional[str] = Field(None, description="食用方式 口服/管饲/两者")
    mixing_ratio      : Optional[str] = Field(None, description="冲调比例 如 22g/100mL")
    energy_density    : Optional[float]= Field(None, description="能量密度 kcal/mL")

    # ── PDF 新增（蛋白质来源结构）──────────────────────────
    protein_source    : Optional[str] = Field(None,
        description="蛋白质来源 如 乳清蛋白/酪蛋白/大豆蛋白/氨基酸混合")
    whey_ratio_pct    : Optional[float]= Field(None,
        description="乳清蛋白占比 % (若可提取)")
    fat_source        : Optional[str] = Field(None,
        description="脂肪来源 如 MCT/植物油/鱼油")
    fiber_source      : Optional[str] = Field(None,
        description="膳食纤维来源 如 低聚果糖/菊粉/FOS+Inulin")
    has_mcfa          : Optional[bool] = Field(None, description="是否含中链脂肪酸")
    lactose_free      : Optional[bool] = Field(None, description="是否无乳糖")
    gluten_free       : Optional[bool] = Field(None, description="是否无麸质")

    # ── PDF 新增（功能声称）────────────────────────────────
    formula_feature   : Optional[str] = Field(None,
        description="配方特点/营养学特征（摘要）")
    special_additions : Optional[str] = Field(None,
        description="特殊添加成分声称 如 含DHA/核苷酸/益生元")
    single_source_ok  : Optional[bool] = Field(None,
        description="是否可作为单一营养来源")

    # ── 营养成分（嵌套对象，每100mL）──────────────────────
    nutrition         : Optional[NutritionPer100mL] = None

    # ── PDF 新增（安全信息）────────────────────────────────
    contraindications : Optional[str] = Field(None,
        description="禁忌/警示（摘要）")
    storage_info      : Optional[str] = Field(None, description="储存条件")

    @field_validator('reg_no')
    @classmethod
    def reg_no_format(cls, v):
        if not v.startswith(('国食注字', 'TY', 'TZ', 'TB')):
            # 宽松校验：只要非空
            if len(v) < 5:
                raise ValueError(f"注册证号格式异常: {v}")
        return v.strip()

    def to_flat_dict(self) -> dict:
        """展平为单行 DataFrame 格式"""
        d = self.model_dump(exclude={'nutrition'})
        if self.nutrition:
            for k, v in self.nutrition.model_dump().items():
                d[f'nut_{k}'] = v
        return d


# ══════════════════════════════════════════════════════════════
# 从现有 result1/2 构建基础产品对象
# ══════════════════════════════════════════════════════════════
def load_existing_data(r1_path, r2_path) -> pd.DataFrame:
    """加载并合并现有结构化数据"""
    df1 = pd.read_excel(r1_path)
    df2 = pd.read_excel(r2_path)
    df1.rename(columns={
        '能量(kJ)':'能量_kJ','蛋白质(g)':'蛋白质_g','脂肪(g)':'脂肪_g',
        '碳水化合物(g)':'碳水_g','钠(mg)':'钠_mg','氯(mg)':'氯_mg',
        '钾(mg)':'钾_mg','磷(mg)':'磷_mg'
    }, inplace=True)
    df = pd.merge(df2, df1, on='注册证号', how='left')
    df.rename(columns={'产品名称_x':'产品名称'}, inplace=True)
    df.drop(columns=[c for c in df.columns if c.endswith('_y')],
            inplace=True, errors='ignore')
    return df


# ══════════════════════════════════════════════════════════════
# 字段映射说明（报告用）
# ══════════════════════════════════════════════════════════════
FIELD_SOURCE_MAP = {
    'result1_result2': [
        '注册证号','产品名称','企业名称','产品类别','适用人群',
        '适用人群类别','产品来源','登记年份','组织状态',
        '能量_kJ','蛋白质_g','脂肪_g','碳水_g','钠_mg','氯_mg','钾_mg','磷_mg'
    ],
    'pdf_extracted': [
        '净含量规格','保质期','食用方式','冲调比例','能量密度',
        '蛋白质来源','乳清蛋白占比','脂肪来源','膳食纤维来源',
        '是否含MCT','是否无乳糖','是否无麸质',
        '亚油酸','α-亚麻酸','DHA',
        '维生素A-K(14项)','矿物质全谱(10项)',
        '胆碱','牛磺酸','左旋肉碱','膳食纤维',
        '配方特点摘要','特殊功能成分声称','是否单一营养来源',
        '禁忌摘要','储存条件'
    ]
}

if __name__ == '__main__':
    print(f"Schema 定义完成")
    print(f"  result1/2 原有字段: {len(FIELD_SOURCE_MAP['result1_result2'])}项")
    print(f"  PDF 新增字段:       {len(FIELD_SOURCE_MAP['pdf_extracted'])}项")
    print(f"  NutritionPer100mL:  {len(NutritionPer100mL.model_fields)}项营养成分")
    # 计算总字段数
    total = (len(FSMPProduct.model_fields) +
             len(NutritionPer100mL.model_fields))
    print(f"  FSMPProduct 总字段: {total}项")
