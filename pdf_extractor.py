"""
pdf_extractor.py  ——  批量提取 182 份 PDF（修复版）
"""
import json, sys, time
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config import (EXT_CFG, LLM_CFG,
                    DATA_DIR, PDF_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR)
from llm_providers import get_provider

# ══════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════
EXTRACT_SYSTEM_PROMPT = """\
你是特殊医学用途配方食品（特医食品）数据提取专家。
从产品说明书图片中精确提取字段，严格按 JSON 格式输出。
规则：
1. 数值只输出数字，不含单位
2. 缺失字段输出 null，布尔值输出 true/false
3. nutrition 统一使用每 100mL 标准冲调液数值
4. 只输出 JSON，禁止 markdown 和任何解释文字"""

EXTRACT_USER_PROMPT = """\
请从这份特医食品说明书中提取以下信息，输出完整 JSON：

{
  "reg_no": "注册证号，如 国食注字TY20195007",
  "product_name": "产品名称（中文全称）",
  "company": "生产企业名称",
  "category": "产品类别，如 全营养配方食品",
  "population": "适用人群全文（原文）",
  "net_weight": "净含量规格，如 400g",
  "shelf_life_months": 保质期月数（整数），
  "admin_route": "食用方式：口服 或 管饲 或 口服/管饲",
  "mixing_ratio": "冲调比例，如 每100mL含22g本品",
  "energy_density_kcal_ml": 能量密度（kcal/mL，数字），
  "protein_source": "蛋白质来源",
  "whey_ratio_pct": 乳清蛋白占总蛋白%（数字，不能判断则null），
  "fat_source": "脂肪来源",
  "fiber_source": "膳食纤维来源",
  "has_mcfa": 是否含MCT（true/false/null），
  "lactose_free": 是否无乳糖（true/false/null），
  "single_source_ok": 是否可作为单一营养来源（true/false/null），
  "formula_feature": "配方特点摘要（50字内）",
  "special_additions": "特殊添加成分，如 胆碱、牛磺酸",
  "contraindications": "禁忌警示要点（30字内）",
  "nutrition": {
    "energy_kJ": 每100mL能量（kJ），
    "protein_g": 每100mL蛋白质（g），
    "fat_g": 每100mL脂肪（g），
    "carbohydrate_g": 每100mL碳水化合物（g），
    "sodium_mg": 每100mL钠（mg），
    "chloride_mg": 每100mL氯（mg，无则null），
    "potassium_mg": 每100mL钾（mg），
    "phosphorus_mg": 每100mL磷（mg），
    "linoleic_acid_g": 每100mL亚油酸（g），
    "ala_mg": 每100mL α-亚麻酸（mg），
    "dha_mg": 每100mL DHA（mg，无则null），
    "vit_a_ug": 每100mL维生素A（μg RE），
    "vit_d_ug": 每100mL维生素D（μg），
    "vit_e_mg": 每100mL维生素E（mg αTE），
    "vit_k1_ug": 每100mL维生素K1（μg），
    "vit_b1_mg": 每100mL维生素B1（mg），
    "vit_b2_mg": 每100mL维生素B2（mg），
    "vit_b6_mg": 每100mL维生素B6（mg），
    "vit_b12_ug": 每100mL维生素B12（μg），
    "niacin_mg": 每100mL烟酸（mg），
    "folate_ug": 每100mL叶酸（μg），
    "pantothenic_mg": 每100mL泛酸（mg），
    "vit_c_mg": 每100mL维生素C（mg），
    "biotin_ug": 每100mL生物素（μg），
    "calcium_mg": 每100mL钙（mg），
    "magnesium_mg": 每100mL镁（mg），
    "iron_mg": 每100mL铁（mg），
    "zinc_mg": 每100mL锌（mg），
    "copper_ug": 每100mL铜（μg），
    "manganese_ug": 每100mL锰（μg），
    "iodine_ug": 每100mL碘（μg），
    "selenium_ug": 每100mL硒（μg），
    "chromium_ug": 每100mL铬（μg），
    "molybdenum_ug": 每100mL钼（μg），
    "choline_mg": 每100mL胆碱（mg，无则null），
    "taurine_mg": 每100mL牛磺酸（mg，无则null），
    "carnitine_mg": 每100mL左旋肉碱（mg，无则null），
    "dietary_fiber_g": 每100mL膳食纤维（g，无则null）
  }
}"""

# ══════════════════════════════════════════════════════════════
# 校验
# ══════════════════════════════════════════════════════════════
REQUIRED     = ['reg_no', 'product_name', 'category', 'population']
NUT_CRITICAL = ['energy_kJ', 'protein_g', 'fat_g', 'carbohydrate_g']

def validate(data: dict) -> tuple:
    missing  = [f for f in REQUIRED if not data.get(f)]
    nut      = data.get('nutrition') or {}
    missing += [f'nutrition.{f}' for f in NUT_CRITICAL if nut.get(f) is None]
    return len(missing) == 0, missing

# ══════════════════════════════════════════════════════════════
# 批量提取器
# ══════════════════════════════════════════════════════════════
class PDFBatchExtractor:

    def __init__(self, pdf_dir: Path, ext_cfg=EXT_CFG, llm_cfg=LLM_CFG):
        self.pdf_dir  = Path(pdf_dir)
        self.ext_cfg  = ext_cfg
        self.provider = get_provider(llm_cfg)
        self.provider.config.pdf_dpi   = ext_cfg.pdf_dpi
        self.provider.config.max_pages = ext_cfg.max_pages
        self.cache_path = CACHE_DIR / "extraction_progress.json"

        # 日志
        logger.remove()
        logger.add(sys.stdout,
                   format="<green>{time:HH:mm:ss}</green> | "
                          "<level>{level:<7}</level> | {message}",
                   level="INFO", colorize=True)
        logger.add(str(LOG_DIR / "extractor_{time}.log"),
                   level="DEBUG", rotation="50MB")

    def _load_progress(self) -> dict:
        if self.cache_path.exists():
            prog = json.loads(self.cache_path.read_text(encoding='utf-8'))
            logger.info(f"恢复断点：已完成 {len(prog.get('done', {}))} 份")
            return prog
        return {'done': {}, 'failed': []}

    def _save_progress(self, prog: dict):
        self.cache_path.write_text(
            json.dumps(prog, ensure_ascii=False, indent=2), encoding='utf-8')

    def _extract_one(self, pdf_path: Path) -> Optional[dict]:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception as e:
            logger.error(f"读取失败 {pdf_path.name}: {e}")
            return None

        raw = self.provider.extract_from_pdf(pdf_bytes, EXTRACT_USER_PROMPT)
        if not raw:
            logger.warning(f"API无响应: {pdf_path.name}")
            return None

        data = self.provider._safe_json_parse(raw)
        if not data:
            logger.warning(f"JSON解析失败: {pdf_path.name} | {raw[:60]}")
            return None

        ok, missing = validate(data)
        data['_pdf_file']   = pdf_path.name
        data['_incomplete'] = not ok
        if not ok:
            data['_missing'] = missing
            logger.warning(f"字段不完整 {pdf_path.name}: {missing}")
        return data

    def run(self) -> pd.DataFrame:
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdfs:
            logger.error(f"未找到 PDF: {self.pdf_dir}")
            return pd.DataFrame()

        logger.info(f"发现 {len(pdfs)} 份 PDF | "
                    f"模型: {self.provider.config.model_name}")

        prog   = self._load_progress()
        done   = prog['done']
        failed = prog.get('failed', [])
        remain = [p for p in pdfs if p.name not in done]
        logger.info(f"待处理: {len(remain)} 份 | 已完成: {len(done)} 份")

        with tqdm(remain, desc="提取中", unit="份",
                  dynamic_ncols=True) as pbar:
            for i, pdf_path in enumerate(pbar):
                pbar.set_description(pdf_path.stem[:20])
                data = self._extract_one(pdf_path)
                if data:
                    done[pdf_path.name] = data
                    pbar.set_postfix(完成=len(done), 失败=len(failed))
                else:
                    failed.append(pdf_path.name)
                    logger.error(f"✗ {pdf_path.name}")

                if (i + 1) % self.ext_cfg.save_every == 0:
                    self._save_progress({'done': done, 'failed': failed})
                time.sleep(0.3)

        self._save_progress({'done': done, 'failed': failed})
        incomplete = sum(1 for d in done.values() if d.get('_incomplete'))
        logger.info(f"完成 {len(done)}/{len(pdfs)} | "
                    f"失败 {len(failed)} | 不完整 {incomplete}")
        if failed:
            logger.warning(f"失败文件: {failed}")
        return self._to_dataframe(list(done.values()))

    def _to_dataframe(self, results: list) -> pd.DataFrame:
        rows = []
        for d in results:
            row = {k: v for k, v in d.items() if k != 'nutrition'}
            for k, v in (d.get('nutrition') or {}).items():
                row[f'nut_{k}'] = v
            rows.append(row)
        df  = pd.DataFrame(rows)
        out = OUTPUT_DIR / "products_enriched.xlsx"
        df.to_excel(out, index=False)
        logger.info(f"已保存: {out}  ({len(df)}行 × {len(df.columns)}列)")
        return df


# ══════════════════════════════════════════════════════════════
# 单文件测试函数（供 test_single 调用）
# ══════════════════════════════════════════════════════════════
def test_single(pdf_path: str = None):
    """测试单个 PDF 的提取效果"""
    if pdf_path is None:
        # 自动找 data/pdfs/ 里的第一个
        candidates = list(PDF_DIR.glob("*.pdf"))
        if not candidates:
            print(f"✗ data/pdfs/ 目录下没有 PDF 文件")
            return
        pdf_path = str(candidates[0])

    print(f"测试文件: {Path(pdf_path).name}")
    provider = get_provider(LLM_CFG)
    raw = provider.extract_from_pdf(
        Path(pdf_path).read_bytes(), EXTRACT_USER_PROMPT)

    if raw:
        print(f"\n原始响应（前600字符）:\n{raw[:600]}")
        data = provider._safe_json_parse(raw)
        if data:
            nut = data.get('nutrition') or {}
            print(f"\n✓ 解析成功 | 顶层字段: {len(data)} | 营养字段: {len(nut)}")
            print(f"  蛋白质: {nut.get('protein_g')} g/100mL")
            print(f"  能量:   {nut.get('energy_kJ')} kJ/100mL")
            print(f"  钙:     {nut.get('calcium_mg')} mg/100mL")
            ok, missing = validate(data)
            print(f"  校验:   {'✓ 通过' if ok else '✗ 缺失 ' + str(missing)}")
    else:
        print("✗ 提取失败")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'test':
        pdf = sys.argv[2] if len(sys.argv) > 2 else None
        test_single(pdf)
    else:
        extractor = PDFBatchExtractor(PDF_DIR)
        df = extractor.run()
        if not df.empty:
            print(f"\n✓ 全量提取完成: {len(df)} 款产品，{len(df.columns)} 字段")
            print(f"  输出文件: outputs/products_enriched.xlsx")