"""
retry_failed.py  ——  重跑失败和不完整的 PDF
策略：加长超时 + 降低分辨率（减少 token，减少超时概率）
"""
import json, sys, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config import (LLM_CFG, PDF_DIR, OUTPUT_DIR,
                    LOG_DIR, CACHE_DIR, EXT_CFG)
from llm_providers import get_provider
from pdf_extractor import EXTRACT_USER_PROMPT, validate

# ── 重试参数（更保守，减少超时）──────────────────────────────
RETRY_TIMEOUT  = 180    # 3 分钟超时（原来 90s）
RETRY_DPI      = 120    # 降低分辨率（原来 150），token 减少约 35%
RETRY_MAX      = 4      # 最多重试 4 次
RETRY_DELAY    = 15     # 重试间隔 15s

def main():
    # ── 读取现有进度 ────────────────────────────────────────
    cache_path = CACHE_DIR / "extraction_progress.json"
    if not cache_path.exists():
        print("✗ 找不到进度文件，请先运行 pdf_extractor.py")
        return
    
    prog      = json.loads(cache_path.read_text(encoding='utf-8'))
    done      = prog.get('done', {})
    failed    = prog.get('failed', [])
    
    # 找不完整的
    incomplete = [
        fname for fname, data in done.items()
        if data.get('_incomplete')
    ]
    
    print(f"{'='*52}")
    print(f"  失败文件:  {len(failed)} 份")
    print(f"  不完整:    {len(incomplete)} 份")
    print(f"  合计重跑:  {len(set(failed+incomplete))} 份")
    print(f"  超时设置:  {RETRY_TIMEOUT}s（原 90s）")
    print(f"  图片分辨率: {RETRY_DPI} dpi（原 150 dpi）")
    print(f"{'='*52}")
    
    # ── 配置：加长超时，降低 dpi ─────────────────────────────
    from copy import deepcopy
    retry_cfg         = deepcopy(LLM_CFG)
    retry_cfg.timeout = RETRY_TIMEOUT
    retry_cfg.pdf_dpi = RETRY_DPI
    retry_cfg.max_pages = 3
    
    provider = get_provider(retry_cfg)
    provider.config.pdf_dpi   = RETRY_DPI
    provider.config.max_pages = 3

    targets = list(dict.fromkeys(failed + incomplete))  # 去重保序
    success, still_fail = 0, []

    for fname in tqdm(targets, desc="重试中", unit="份"):
        pdf_path = PDF_DIR / fname
        if not pdf_path.exists():
            logger.error(f"文件不存在: {fname}")
            still_fail.append(fname)
            continue

        logger.info(f"重试: {fname}")
        pdf_bytes = pdf_path.read_bytes()

        result = None
        for attempt in range(1, RETRY_MAX + 1):
            try:
                raw = provider.extract_from_pdf(pdf_bytes, EXTRACT_USER_PROMPT)
                if raw:
                    data = provider._safe_json_parse(raw)
                    if data:
                        ok, missing = validate(data)
                        data['_pdf_file']   = fname
                        data['_incomplete'] = not ok
                        if not ok:
                            data['_missing'] = missing
                            logger.warning(f"  仍不完整: {missing}")
                        result = data
                        break
            except Exception as e:
                logger.warning(f"  第{attempt}次失败: {e}")
                if attempt < RETRY_MAX:
                    time.sleep(RETRY_DELAY * attempt)

        if result:
            done[fname]     = result
            if fname in failed: failed.remove(fname)
            success += 1
            logger.info(f"  ✓ 成功")
        else:
            still_fail.append(fname)
            logger.error(f"  ✗ 仍然失败")

        time.sleep(1)

    # ── 更新进度 ─────────────────────────────────────────────
    prog['done']   = done
    prog['failed'] = still_fail
    cache_path.write_text(
        json.dumps(prog, ensure_ascii=False, indent=2), encoding='utf-8')

    # ── 重新生成 Excel ────────────────────────────────────────
    rows = []
    for data in done.values():
        row = {k: v for k, v in data.items() if k != 'nutrition'}
        for k, v in (data.get('nutrition') or {}).items():
            row[f'nut_{k}'] = v
        rows.append(row)
    df  = pd.DataFrame(rows)
    out = OUTPUT_DIR / "products_enriched.xlsx"
    df.to_excel(out, index=False)

    # ── 汇总 ────────────────────────────────────────────────
    incomplete_now = sum(1 for d in done.values() if d.get('_incomplete'))
    print(f"\n{'='*52}")
    print(f"  本次成功补救: {success}/{len(targets)} 份")
    print(f"  当前总完成:   {len(done)}/182 份")
    print(f"  仍然失败:     {len(still_fail)} 份")
    print(f"  仍然不完整:   {incomplete_now} 份")
    print(f"  Excel 已更新: {out}")
    if still_fail:
        print(f"\n  仍失败文件（可能是 PDF 本身有问题）:")
        for f in still_fail:
            print(f"    {f}")
    print(f"{'='*52}")


if __name__ == '__main__':
    main()
