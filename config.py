"""
config.py  ——  全局配置
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
PDF_DIR      = DATA_DIR / "pdfs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
LOG_DIR      = PROJECT_ROOT / "logs"
CACHE_DIR    = PROJECT_ROOT / "cache"
for d in [DATA_DIR, PDF_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RESULT1_PATH  = DATA_DIR / "result1.xlsx"
RESULT2_PATH  = DATA_DIR / "result2.xlsx"
ENRICHED_PATH = DATA_DIR / "products_enriched.xlsx"


@dataclass
class LLMConfig:
    provider    : str           = "siliconflow"
    model_name  : str           = "Qwen/Qwen2.5-VL-72B-Instruct"
    text_model  : str           = "Qwen/Qwen2.5-72B-Instruct"
    api_key     : Optional[str] = None
    base_url    : str           = ""
    max_tokens  : int           = 2000
    temperature : float         = 0.1
    timeout     : int           = 90
    pdf_dpi     : int           = 150
    max_pages   : int           = 3
    retry_max   : int           = 3
    retry_delay : int           = 8

    PRESETS = {
        "siliconflow": {
            "base_url"    : "https://api.siliconflow.cn/v1",
            "vision_model": "deepseek-ai/DeepSeek-V4-Pro",   # 或你需要的视觉模型
            "text_model"  : "deepseek-ai/DeepSeek-V4-Pro",
            "env_key"     : "sk-bfebkmzrimcietonbudesdcicfobxjdytszbrlcxzyymiwzd",
        },
        "claude": {
            "base_url"    : "https://api.anthropic.com/v1/messages",
            "vision_model": "claude-sonnet-4-20250514",
            "text_model"  : "claude-sonnet-4-20250514",
            "env_key"     : "ANTHROPIC_API_KEY",
        },
        "gpt": {
            "base_url"    : "https://api.openai.com/v1",
            "vision_model": "gpt-4o",
            "text_model"  : "gpt-4o",
            "env_key"     : "OPENAI_API_KEY",
        },
    }

    @classmethod
    def from_provider(cls, provider: str, api_key: str = None):
        preset = cls.PRESETS.get(provider, {})
        key    = api_key or os.environ.get(preset.get("env_key", ""), "")
        return cls(
            provider   = provider,
            model_name = preset.get("vision_model", ""),
            text_model = preset.get("text_model", ""),
            api_key    = key,
            base_url   = preset.get("base_url", ""),
        )


# ══════════════════════════════════════════════════════════════
# ★ 当前使用的提供商和 Key ★
# ══════════════════════════════════════════════════════════════
LLM_CFG = LLMConfig.from_provider(
    "siliconflow",
    api_key="sk-bfebkmzrimcietonbudesdcicfobxjdytszbrlcxzyymiwzd",
)


@dataclass
class MindSporeConfig:
    device    : str = "CPU"
    precision : str = "float32"

@dataclass
class RecConfig:
    top_k              : int   = 8
    route_b_nut_weight : float = 1.5
    route_b_cat_weight : float = 3.0
    tier_weights       : dict  = field(default_factory=lambda: {
        "S": 1.00, "A": 0.82, "C": 0.50
    })
    use_enriched_data  : bool  = True

@dataclass
class ExtractConfig:
    batch_size   : int  = 1
    retry_max    : int  = 3
    retry_delay  : int  = 8
    save_every   : int  = 10
    pdf_dpi      : int  = 150
    max_pages    : int  = 3

MS_CFG  = MindSporeConfig()
REC_CFG = RecConfig()
EXT_CFG = ExtractConfig()

if __name__ == '__main__':
    print(f"提供商:   {LLM_CFG.provider}")
    print(f"视觉模型: {LLM_CFG.model_name}")
    print(f"文本模型: {LLM_CFG.text_model}")
    k = LLM_CFG.api_key or ""
    print(f"API Key:  {'✓ ' + k[:10]+'...'+k[-4:] if k else '✗ 未配置'}")