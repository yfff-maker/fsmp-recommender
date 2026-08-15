"""
llm_providers.py  ——  可扩展 LLM 接口层
硅基流动（SiliconFlow） / Claude / GPT 三路实现
核心差异：SiliconFlow/GPT 需要先把 PDF 转成图片再送给视觉模型
"""
from abc import ABC, abstractmethod
from typing import Optional
import base64, json, re, time, requests
from pathlib import Path
from loguru import logger

# PDF → 图片转换（PyMuPDF，无需 poppler 依赖）
try:
    import fitz   # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF 未安装，PDF 转图片功能不可用。运行: pip install PyMuPDF")


# ══════════════════════════════════════════════════════════════
# PDF 工具函数
# ══════════════════════════════════════════════════════════════
def pdf_to_images_b64(pdf_bytes: bytes,
                      dpi: int = 150,
                      max_pages: int = 3) -> list[str]:
    """
    将 PDF 的前 N 页转为 base64 编码的 JPEG 图片列表
    用于不支持原生 PDF 的视觉模型（Qwen VL / GPT-4o）
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("请安装 PyMuPDF: pip install PyMuPDF")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_b64 = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        jpg_bytes = pix.tobytes("jpeg")
        images_b64.append(base64.standard_b64encode(jpg_bytes).decode("utf-8"))
    logger.debug(f"PDF转图片: {len(images_b64)}页，"
                 f"每页约{len(images_b64[0])//1024}KB(base64)")
    return images_b64


# ══════════════════════════════════════════════════════════════
# 抽象基类
# ══════════════════════════════════════════════════════════════
class LLMProvider(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def extract_from_pdf(self, pdf_bytes: bytes, prompt: str) -> Optional[str]:
        """从 PDF 文件提取结构化 JSON 文本"""
        ...

    @abstractmethod
    def generate_text(self, system: str, user: str) -> Optional[str]:
        """生成自然语言文本（推荐解释）"""
        ...

    def _safe_json_parse(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        clean = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败: {e} | 前200字符: {clean[:200]}")
        return None

    def _retry(self, func, *args, max_retries=3, delay=8, **kwargs):
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                logger.warning(f"  第{attempt}次返回空，重试中…")
            except requests.exceptions.Timeout:
                logger.warning(f"  第{attempt}次超时")
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else '?'
                logger.warning(f"  第{attempt}次 HTTP {status}: {e}")
                if status == 429:           # rate limit
                    time.sleep(delay * 3)
                elif status in (400, 401):  # 配置错误，直接退出
                    logger.error(f"  致命错误 {status}，停止重试")
                    return None
            except Exception as e:
                logger.warning(f"  第{attempt}次异常: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = delay * attempt
                logger.info(f"  等待 {wait}s 后重试…")
                time.sleep(wait)
        logger.error(f"  已重试 {max_retries} 次，放弃")
        return None


# ══════════════════════════════════════════════════════════════
# 硅基流动（SiliconFlow）— OpenAI 兼容接口
# 视觉模型：Qwen2.5-VL-72B-Instruct（PDF提取）
# 文本模型：Qwen2.5-72B-Instruct（Route C解释）
# ══════════════════════════════════════════════════════════════
class SiliconFlowProvider(LLMProvider):
    """
    硅基流动 API
    - 与 OpenAI chat/completions 接口完全兼容
    - 视觉模型接受 image_url（base64 格式）
    - PDF需先用 PyMuPDF 转为图片
    """

    def _chat_url(self):
        base = self.config.base_url.rstrip('/')
        return f"{base}/chat/completions"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type":  "application/json",
        }

    def _build_image_messages(self, images_b64: list[str],
                               prompt: str) -> list[dict]:
        """构建多图消息（每页PDF作为一张图片）"""
        content = []
        for i, img_b64 in enumerate(images_b64):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"   # 高细节模式，保证数值可读
                }
            })
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def extract_from_pdf(self, pdf_bytes: bytes, prompt: str) -> Optional[str]:
        def _call():
            # 1. PDF → 图片列表
            images_b64 = pdf_to_images_b64(
                pdf_bytes,
                dpi      = getattr(self.config, 'pdf_dpi', 150),
                max_pages= getattr(self.config, 'max_pages', 3),
            )
            # 2. 构建请求
            payload = {
                "model":       self.config.model_name,   # 视觉模型
                "max_tokens":  self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages":    self._build_image_messages(images_b64, prompt),
            }
            resp = requests.post(
                self._chat_url(),
                headers = self._headers(),
                json    = payload,
                timeout = self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content']

        return self._retry(_call,
                           max_retries=getattr(self.config,'retry_max',3),
                           delay=getattr(self.config,'retry_delay',8))

    def generate_text(self, system: str, user: str) -> Optional[str]:
        def _call():
            payload = {
                "model":       self.config.text_model,   # 纯文本模型
                "max_tokens":  self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            }
            resp = requests.post(
                self._chat_url(),
                headers = self._headers(),
                json    = payload,
                timeout = self.config.timeout,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']

        return self._retry(_call)


# ══════════════════════════════════════════════════════════════
# Anthropic Claude — 原生 PDF 支持（document 类型）
# ══════════════════════════════════════════════════════════════
class ClaudeProvider(LLMProvider):

    API_URL = "https://api.anthropic.com/v1/messages"

    def _headers(self):
        return {
            "x-api-key":         self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }

    def extract_from_pdf(self, pdf_bytes: bytes, prompt: str) -> Optional[str]:
        def _call():
            payload = {
                "model":      self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type":       "base64",
                                "media_type": "application/pdf",
                                "data": base64.standard_b64encode(
                                    pdf_bytes).decode("utf-8"),
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            }
            resp = requests.post(
                self.API_URL,
                headers=self._headers(),
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            return resp.json()['content'][0]['text']

        return self._retry(_call)

    def generate_text(self, system: str, user: str) -> Optional[str]:
        def _call():
            payload = {
                "model":      self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "system":     system,
                "messages":   [{"role": "user", "content": user}],
            }
            resp = requests.post(
                self.API_URL, headers=self._headers(),
                json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
            return resp.json()['content'][0]['text']

        return self._retry(_call)


# ══════════════════════════════════════════════════════════════
# OpenAI GPT-4o（OpenAI 官方，与 SiliconFlow 接口同格式）
# ══════════════════════════════════════════════════════════════
class GPTProvider(SiliconFlowProvider):
    """
    GPT-4o 与 SiliconFlow 使用相同的 OpenAI 格式
    直接继承 SiliconFlowProvider，无需重复实现
    """
    pass


# ══════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════
_PROVIDER_MAP = {
    "siliconflow"   : SiliconFlowProvider,
    "claude"        : ClaudeProvider,
    "gpt"           : GPTProvider,
    "qwen_dashscope": SiliconFlowProvider,  # DashScope 也是 OpenAI 兼容
    "deepseek"      : SiliconFlowProvider,   # ← 新增这行，复用同一实现
}

def get_provider(config) -> LLMProvider:
    cls = _PROVIDER_MAP.get(config.provider)
    if cls is None:
        raise ValueError(
            f"未知 provider: '{config.provider}'. "
            f"可用: {list(_PROVIDER_MAP.keys())}"
        )
    logger.info(f"LLM Provider: [{config.provider}] "
                f"视觉={config.model_name} 文本={config.text_model}")
    return cls(config)


# ══════════════════════════════════════════════════════════════
# 测试：验证 API 连通性（不消耗 PDF token）
# ══════════════════════════════════════════════════════════════
def test_connection(cfg=None):
    """
    用一条简单文本请求验证 API key 和网络是否正常
    不发 PDF，节省费用
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    if cfg is None:
        from config import LLM_CFG
        cfg = LLM_CFG

    if not cfg.api_key:
        print("✗ API key 未配置，请在 config.py 或环境变量中设置")
        return False

    provider = get_provider(cfg)
    print(f"测试连通性: {cfg.provider} / {cfg.text_model}")
    result = provider.generate_text(
        system="你是助手，只回复 JSON",
        user='返回 {"status": "ok", "provider": "' + cfg.provider + '"}'
    )
    if result:
        data = provider._safe_json_parse(result)
        if data and data.get('status') == 'ok':
            print(f"✓ API 连通正常 | 响应: {data}")
            return True
        else:
            print(f"⚠ API 有响应但格式异常: {result[:100]}")
            return False
    else:
        print("✗ API 无响应，请检查 key 和网络")
        return False


def test_pdf_extraction(pdf_path: str, cfg=None):
    """
    用单个 PDF 测试提取效果
    用法: python llm_providers.py /path/to/test.pdf
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    if cfg is None:
        from config import LLM_CFG
        cfg = LLM_CFG
    from pdf_extractor import EXTRACT_USER_PROMPT

    pdf_bytes = Path(pdf_path).read_bytes()
    provider  = get_provider(cfg)

    print(f"测试 PDF 提取: {Path(pdf_path).name}")
    print(f"提供商: {cfg.provider} / {cfg.model_name}")
    raw = provider.extract_from_pdf(pdf_bytes, EXTRACT_USER_PROMPT)
    if raw:
        print(f"\n原始响应（前500字符）:\n{raw[:500]}")
        data = provider._safe_json_parse(raw)
        if data:
            print(f"\n✓ JSON解析成功，共 {len(data)} 个顶层字段")
            nut = data.get('nutrition', {}) or {}
            print(f"  营养字段: {len(nut)} 项")
            print(f"  蛋白质(100mL): {nut.get('protein_g', 'N/A')} g")
            print(f"  能量(100mL):   {nut.get('energy_kJ', 'N/A')} kJ")
            print(f"  钙:            {nut.get('calcium_mg', 'N/A')} mg")
    else:
        print("✗ 提取失败")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        test_pdf_extraction(sys.argv[1])
    else:
        test_connection()
