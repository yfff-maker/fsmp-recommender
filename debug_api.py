"""
debug_api.py v2  ——  测试可用视觉模型 + 推荐最优选择
"""
import requests, sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_CFG

key  = LLM_CFG.api_key or os.environ.get("SILICONFLOW_API_KEY","")
BASE = "https://api.siliconflow.cn/v1"
H    = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# 最小合法 JPEG（1x1白色像素）
TINY = ("/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH"
        "BwYIDAoMCwsKCwsNCxAQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAAR"
        "CAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
        "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAA"
        "AAAAAAAA/9oADAMBAAIRAxEAPwCwABmX/9k=")

# 你账号实际可用的视觉模型（按质量排序）
CANDIDATES = [
    ("Qwen/Qwen3-VL-32B-Instruct",   "★★★★  最强，推荐PDF提取"),
    ("Qwen/Qwen3-VL-32B-Thinking",   "★★★★  思考模式，慢但准"),
    ("Qwen/Qwen3-VL-30B-A3B-Instruct","★★★   MoE精简，速度快"),
    ("Qwen/Qwen3-VL-8B-Instruct",    "★★★   轻量，可能免费"),
    ("PaddlePaddle/PaddleOCR-VL-1.5","★★★   专用OCR，可能免费"),
]

print("=" * 58)
print("  测试各视觉模型可用性")
print("=" * 58)

results = []
for model, desc in CANDIDATES:
    print(f"\n  测试: {model}")
    print(f"  {desc}")
    try:
        r = requests.post(
            f"{BASE}/chat/completions", headers=H, timeout=30,
            json={
                "model": model, "max_tokens": 5,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{TINY}"}},
                    {"type": "text", "text": "OK"}
                ]}]
            }
        )
        if r.status_code == 200:
            print(f"  ✓ 可用！")
            results.append((model, desc, True))
        elif r.status_code == 403:
            print(f"  ✗ 403 无权限")
            results.append((model, desc, False))
        elif r.status_code == 402:
            print(f"  ✗ 402 余额不足（充值后可用）")
            results.append((model, desc, False))
        else:
            msg = r.json().get('message','') if r.text else ''
            print(f"  ? {r.status_code}: {msg[:60]}")
            results.append((model, desc, False))
    except Exception as e:
        print(f"  ! 错误: {e}")
        results.append((model, desc, False))
    time.sleep(1)

print("\n" + "=" * 58)
print("  结果汇总")
print("=" * 58)
ok = [r for r in results if r[2]]
if ok:
    best = ok[0]
    print(f"\n  推荐使用: {best[0]}")
    print(f"  {best[1]}")
    print(f"\n  在 config.py 中更新这一行：")
    print(f'  "vision_model": "{best[0]}",')
else:
    print("\n  ✗ 所有视觉模型均不可用")
    print("  解决方案：")
    print("  1. 充值 20 元（足够跑完 182 份 PDF）")
    print("     https://cloud.siliconflow.cn/account/finance")
    print("  2. 或者使用 Claude API（原生支持PDF，无需转图片）")