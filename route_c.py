"""
route_c.py  ——  LLM 推荐解释层（真实调用版）
提供商：硅基流动 Qwen2.5-72B-Instruct（文本模型）
输入：Route A/B 的 Top-K 产品 + 客户信息 + 临床规则
输出：专业推荐理由 + 组合方案 + 临床警示 + 监测建议
"""
from __future__ import annotations
import json, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config       import LLM_CFG
from llm_providers import get_provider


# ══════════════════════════════════════════════════════════════
# 输出结构
# ══════════════════════════════════════════════════════════════
@dataclass
class ProductRec:
    rank        : int
    name        : str
    reason      : str    # 推荐理由（≤50字）
    key_benefit : str    # 核心优势（≤10字）

@dataclass
class RouteCOutput:
    recommendations   : list[ProductRec]
    combination_advice: str
    clinical_warnings : list[str]
    monitoring        : list[str]
    confidence        : str          # high / medium / low
    reviewer_note     : str          # LLM 对算法排序的纠错意见
    provider_used     : str
    raw_json          : dict


# ══════════════════════════════════════════════════════════════
# Prompt 设计
# ══════════════════════════════════════════════════════════════
SYSTEM_RERANK = """\
你是拥有十年临床经验的注册临床营养师，专长特殊医学用途配方食品（特医食品）。
根据临床适应症对候选产品重新排序，最适合的排在最前。
tier S = ESPGHAN/ESPEN指南明确推荐的临床首选；tier A = 次选；tier C = 规则外。
只输出重排后的注册证号 JSON 数组，不要任何解释或 markdown。"""

SYSTEM = """\
你是拥有十年临床经验的注册临床营养师，专长特殊医学用途配方食品（特医食品）。
请根据算法预筛选结果，为医生或营养师提供简洁专业的推荐说明。

输出规则（严格遵守）：
1. 推荐理由必须引用具体临床依据（ESPGHAN/ESPEN/GB标准），不编造
2. 每条 reason 不超过50个汉字
3. key_benefit 不超过10个汉字
4. 若算法排序与临床判断不符，在 reviewer_note 中明确指出
5. 只输出 JSON，不要任何 markdown 或解释文字"""

def _build_prompt(client: dict, top5: list[dict],
                  active_rules: dict, notes: list, combination: list) -> str:

    # 精简产品信息（控制 token）
    prod_info = []
    for p in top5:
        prod_info.append({
            "排名":   p['rank'],
            "产品名": p['name'][:25],
            "类别":   p['category'],
            "级别":   p['tier'],          # S/A/C
            "综合分": p['score'],
            "蛋白g":  p['protein_g'],
            "碳水g":  p['carb_g'],
            "适应症": p['indication'][:50],
        })

    # 临床规则摘要
    rules_info = {}
    for cond, rule in active_rules.items():
        rules_info[cond] = {
            "首选": rule.get('preferred', []),
            "依据": rule.get('evidence', '')[:80],
        }

    return f"""
【客户信息】
年龄段：{client.get('age_group')}
主要病症：{client.get('diseases')}
备注：{client.get('query', '')}

【临床规则（ESPGHAN/ESPEN依据）】
{json.dumps(rules_info, ensure_ascii=False)}

【算法预筛 Top-5 产品】（tier: S=临床首选 A=次选 C=规则外）
{json.dumps(prod_info, ensure_ascii=False, indent=2)}

【系统提示】
{json.dumps(notes + combination, ensure_ascii=False)}

请输出以下 JSON（只输出JSON，无任何其他内容）：
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "产品名",
      "reason": "推荐理由，引用具体标准，≤50字",
      "key_benefit": "核心优势≤10字"
    }}
  ],
  "combination_advice": "若需组合使用说明方案（≤80字），不需要则空字符串",
  "clinical_warnings": ["警示1≤30字", "警示2"],
  "monitoring": ["监测指标1", "监测指标2"],
  "confidence": "high或medium或low",
  "reviewer_note": "若算法排序有临床问题在此指出，否则空字符串"
}}"""


# ══════════════════════════════════════════════════════════════
# Route C 主类
# ══════════════════════════════════════════════════════════════
class RouteC:

    def __init__(self, llm_cfg=None):
        self.cfg      = llm_cfg or LLM_CFG
        self.provider = get_provider(self.cfg)

    def _check_key(self) -> bool:
        key = self.cfg.api_key or ''
        if not key or key in ('your_key_here', ''):
            logger.warning("[C] API key 未配置，跳过 LLM 解释层")
            return False
        return True

    def rerank(self, client: dict, ranked_df,
               filter_result, top_n: int = 10) -> list:
        """
        LLM 重排序：对 A+B Top-N 候选重新打分排序。
        Returns: 重排后的注册证号列表（失败时 fallback 到原始顺序）
        """
        if not self._check_key():
            return ranked_df['注册证号'].head(top_n).tolist()

        candidates = ranked_df.head(top_n)

        cand_lines = []
        for _, row in candidates.iterrows():
            reg   = str(row.get('注册证号', ''))
            name  = str(row.get('产品名称', ''))[:20]
            cat   = str(row.get('产品类别', ''))
            tier  = str(row.get('A级别', 'C'))
            score = float(row.get('综合得分', 0))
            cand_lines.append(
                f'"{reg}" | {name} | {cat} | tier={tier} | 算法得分={score:.2f}')

        prompt = (
            f"根据临床规则对以下候选产品重新排序，把最适合的排在前面。\n\n"
            f"客户信息：\n"
            f"  病症：{client.get('diseases')}\n"
            f"  年龄段：{client.get('age_group')}\n"
            f"  备注：{client.get('query', '')}\n\n"
            f"候选产品（注册证号 | 产品名 | 类别 | tier | 算法得分）：\n"
            + "\n".join(cand_lines) +
            f"\n\ntier说明：S=临床首选，A=临床次选，C=规则外\n"
            f"请综合临床适应性重新排序，不必完全跟随算法得分顺序。\n\n"
            f"只输出重排后的注册证号列表，JSON数组格式（包含全部{len(candidates)}个证号）：\n"
            f'["证号1","证号2",...]'
        )

        raw = self.provider.generate_text(SYSTEM_RERANK, prompt)
        if not raw:
            logger.warning("[C] rerank: API 无响应，保持原始排序")
            return candidates['注册证号'].tolist()

        try:
            clean = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
            match = re.search(r'\[.*\]', clean, re.DOTALL)
            if match:
                new_order = json.loads(match.group())
                if isinstance(new_order, list) and len(new_order) > 0:
                    logger.info(f"[C] rerank 完成，重排 {len(new_order)} 个产品")
                    return [str(x) for x in new_order]
        except Exception as e:
            logger.warning(f"[C] rerank 解析失败({e}): {raw[:120]}")

        return candidates['注册证号'].tolist()

    def run(self, client: dict, ranked_df,
            filter_result) -> Optional[RouteCOutput]:

        if not self._check_key():
            return None

        logger.info(f"[C] {self.cfg.provider} / {self.cfg.text_model}")

        # 准备 Top-5 数据
        top5_rows = []
        for i, (_, row) in enumerate(ranked_df.head(5).iterrows(), 1):
            top5_rows.append({
                'rank':       i,
                'name':       str(row.get('产品名称', '')),
                'category':   str(row.get('产品类别', '')),
                'tier':       str(row.get('A级别', 'C')),
                'score':      float(row.get('综合得分', 0)),
                'protein_g':  float(row.get('蛋白质_g', 0) or 0),
                'carb_g':     float(row.get('碳水_g', 0) or 0),
                'indication': str(row.get('适用人群', ''))[:50],
            })

        # 构建 active_rules 字典
        active_rules = {}
        for cond, rule in filter_result.active_rules.items():
            active_rules[cond] = rule.__dict__ if hasattr(rule, '__dict__') else rule

        prompt = _build_prompt(
            client       = client,
            top5         = top5_rows,
            active_rules = active_rules,
            notes        = filter_result.notes,
            combination  = filter_result.combination,
        )

        # 调用 LLM（文本模型）
        raw = self.provider.generate_text(SYSTEM, prompt)
        if not raw:
            logger.error("[C] API 无响应")
            return None

        # 解析 JSON
        data = self.provider._safe_json_parse(raw)
        if not data:
            # 尝试直接解析
            try:
                clean = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
                data  = json.loads(clean)
            except Exception:
                logger.error(f"[C] JSON 解析失败: {raw[:100]}")
                return None

        # 构建输出
        recs = []
        for r in data.get('recommendations', []):
            recs.append(ProductRec(
                rank        = r.get('rank', 0),
                name        = r.get('name', ''),
                reason      = r.get('reason', ''),
                key_benefit = r.get('key_benefit', ''),
            ))

        result = RouteCOutput(
            recommendations    = recs,
            combination_advice = data.get('combination_advice', ''),
            clinical_warnings  = data.get('clinical_warnings', []),
            monitoring         = data.get('monitoring', []),
            confidence         = data.get('confidence', 'medium'),
            reviewer_note      = data.get('reviewer_note', ''),
            provider_used      = f"{self.cfg.provider}/{self.cfg.text_model}",
            raw_json           = data,
        )

        if result.reviewer_note:
            logger.warning(f"[C] 模型纠错：{result.reviewer_note}")

        logger.info(f"[C] 完成 | 置信度={result.confidence} | "
                    f"{len(result.recommendations)}条推荐理由")
        return result

    def format_text(self, output: RouteCOutput, client_name: str) -> str:
        """格式化为报告文字"""
        if not output:
            return "（LLM解释层未激活）"
        lines = [
            f"{'='*50}",
            f"  {client_name} · AI 推荐说明",
            f"  模型：{output.provider_used}  置信度：{output.confidence}",
            f"{'='*50}",
        ]
        for rec in output.recommendations:
            lines += [
                f"\n  #{rec.rank}  {rec.name}",
                f"      理由：{rec.reason}",
                f"      优势：{rec.key_benefit}",
            ]
        if output.combination_advice:
            lines += ["", f"  💡 组合方案：{output.combination_advice}"]
        if output.clinical_warnings:
            lines.append("\n  ⚠  临床警示：")
            for w in output.clinical_warnings:
                lines.append(f"      · {w}")
        if output.monitoring:
            lines.append("\n  📋 监测指标：")
            for m in output.monitoring:
                lines.append(f"      · {m}")
        if output.reviewer_note:
            lines.append(f"\n  🔍 AI纠错：{output.reviewer_note}")
        return '\n'.join(lines)
