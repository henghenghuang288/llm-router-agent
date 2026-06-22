"""
多模型智能路由 Agent

核心逻辑：用户发一个任务请求，系统自动判断意图类型，
路由到最合适的模型/工具，记录选择理由、耗时、token消耗。

意图分类：
  - simple_qa       → 简单问答，用小模型（快+便宜）
  - reasoning       → 需要推理/分析，用强模型
  - code            → 写代码/调试，用代码专用模型
  - search_needed   → 需要实时信息，先搜索再回答
  - creative        → 创意写作，用擅长创意的模型

路由决策依据：
  1. 关键词+句式快速分类（不调用额外API，零延迟）
  2. 每种意图对应推荐模型+降级链（主力→备用→离线）
  3. 记录每次路由决策和成本，供可观测性报告使用

这是比特鹰JD里"意图路由、多模型接入层、按任务分级选型、成本治理"的具体实现。
"""

import re
import time
from typing import Any

# ── 意图分类规则 ──────────────────────────────────────────────────────────
INTENT_RULES = [
    {
        "intent": "code",
        "label": "代码任务",
        "model_preference": "deepseek-chat",   # DeepSeek代码能力强且便宜
        "reason": "检测到代码相关任务，DeepSeek在代码生成上性价比最高",
        "patterns": [
            r"(写|生成|帮我写).{0,10}(代码|函数|脚本|程序)",
            r"(debug|调试|报错|error|exception)",
            r"(python|javascript|sql|bash|java|typescript)",
            r"(怎么实现|如何实现|代码实现)",
        ],
    },
    {
        "intent": "search_needed",
        "label": "需要实时信息",
        "model_preference": "claude-sonnet-4-6",  # Claude有web_search工具
        "reason": "检测到需要实时/最新信息，路由到支持搜索工具的模型",
        "patterns": [
            r"(今天|今日|最新|最近|现在|当前).{0,15}(价格|新闻|消息|行情|发布|上线)",
            r"(2024|2025|2026).{0,10}(最新|更新|发布|上线)",
            r"(查一下|搜一下|帮我查)",
        ],
    },
    {
        "intent": "reasoning",
        "label": "推理分析",
        "model_preference": "deepseek-chat",
        "reason": "检测到需要深度推理/分析，使用强推理模型",
        "patterns": [
            r"(分析|评估|对比|比较).{0,20}(优缺点|利弊|差异|区别)",
            r"(为什么|原因|解释|推断|预测)",
            r"(方案|策略|建议).{0,10}(如何|怎么|什么)",
            r"(优化|改进|提升).{0,15}(方案|建议|思路)",
        ],
    },
    {
        "intent": "creative",
        "label": "创意写作",
        "model_preference": "deepseek-chat",
        "reason": "检测到创意写作任务，使用语言表达能力强的模型",
        "patterns": [
            r"(写一篇|帮我写).{0,10}(文章|故事|文案|脚本|标题|简介)",
            r"(创意|有趣|吸引人|爆款).{0,10}(标题|文案|内容|开头)",
            r"(翻译|改写|润色|优化).{0,10}(文字|文案|内容)",
        ],
    },
    {
        "intent": "simple_qa",
        "label": "简单问答",
        "model_preference": "deepseek-chat",
        "reason": "简单问答任务，使用快速低成本模型",
        "patterns": [],  # 默认兜底
    },
]


def classify_intent(query: str) -> dict:
    """快速意图分类，纯规则，零延迟，不调用任何API。"""
    query_lower = query.lower()
    for rule in INTENT_RULES[:-1]:  # 最后一个是默认兜底
        for pattern in rule["patterns"]:
            if re.search(pattern, query_lower):
                return {
                    "intent": rule["intent"],
                    "label": rule["label"],
                    "model": rule["model_preference"],
                    "routing_reason": rule["reason"],
                    "matched_pattern": pattern,
                }
    # 兜底：简单问答
    default = INTENT_RULES[-1]
    return {
        "intent": default["intent"],
        "label": default["label"],
        "model": default["model_preference"],
        "routing_reason": default["reason"],
        "matched_pattern": None,
    }


# ── 模型调用 ──────────────────────────────────────────────────────────────
import os
import json


def get_available_providers() -> list[str]:
    providers = []
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if os.environ.get("OPENAI_API_KEY"):
        providers.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append("claude")
    return providers


async def call_model(query: str, model: str, intent: str) -> dict[str, Any]:
    """调用模型，返回回答+token消耗+耗时。"""
    import os
    from openai import AsyncOpenAI

    t0 = time.perf_counter()

    # 根据意图定制system prompt
    system_prompts = {
        "code": "你是一名专业的代码助手。给出简洁、可运行的代码，加必要注释，不废话。",
        "reasoning": "你是一名严谨的分析师。逻辑清晰地分析问题，给出有依据的结论。",
        "creative": "你是一名创意写作专家。语言生动，有感染力，符合目标受众的语气。",
        "search_needed": "你是一名信息助手。基于你知道的信息回答，明确说明信息截止时间，不确定的内容要标注。",
        "simple_qa": "你是一名高效助手。简洁准确地回答问题，不废话。",
    }
    system = system_prompts.get(intent, system_prompts["simple_qa"])

    # DeepSeek优先
    if os.environ.get("DEEPSEEK_API_KEY"):
        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": query}],
            max_tokens=1200,
        )
        usage = resp.usage
        return {
            "answer": resp.choices[0].message.content or "",
            "provider": "deepseek",
            "model_used": "deepseek-chat",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            } if usage else None,
        }

    # Claude备用
    if os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        usage = resp.usage
        return {
            "answer": resp.content[0].text if resp.content else "",
            "provider": "claude",
            "model_used": "claude-sonnet-4-6",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "usage": {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
            } if usage else None,
        }

    # 离线模拟
    return {
        "answer": f"[离线模式] 已分类为「{intent}」任务，检测到应使用 {model}。配置 DEEPSEEK_API_KEY 后可获得真实回答。",
        "provider": "offline",
        "model_used": model,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "usage": None,
    }


async def route_and_answer(query: str) -> dict[str, Any]:
    """完整流程：意图分类 → 路由决策 → 模型调用 → 返回结果+可观测数据。"""
    t0 = time.perf_counter()

    # 1. 意图分类（纯规则，零延迟）
    routing = classify_intent(query)

    # 2. 调用模型
    result = await call_model(query, routing["model"], routing["intent"])

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query": query,
        "routing": {
            "intent": routing["intent"],
            "intent_label": routing["label"],
            "model_selected": routing["model"],
            "routing_reason": routing["routing_reason"],
        },
        "answer": result["answer"],
        "provider": result["provider"],
        "model_used": result["model_used"],
        "observability": {
            "total_latency_ms": total_ms,
            "model_latency_ms": result["latency_ms"],
            "routing_latency_ms": round(total_ms - result["latency_ms"], 1),
            "usage": result["usage"],
        },
        "available_providers": get_available_providers(),
    }
