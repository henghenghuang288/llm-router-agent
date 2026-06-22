# LLM Router Agent

> 🇨🇳 [中文版见下方](#中文说明)

A multi-model intelligent routing system. Given a user request, it automatically classifies the intent and routes to the most appropriate model — tracking routing rationale, latency, and token cost for every call.

**Core idea:** Most requests can be classified by fast rules (zero latency, zero cost). Only ambiguous requests need LLM-based semantic classification. This two-tier approach maximizes accuracy while minimizing routing overhead.

## Two-Tier Classification

```
User request
    │
    ▼
[Tier 1] Rule-based classifier (regex patterns, ~0ms)
    │
    ├─ confidence=HIGH → route immediately
    │
    └─ confidence=LOW
           │
           ▼
       [Tier 2] LLM semantic classifier (max_tokens=10, minimal cost)
                    │
                    ▼
               Route to model
```

## Intent Types & Routing

| Intent | Trigger signals | Target model | Rationale |
|--------|----------------|--------------|-----------|
| `code` | "写代码", "debug", "Python"... | DeepSeek | Strong code generation, cost-effective |
| `reasoning` | "分析", "对比", "为什么"... | DeepSeek | Strong reasoning |
| `creative` | "写文案", "爆款标题"... | DeepSeek | Strong language generation |
| `search_needed` | "今天", "最新", "查一下"... | Claude | Has web_search tool |
| `simple_qa` | Default fallback | DeepSeek | Fast & cheap |

## Observability

Every call records:
- Total latency (ms)
- Model latency vs. routing latency (separated)
- Token consumption
- Which classifier fired (rule / llm / rule_fallback)
- Routing decision rationale

This maps directly to "cost governance" and "observability" requirements in FDE job descriptions.

## Stack

Python · FastAPI · asyncio · DeepSeek/Claude/OpenAI-compatible · vanilla HTML/JS

## Quick Start

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxxx
uvicorn backend.main:app --reload
# open http://localhost:8000
```

## Relevance to FDE Roles

This project demonstrates: multi-model ingestion layer · intent routing · per-task model selection · cost governance · observability · graceful degradation (offline simulation when no API key configured).

---

## 中文说明

多模型智能路由系统。输入一个任务请求，两级分类器自动识别意图，路由到最合适的模型，记录每次路由决策理由、耗时、Token 消耗。

**设计核心：**
- 第一级规则分类，零延迟零成本，处理 80%+ 的明确请求
- 第二级 LLM 语义分类，仅在规则置信度低时触发，max_tokens=10 严格控制成本
- 前端实时显示路由预测（输入时即可看到意图分类结果，无需等待）
- 完整可观测性：每次调用记录分类器来源、置信度、耗时、Token 消耗

对应比特鹰等 FDE 岗位 JD 里的"多模型接入层、意图路由、按任务分级选型、成本治理"。
