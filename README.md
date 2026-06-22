# LLM Router Agent · 多模型智能路由

用户发一个任务请求，系统自动识别意图，路由到最合适的模型，记录选择理由、耗时、Token消耗。

## 意图类型与路由逻辑

| 意图 | 触发条件 | 路由目标 | 理由 |
|------|----------|----------|------|
| 代码任务 | 含"写代码/函数/debug/Python"等 | DeepSeek | 代码能力强且便宜 |
| 推理分析 | 含"分析/对比/为什么/方案"等 | DeepSeek | 强推理能力 |
| 创意写作 | 含"写文案/标题/爆款"等 | DeepSeek | 语言表达能力 |
| 需要实时信息 | 含"今天/最新/查一下"等 | Claude | 支持web_search |
| 简单问答 | 默认兜底 | DeepSeek | 快速低成本 |

## 核心设计

- **意图分类零延迟**：纯规则匹配，不调用额外API，不增加成本
- **实时路由预测**：输入时前端实时预览路由决策，无需等待模型调用
- **可观测性**：每次调用记录总耗时、模型延迟、Token消耗、路由理由
- **多提供商降级**：DeepSeek → Claude → 离线模拟，API失败自动降级

## 对应JD能力点

比特鹰/木迪坡等JD里的"多模型接入层、意图路由、按任务分级选型、成本治理、可观测性"的具体实现。

## 运行

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxxx
uvicorn backend.main:app --reload
```

访问 http://localhost:8000
