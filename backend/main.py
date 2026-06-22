import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .router import route_and_answer, classify_intent, get_available_providers

app = FastAPI(title="LLM Router Agent")

class QueryRequest(BaseModel):
    query: str

@app.get("/api/info")
def info():
    """项目信息接口——面试官打开API第一眼就能看到这是什么。"""
    return {
        "name": "多模型智能路由 Agent",
        "name_en": "LLM Router Agent",
        "version": "1.0.0",
        "description": "两级意图分类（规则+LLM兜底），路由到最合适的模型，记录成本和耗时",
        "description_en": "Two-tier intent classification (rules + LLM fallback), routes to optimal model, tracks cost & latency",
        "architecture": "two-tier classification, multi-model routing, full observability",
        "github": "https://github.com/henghenghuang288/llm-router-agent",
        "endpoints": [
                "/api/health",
                "/api/route",
                "/api/classify"
        ]
}


@app.get("/api/health")
def health():
    providers = get_available_providers()
    return {"status": "ok", "live_mode": len(providers) > 0, "providers": providers}

@app.post("/api/route")
async def route(body: QueryRequest):
    if not body.query.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="请输入任务内容")
    return await route_and_answer(body.query.strip())

@app.post("/api/classify")
def classify(body: QueryRequest):
    """只做意图分类，不调用模型，用于前端实时预览路由决策。"""
    return classify_intent(body.query.strip())

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
