import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .router import route_and_answer, classify_intent, get_available_providers

app = FastAPI(title="LLM Router Agent")

class QueryRequest(BaseModel):
    query: str

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
