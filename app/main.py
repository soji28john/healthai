from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.gateway.router import router
from app.gateway.policy_agent import PolicyMiddleware

app = FastAPI(title="HealthAI", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PolicyMiddleware)
app.include_router(router, prefix="/api/v1")