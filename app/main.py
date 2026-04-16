from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.llm.router import router
from app.gateway.policy_agent import PolicyMiddleware
import os

os.environ["ENV"] = "test"
app = FastAPI(title="HealthAI", version="0.1.0", description ="Agentic health recommendation system")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PolicyMiddleware)
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "HealthAI is running", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}