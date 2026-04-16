from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from groq import Groq

router = APIRouter()

#  Request Model 
class ChatRequest(BaseModel):
    user_id: str
    message: str


# Chat Endpoint 
@router.post("/chat")
async def chat(req: ChatRequest):

    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message")

    # Crisis detection 
    if "kill myself" in req.message.lower():
        return {
            "response": "Please seek immediate professional help.",
            "agent_name": "safety_agent",
            "policy_action": "BLOCKED",
            "resources": ["988 Suicide Hotline"]
        }

    # Try LLM 
    try:
        response_text = await call_llm(req.message)
    except Exception:
        # fallback for tests / no API key
        response_text = f"Basic advice for: {req.message}. Stay hydrated and rest."

    return {
        "response": response_text,
        "agent_name": "health_agent",
        "policy_action": "ALLOWED"
    }


#  Existing LLM endpoint 
@router.get("/llm")
async def llm_endpoint(prompt: str):
    return {"response": await call_llm(prompt)}


# LLM logic 
async def call_llm(prompt: str, system: str = "", max_tokens: int = 1024) -> str:

    # 1. Gemini
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=system or "You are a helpful health assistant. Always add a disclaimer."
        )
        resp = model.generate_content(prompt)
        return resp.text

    except Exception as e:
        print(f"Gemini failed: {e}, trying Groq...")

    # 2. Groq fallback
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"All LLM backends failed: {e}")