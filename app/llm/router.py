from fastapi import APIRouter
import os
import google.generativeai as genai
from groq import Groq

router = APIRouter()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@router.get("/llm")
async def llm_endpoint(prompt: str):
    return {"response": await call_llm(prompt)}

async def call_llm(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Try Gemini Flash → Groq → raise. Ollama fallback can be added locally."""
    # 1. Gemini Flash
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=system or "You are a helpful health assistant. Always add a medical disclaimer."
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