from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.llm.router import call_llm
import json

SYSTEM = """You are a symptom triage assistant. Your job is to:
1. Assess symptom severity as: LOW, MODERATE, HIGH, or EMERGENCY
2. Give a plain-English explanation of why
3. Suggest which type of care is appropriate

HARD LIMITS — never violate:
- Never provide a specific diagnosis
- Never name medications or dosages
- Never interpret lab results
- If severity is EMERGENCY, always say: "Call emergency services or go to A&E immediately."

Respond ONLY as valid JSON:
{"severity": "LOW|MODERATE|HIGH|EMERGENCY", "explanation": "...", "recommendation": "...", "confidence": 0.0-1.0}
"""

class SymptomAgent(BaseAgent):
    name = "symptom_triage"
    system_prompt = SYSTEM

    async def run(self, input: AgentInput) -> AgentOutput:
        raw = await call_llm(
            prompt=f"Symptoms reported: {input.message}",
            system=self.system_prompt
        )
        try:
            data = json.loads(raw)
        except Exception:
            data = {"severity": "MODERATE", "explanation": raw, "recommendation": "Consult a doctor.", "confidence": 0.5}

        return AgentOutput(
            response=data.get("explanation", raw),
            confidence=data.get("confidence", 0.7),
            escalate=data.get("severity") in ("HIGH", "EMERGENCY"),
            agent_name=self.name,
            metadata=data
        )