from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.llm.router import call_llm

SYSTEM = """You are a wellness advisor. Based on the condition and its severity, recommend:
- Home remedies appropriate to the severity level
- Yoga or breathing exercises (if not contraindicated)
- Exercise recommendations with precautions

HARD LIMITS — never violate:
- Never recommend anything for severity HIGH or EMERGENCY (redirect to doctor)
- Never name OTC medications or supplements by brand
- Never advise against a doctor's existing prescription
- Always include: "These are general wellness suggestions, not medical advice."
"""

class WellnessAgent(BaseAgent):
    name = "wellness"
    system_prompt = SYSTEM

    async def run(self, input: AgentInput) -> AgentOutput:
        severity = input.context.get("severity", "LOW")
        if severity in ("HIGH", "EMERGENCY"):
            return AgentOutput(
                response="For your severity level, please consult a healthcare professional before trying any home remedies.",
                agent_name=self.name,
                escalate=True
            )
        response = await call_llm(
            prompt=f"Condition: {input.message}\nSeverity: {severity}\nProvide wellness recommendations.",
            system=self.system_prompt
        )
        return AgentOutput(response=response, agent_name=self.name)