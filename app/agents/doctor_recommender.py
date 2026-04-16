from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.llm.router import call_llm

SYSTEM = """You are a doctor recommendation assistant. Based on the user's symptoms and severity:
1. Suggest the appropriate medical specialty to consult
2. Explain why that specialty is relevant
3. Describe what to expect at the appointment

HARD LIMITS — never violate:
- Never recommend a specific named doctor or clinic
- Never guarantee insurance coverage — always say 'verify with your provider'
- Never advise delaying emergency care to find a doctor
- Always remind the user this is a suggestion, not a referral
"""

SPECIALTY_MAP = {
    "chest pain": "Cardiologist",
    "heart": "Cardiologist",
    "skin": "Dermatologist",
    "rash": "Dermatologist",
    "mental": "Psychiatrist or Psychologist",
    "anxiety": "Psychiatrist or Psychologist",
    "depression": "Psychiatrist or Psychologist",
    "bone": "Orthopedic Specialist",
    "joint": "Orthopedic Specialist",
    "eye": "Ophthalmologist",
    "vision": "Ophthalmologist",
    "stomach": "Gastroenterologist",
    "digestion": "Gastroenterologist",
    "child": "Pediatrician",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "kidney": "Nephrologist",
    "lung": "Pulmonologist",
    "breathing": "Pulmonologist",
}

class DoctorRecommenderAgent(BaseAgent):
    name = "doctor_recommender"
    system_prompt = SYSTEM

    async def run(self, input: AgentInput) -> AgentOutput:
        # quick specialty lookup from keyword map
        suggested_specialty = self._suggest_specialty(input.message)

        prompt = f"""
Symptoms: {input.message}
Severity: {input.context.get('severity', 'MODERATE')}
Suggested specialty (from keyword match): {suggested_specialty}

Provide a clear recommendation for which type of doctor to see and why.
Include what to tell the doctor and what tests might be ordered.
"""
        response = await call_llm(prompt=prompt, system=self.system_prompt)

        if "not medical advice" not in response.lower():
            response += "\n\nNote: This is a general suggestion. Please verify insurance coverage with your provider."

        return AgentOutput(
            response=response,
            agent_name=self.name,
            escalate=False,
            metadata={"suggested_specialty": suggested_specialty}
        )

    def _suggest_specialty(self, text: str) -> str:
        text_lower = text.lower()
        for keyword, specialty in SPECIALTY_MAP.items():
            if keyword in text_lower:
                return specialty
        return "General Practitioner (start here)"