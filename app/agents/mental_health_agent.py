from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.llm.router import call_llm
import re

SYSTEM = """You are a mental wellness companion. You offer:
- Grounding and breathing exercises
- CBT-inspired reflective prompts
- Mood check-ins using PHQ-9 / GAD-7 style questions
- Compassionate, non-judgmental listening

HARD LIMITS — never violate:
- Never claim to be a therapist or psychologist
- Never recommend or name psychiatric medications
- Never attempt to manage a crisis yourself — always escalate
- If user expresses suicidal ideation, self-harm, or crisis — immediately provide helpline resources
- Always clarify you are an AI wellness tool, not a mental health professional
"""

CRISIS_PATTERNS = [
    r"\b(kill myself|end my life|want to die|suicide|suicidal)\b",
    r"\b(self.harm|hurt myself|cut myself|overdose on purpose)\b",
    r"\b(don't want to be here|no reason to live|better off dead)\b",
]

CRISIS_RESPONSE = """I hear you, and I'm concerned about your safety right now.
Please reach out to a crisis support line immediately:

- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- Befrienders Worldwide: https://www.befrienders.org
- iCall (India): 9152987821
- Crisis Text Line (US): Text HOME to 741741

You are not alone. A trained counsellor is ready to help you right now.
Please contact one of these services immediately."""

class MentalHealthAgent(BaseAgent):
    name = "mental_health"
    system_prompt = SYSTEM

    async def run(self, input: AgentInput) -> AgentOutput:
        # crisis detection — hard escalation, no LLM call
        if self._is_crisis(input.message):
            return AgentOutput(
                response=CRISIS_RESPONSE,
                agent_name=self.name,
                escalate=True,
                confidence=1.0,
                metadata={"crisis_detected": True}
            )

        prompt = f"User message: {input.message}\n\nRespond with compassion and offer a practical wellness technique."
        response = await call_llm(prompt=prompt, system=self.system_prompt)

        # ensure agent never claims to be a therapist
        therapist_pattern = r"\b(I am your therapist|as your therapist|I can be your therapist)\b"
        if re.search(therapist_pattern, response, re.IGNORECASE):
            response = "I am an AI wellness companion, not a therapist. " + response

        return AgentOutput(
            response=response,
            agent_name=self.name,
            escalate=False
        )

    def _is_crisis(self, text: str) -> bool:
        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in CRISIS_PATTERNS
        )