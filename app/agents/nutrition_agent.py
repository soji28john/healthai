from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.llm.router import call_llm
from app.rag.pipeline import retrieve_context
import re

SYSTEM = """You are a nutrition advisor. Based on the user's diet description and health conditions:
1. Identify likely nutrient gaps compared to WHO/RDA recommended daily intake
2. Suggest specific foods (not supplements by dosage) to close those gaps
3. Adjust recommendations based on any health conditions provided

HARD LIMITS — never violate:
- Never recommend supplement dosages (e.g. never say '200mg of iron')
- Never advise clinical diet therapy for eating disorders
- Never override a doctor's existing dietary prescription
- Never recommend foods contraindicated by medications listed in context
- Always end with: 'These are general wellness suggestions, not medical advice.'
"""

class NutritionAgent(BaseAgent):
    name = "nutrition"
    system_prompt = SYSTEM

    async def run(self, input: AgentInput) -> AgentOutput:
        
        if not input.message.strip():
            return AgentOutput(
                response="Please describe your typical daily diet so I can analyse your nutrient intake.",
                agent_name=self.name,
                escalate=False
            )
        message = input.message.lower()
        # Retrieve relevant nutrition context from knowledge base
        try:
            rag_context = await retrieve_context(
                f"nutrition deficiency {input.message}", top_k=3
            )
        except Exception:
            rag_context = ""
            
        # try LLM
        try:
            condition = input.context.get("condition", "none")
            age = input.context.get("age", "unknown")
            gender = input.context.get("gender", "unknown")

            prompt = f"""
Medical context from knowledge base:
{rag_context}

User profile: age={age}, gender={gender}, condition={condition}
Diet description: {input.message}

Analyse nutrient gaps and provide food recommendations.
"""
            result = call_llm(prompt=prompt, system=self.system_prompt)
            if hasattr(result, "__await__"):
                response = await result
            else:
                response = result
        # fallback only if LLmM fails
        except Exception:
            response_parts = []
            response_parts.append("Based on your diet description, here are some general suggestions:")
            response_parts.append("- Consider adding more fruits and vegetables for vitamins and fiber.")
            response_parts.append("You may have nutrient gaps in essential vitamins and minerals.")
            
            if "diabetes" in message:
                response_parts.append("Focus on low glycemic index foods and balanced carbohydrate intake.")
                
            if "iron" in message or "anaemia" in message or "anemia" in message:
                response_parts.append("- Include iron-rich foods like spinach, tofu, pumpkin seeds, lentils, or red meat.")
            response = " ".join(response_parts)
            
        # enforce no-dosage hard limit
        dosage_pattern = r"\b\d+\s*(mg|mcg|iu|g)\b"
        if re.search(dosage_pattern, response, re.IGNORECASE):
            response = re.sub(dosage_pattern, "[dosage removed — consult doctor]", response)

        # ensure disclaimer present
        if "not medical advice" not in response.lower():
            response += "\n\nThese are general wellness suggestions, not medical advice."

        return AgentOutput(
            response=response,
            agent_name=self.name,
            escalate=False
        )