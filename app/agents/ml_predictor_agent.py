from app.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from app.ml.predict import predict_diabetes_risk

class MLPredictorAgent(BaseAgent):
    name = "ml_predictor"

    async def run(self, input: AgentInput) -> AgentOutput:
        features = input.context.get("vitals", {})

        if not features:
            return AgentOutput(
                response="Please provide your health vitals (glucose, BMI, blood pressure, age) for a risk assessment.",
                agent_name=self.name,
                escalate=False
            )

        result = predict_diabetes_risk(features)

        response = f"""
Risk Assessment Result:
- Risk level: {result['risk_level']}
- Risk score: {round(result['risk_score'] * 100, 1)}%
- {result['explanation']}

{result['disclaimer']}
"""
        return AgentOutput(
            response=response.strip(),
            agent_name=self.name,
            escalate=result["risk_level"] == "HIGH",
            confidence=result["risk_score"],
            metadata=result
        )