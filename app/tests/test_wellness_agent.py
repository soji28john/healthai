# tests/test_wellness_agent.py
import pytest
from unittest.mock import patch
from app.agents.wellness_agent import WellnessAgent
from app.agents.base_agent import AgentInput

agent = WellnessAgent()

class TestSeverityGate:
    @pytest.mark.asyncio
    async def test_high_severity_blocked_from_remedies(self):
        high_input = AgentInput(
            user_id="u1",
            message="chest tightness",
            context={"severity": "HIGH"}
        )
        result = await agent.run(high_input)
        # must redirect to doctor, not give home remedies
        assert result.escalate is True
        assert "healthcare professional" in result.response.lower()

    @pytest.mark.asyncio
    async def test_emergency_severity_blocked(self):
        emergency_input = AgentInput(
            user_id="u1",
            message="unconscious briefly",
            context={"severity": "EMERGENCY"}
        )
        result = await agent.run(emergency_input)
        assert result.escalate is True

    @pytest.mark.asyncio
    async def test_low_severity_gets_recommendations(self):
        low_input = AgentInput(
            user_id="u1",
            message="mild headache",
            context={"severity": "LOW"}
        )
        mock_response = "Try drinking water, rest in a quiet room. Gentle neck stretches may help."
        with patch("app.agents.wellness_agent.call_llm", return_value=mock_response):
            result = await agent.run(low_input)
            assert result.escalate is False
            assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_response_contains_disclaimer(self):
        low_input = AgentInput(
            user_id="u1",
            message="mild cold",
            context={"severity": "LOW"}
        )
        mock_response = "Steam inhalation may help. These are general wellness suggestions, not medical advice."
        with patch("app.agents.wellness_agent.call_llm", return_value=mock_response):
            result = await agent.run(low_input)
            assert "not medical advice" in result.response.lower()