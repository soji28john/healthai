# tests/test_symptom_agent.py
import pytest
from unittest.mock import patch
from app.agents.symptom_agent import SymptomAgent, validate_symptom_output
from app.agents.base_agent import AgentInput
import json

agent = SymptomAgent()

class TestSymptomAgentOutput:
    @pytest.mark.asyncio
    async def test_returns_severity_field(self, sample_input):
        mock_response = json.dumps({
            "severity": "LOW",
            "explanation": "Mild symptoms, likely viral",
            "recommendation": "Rest and hydration",
            "confidence": 0.85
        })
        with patch("app.agents.symptom_agent.call_llm", return_value=mock_response):
            result = await agent.run(sample_input)
            assert result.metadata.get("severity") == "LOW"

    @pytest.mark.asyncio
    async def test_emergency_sets_escalate_true(self, emergency_input):
        mock_response = json.dumps({
            "severity": "EMERGENCY",
            "explanation": "Possible cardiac event",
            "recommendation": "Call emergency services immediately",
            "confidence": 0.95
        })
        with patch("app.agents.symptom_agent.call_llm", return_value=mock_response):
            result = await agent.run(emergency_input)
            assert result.escalate is True

    @pytest.mark.asyncio
    async def test_low_severity_does_not_escalate(self, low_severity_input):
        mock_response = json.dumps({
            "severity": "LOW",
            "explanation": "Common cold symptoms",
            "recommendation": "Rest and fluids",
            "confidence": 0.9
        })
        with patch("app.agents.symptom_agent.call_llm", return_value=mock_response):
            result = await agent.run(low_severity_input)
            assert result.escalate is False

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_response(self, sample_input):
        with patch("app.agents.symptom_agent.call_llm", return_value="not valid json"):
            result = await agent.run(sample_input)
            # should not crash — falls back to safe defaults
            assert result.response is not None
            assert result.agent_name == "symptom_triage"

class TestHardLimits:
    def test_diagnosis_language_is_flagged(self):
        assert validate_symptom_output("you have appendicitis") is False

    def test_medication_dosage_is_flagged(self):
        assert validate_symptom_output("take 500mg of paracetamol") is False

    def test_drug_name_is_flagged(self):
        assert validate_symptom_output("try ibuprofen for the pain") is False

    def test_clean_response_passes(self):
        assert validate_symptom_output(
            "Your symptoms suggest a mild viral infection. Rest and stay hydrated."
        ) is True

    def test_confidence_field_present(self, sample_input):
        mock_response = json.dumps({
            "severity": "MODERATE",
            "explanation": "Symptoms may indicate infection",
            "recommendation": "Monitor and see a doctor if worsens",
            "confidence": 0.75
        })
        with patch("app.agents.symptom_agent.call_llm", return_value=mock_response):
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                agent.run(sample_input)
            )
            assert result.confidence > 0