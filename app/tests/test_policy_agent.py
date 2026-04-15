# tests/test_policy_agent.py
import pytest
import re
from app.gateway.policy_agent import PolicyMiddleware, BLOCK_PATTERNS, ESCALATE_PATTERNS

middleware = PolicyMiddleware(None)

class TestBlockPatterns:
    def test_suicide_message_is_blocked(self):
        text = "I want to kill myself"
        matched = any(re.search(p, text, re.IGNORECASE) for p in BLOCK_PATTERNS)
        assert matched is True

    def test_self_harm_is_blocked(self):
        text = "I am thinking about self harm"
        matched = any(re.search(p, text, re.IGNORECASE) for p in BLOCK_PATTERNS)
        assert matched is True

    def test_normal_health_query_passes(self):
        text = "I have a headache for two days"
        matched = any(re.search(p, text, re.IGNORECASE) for p in BLOCK_PATTERNS)
        assert matched is False

    def test_nutrition_query_passes(self):
        text = "What foods are rich in iron?"
        matched = any(re.search(p, text, re.IGNORECASE) for p in BLOCK_PATTERNS)
        assert matched is False

class TestEscalatePatterns:
    def test_chest_pain_triggers_escalation(self):
        text = "I have severe chest pain"
        matched = any(re.search(p, text, re.IGNORECASE) for p in ESCALATE_PATTERNS)
        assert matched is True

    def test_cant_breathe_triggers_escalation(self):
        text = "I can't breathe properly"
        matched = any(re.search(p, text, re.IGNORECASE) for p in ESCALATE_PATTERNS)
        assert matched is True

    def test_mild_symptom_does_not_escalate(self):
        text = "I have a mild cold"
        matched = any(re.search(p, text, re.IGNORECASE) for p in ESCALATE_PATTERNS)
        assert matched is False

class TestCrisisResponse:
    def test_crisis_response_contains_resources(self):
        response = middleware._crisis_response()
        import json
        body = json.loads(response.body)
        assert "resources" in body
        assert body["policy_action"] == "BLOCKED"
        assert len(body["resources"]) > 0