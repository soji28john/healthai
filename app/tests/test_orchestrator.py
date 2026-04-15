# tests/test_orchestrator.py
import pytest
from unittest.mock import patch, AsyncMock
from app.agents.orchestrator import build_graph, HealthState

class TestOrchestratorRouting:
    @pytest.mark.asyncio
    async def test_emergency_routes_to_escalate_node(self):
        graph = build_graph()
        initial_state = HealthState(
            user_id="u1",
            message="chest pain radiating to arm",
            severity=None,
            symptom_response=None,
            wellness_response=None,
            nutrition_response=None,
            final_response="",
            escalate=False
        )
        mock_triage = AsyncMock(return_value={
            **initial_state,
            "severity": "EMERGENCY",
            "escalate": True,
            "symptom_response": "This is an emergency."
        })
        with patch("app.agents.orchestrator.triage_node", mock_triage):
            result = await graph.ainvoke(initial_state)
            assert result["escalate"] is True

    @pytest.mark.asyncio
    async def test_low_severity_routes_to_wellness(self):
        graph = build_graph()
        initial_state = HealthState(
            user_id="u1",
            message="mild headache",
            severity=None,
            symptom_response=None,
            wellness_response=None,
            nutrition_response=None,
            final_response="",
            escalate=False
        )
        mock_triage = AsyncMock(return_value={
            **initial_state,
            "severity": "LOW",
            "escalate": False,
            "symptom_response": "Mild symptoms detected."
        })
        mock_wellness = AsyncMock(return_value={
            **initial_state,
            "severity": "LOW",
            "escalate": False,
            "wellness_response": "Try resting and drinking water."
        })
        with patch("app.agents.orchestrator.triage_node", mock_triage):
            with patch("app.agents.orchestrator.wellness_node", mock_wellness):
                result = await graph.ainvoke(initial_state)
                assert result["wellness_response"] is not None