# tests/test_api_endpoints.py
import pytest

class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_returns_200(self, async_client):
        response = await async_client.post("/api/v1/chat", json={
            "user_id": "test_user",
            "message": "I have a mild headache"
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_crisis_message_returns_blocked(self, async_client):
        response = await async_client.post("/api/v1/chat", json={
            "user_id": "test_user",
            "message": "I want to kill myself"
        })
        data = response.json()
        assert data["policy_action"] == "BLOCKED"
        assert "resources" in data

    @pytest.mark.asyncio
    async def test_empty_message_returns_422(self, async_client):
        response = await async_client.post("/api/v1/chat", json={
            "user_id": "test_user",
            "message": ""
        })
        assert response.status_code in (422, 400)

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, async_client):
        response = await async_client.post("/api/v1/chat", json={
            "user_id": "test_user",
            "message": "What foods are good for iron deficiency?"
        })
        data = response.json()
        assert "response" in data
        assert "agent_name" in data