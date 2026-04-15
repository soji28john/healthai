# tests/conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.agents.base_agent import AgentInput

@pytest.fixture
def sample_input():
    return AgentInput(
        user_id="test_user_001",
        message="I have a headache and mild fever for two days",
        context={}
    )

@pytest.fixture
def emergency_input():
    return AgentInput(
        user_id="test_user_002",
        message="severe chest pain radiating to my left arm, can't breathe",
        context={}
    )

@pytest.fixture
def low_severity_input():
    return AgentInput(
        user_id="test_user_003",
        message="I have a mild runny nose since this morning",
        context={}
    )

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client