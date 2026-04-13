from pydantic import BaseModel
from typing import Any
from app.llm.router import call_llm

class AgentInput(BaseModel):
    user_id: str
    message: str
    context: dict = {}

class AgentOutput(BaseModel):
    response: str
    confidence: float = 1.0
    escalate: bool = False
    agent_name: str
    metadata: dict = {}

class BaseAgent:
    name: str = "base"
    system_prompt: str = "You are a helpful health assistant."

    async def run(self, input: AgentInput) -> AgentOutput:
        response = await call_llm(
            prompt=self._build_prompt(input),
            system=self.system_prompt
        )
        return AgentOutput(
            response=response,
            agent_name=self.name
        )

    def _build_prompt(self, input: AgentInput) -> str:
        return input.message