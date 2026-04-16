from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: dict = {}

class ChatResponse(BaseModel):
    response: str
    agent_name: str
    severity: Optional[str] = None
    escalate: bool = False
    confidence: float = 1.0
    metadata: dict = {}
    disclaimer: str = "This is for informational purposes only. Not a substitute for professional medical advice."

class RiskRequest(BaseModel):
    user_id: str
    features: dict

class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    explanation: str
    disclaimer: str

class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int
    gender: str
    conditions: list[str] = []
    medications: list[str] = []