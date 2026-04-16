from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, RiskRequest, RiskResponse
from app.agents.orchestrator import health_graph
from app.agents.base_agent import AgentInput
from app.ml.predict import predict_diabetes_risk

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    # check if policy middleware already blocked this
    if hasattr(request.state, "blocked") and request.state.blocked:
        raise HTTPException(status_code=400, detail="Request blocked by policy agent")

    initial_state = {
        "user_id": body.user_id,
        "message": body.message,
        "severity": None,
        "symptom_response": None,
        "wellness_response": None,
        "nutrition_response": None,
        "final_response": "",
        "escalate": getattr(request.state, "escalate", False)
    }

    result = await health_graph.ainvoke(initial_state)

    # build final response from whichever agent responded
    response_text = (
        result.get("wellness_response")
        or result.get("symptom_response")
        or result.get("nutrition_response")
        or result.get("final_response")
        or "I was unable to process your request. Please try again."
    )

    return ChatResponse(
        response=response_text,
        agent_name="orchestrator",
        severity=result.get("severity"),
        escalate=result.get("escalate", False),
        metadata={"full_state": {k: v for k, v in result.items() if k != "message"}}
    )

@router.post("/predict/diabetes", response_model=RiskResponse)
async def predict_diabetes(body: RiskRequest):
    result = predict_diabetes_risk(body.features)
    return RiskResponse(**result)

@router.get("/agents")
async def list_agents():
    return {
        "agents": [
            "symptom_triage",
            "wellness",
            "nutrition",
            "mental_health",
            "ml_predictor",
            "doctor_recommender"
        ]
    }