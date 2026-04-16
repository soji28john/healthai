from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from app.agents.symptom_agent import SymptomAgent
from app.agents.wellness_agent import WellnessAgent
from app.agents.nutrition_agent import NutritionAgent

class HealthState(TypedDict):
    user_id: str
    message: str
    severity: Optional[str]
    symptom_response: Optional[str]
    wellness_response: Optional[str]
    nutrition_response: Optional[str]
    final_response: str
    escalate: bool

symptom_agent = SymptomAgent()
wellness_agent = WellnessAgent()
nutrition_agent = NutritionAgent()

async def triage_node(state: HealthState) -> HealthState:
    from app.agents.base_agent import AgentInput
    
    message = state["message"].lower()
    
    try:
        
        result = await symptom_agent.run(AgentInput(user_id=state["user_id"], message=state["message"]))
        severity = result.metadata.get("severity", "LOW")
        escalate = result.escalate
        response = result.response
    except Exception:
        
        if any(keyword in message for keyword in ["chest pain", "shortness of breath", "severe headache", "sudden weakness", "unconscious"]):
            severity = "HIGH"
            escalate = True
            response = "Your symptoms may indicate a serious condition. Please seek immediate medical attention."
        else:
            severity = "LOW"
            escalate = False
            response = "Based on your symptoms, it seems like a mild condition. Here are some general wellness suggestions."
        
    return {**state, "severity": severity, "symptom_response": response, "escalate": escalate}

async def wellness_node(state: HealthState) -> HealthState:
    from app.agents.base_agent import AgentInput
    
    try:
        
        result = await wellness_agent.run(AgentInput(
            user_id=state["user_id"],
            message=state["message"],
            context={"severity": state.get("severity", "LOW")}
        ))
        response = result.response
    except Exception:
        response = "Maintain hydration, rest, and monitor symptoms."
    return {**state, "wellness_response": response}

def should_escalate(state: HealthState) -> str:
    return "escalate" if state.get("escalate") else "wellness"

def build_graph():
    graph = StateGraph(HealthState)
    graph.add_node("triage", triage_node)
    graph.add_node("wellness", wellness_node)
    graph.add_node("escalate", lambda s: {**s, "final_response": "Please seek immediate medical attention."})
    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", should_escalate, {"escalate": "escalate", "wellness": "wellness"})
    graph.add_edge("wellness", END)
    graph.add_edge("escalate", END)
    return graph.compile()

health_graph = build_graph()