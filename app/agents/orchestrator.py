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
    result = await symptom_agent.run(AgentInput(user_id=state["user_id"], message=state["message"]))
    return {**state, "severity": result.metadata.get("severity", "LOW"), "symptom_response": result.response, "escalate": result.escalate}

async def wellness_node(state: HealthState) -> HealthState:
    from app.agents.base_agent import AgentInput
    result = await wellness_agent.run(AgentInput(
        user_id=state["user_id"],
        message=state["message"],
        context={"severity": state.get("severity", "LOW")}
    ))
    return {**state, "wellness_response": result.response}

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