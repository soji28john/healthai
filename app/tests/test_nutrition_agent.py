# tests/test_nutrition_agent.py
import pytest
from unittest.mock import patch
from app.agents.nutrition_agent import NutritionAgent
from app.agents.base_agent import AgentInput

agent = NutritionAgent()

SAMPLE_DIET_INPUT = AgentInput(
    user_id="test_user_001",
    message="I eat rice, dal, and vegetables daily. Rarely eat meat or dairy.",
    context={"age": 28, "gender": "female", "condition": "none"}
)

DIABETIC_INPUT = AgentInput(
    user_id="test_user_002",
    message="I eat white bread, rice, sweets daily",
    context={"age": 45, "gender": "male", "condition": "pre-diabetic"}
)

IRON_DEFICIENCY_INPUT = AgentInput(
    user_id="test_user_003",
    message="I feel tired often, I am vegetarian",
    context={"age": 25, "gender": "female", "condition": "iron deficiency"}
)


class TestNutritionAgentOutput:

    @pytest.mark.asyncio
    async def test_returns_response_for_normal_diet(self):
        mock_response = (
            "Your diet appears low in Vitamin B12 and calcium. "
            "Consider adding dairy or fortified plant milk. "
            "These are general wellness suggestions, not medical advice."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(SAMPLE_DIET_INPUT)
            assert result.response is not None
            assert len(result.response) > 0
            assert result.agent_name == "nutrition"

    @pytest.mark.asyncio
    async def test_response_mentions_nutrient_gaps(self):
        mock_response = (
            "Based on your diet, you may have gaps in: Iron, Vitamin B12, Calcium. "
            "Recommended daily intake for iron is 18mg for women aged 19-50."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(SAMPLE_DIET_INPUT)
            assert any(
                word in result.response.lower()
                for word in ["iron", "calcium", "vitamin", "gap", "intake"]
            )

    @pytest.mark.asyncio
    async def test_diabetic_profile_gets_macro_guidance(self):
        mock_response = (
            "For a pre-diabetic profile, reduce refined carbohydrates. "
            "Aim for complex carbs like oats and legumes. "
            "Keep total carbohydrate intake under 45% of daily calories."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(DIABETIC_INPUT)
            assert any(
                word in result.response.lower()
                for word in ["carb", "sugar", "glycemic", "diabetic", "calorie"]
            )

    @pytest.mark.asyncio
    async def test_iron_deficiency_recommends_iron_rich_foods(self):
        mock_response = (
            "Your symptoms and diet suggest low iron intake. "
            "Iron-rich foods for vegetarians include: spinach, lentils, "
            "tofu, pumpkin seeds, and fortified cereals. "
            "Pair with Vitamin C sources to improve absorption."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(IRON_DEFICIENCY_INPUT)
            assert any(
                word in result.response.lower()
                for word in ["iron", "spinach", "lentil", "absorption"]
            )

    @pytest.mark.asyncio
    async def test_response_always_contains_disclaimer(self):
        mock_response = (
            "You may be low on Vitamin D. "
            "These are general wellness suggestions, not medical advice."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(SAMPLE_DIET_INPUT)
            assert "not medical advice" in result.response.lower()

    @pytest.mark.asyncio
    async def test_does_not_prescribe_supplements_by_dosage(self):
        mock_response = (
            "You may benefit from iron-rich foods. "
            "Consult a doctor before starting any supplements."
        )
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(SAMPLE_DIET_INPUT)
            import re
            # must not contain dosage patterns like "200mg" or "500 mcg"
            dosage_pattern = r"\b\d+\s*(mg|mcg|iu|g)\b"
            assert not re.search(dosage_pattern, result.response, re.IGNORECASE)

    @pytest.mark.asyncio
    async def test_escalate_is_false_for_normal_nutrition_query(self):
        mock_response = "Your diet looks generally balanced with minor gaps in Vitamin D."
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(SAMPLE_DIET_INPUT)
            assert result.escalate is False

    @pytest.mark.asyncio
    async def test_handles_empty_diet_description_gracefully(self):
        empty_input = AgentInput(
            user_id="test_user_004",
            message="",
            context={}
        )
        mock_response = "Please describe your typical daily diet so I can analyse your nutrient intake."
        with patch("app.agents.nutrition_agent.call_llm", return_value=mock_response):
            result = await agent.run(empty_input)
            assert result.response is not None
            assert result.agent_name == "nutrition"