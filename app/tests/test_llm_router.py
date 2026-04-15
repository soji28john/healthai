# tests/test_llm_router.py
import pytest
from unittest.mock import patch, AsyncMock
from app.llm.router import call_llm

class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_gemini_success_returns_response(self):
        with patch("app.llm.router.genai") as mock_genai:
            mock_model = mock_genai.GenerativeModel.return_value
            mock_model.generate_content.return_value.text = "Gemini response"
            result = await call_llm("test prompt")
            assert result == "Gemini response"

    @pytest.mark.asyncio
    async def test_falls_back_to_groq_when_gemini_fails(self):
        with patch("app.llm.router.genai") as mock_gemini:
            mock_gemini.GenerativeModel.side_effect = Exception("Gemini down")
            with patch("app.llm.router.Groq") as mock_groq:
                mock_client = mock_groq.return_value
                mock_client.chat.completions.create.return_value \
                    .choices[0].message.content = "Groq response"
                result = await call_llm("test prompt")
                assert result == "Groq response"

    @pytest.mark.asyncio
    async def test_raises_when_all_backends_fail(self):
        with patch("app.llm.router.genai") as mock_gemini:
            mock_gemini.GenerativeModel.side_effect = Exception("Gemini down")
            with patch("app.llm.router.Groq") as mock_groq:
                mock_groq.return_value.chat.completions.create \
                    .side_effect = Exception("Groq down")
                with pytest.raises(RuntimeError, match="All LLM backends failed"):
                    await call_llm("test prompt")