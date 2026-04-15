# tests/test_rag_pipeline.py
import pytest
from unittest.mock import patch, MagicMock


class TestRAGPipelineRetrieval:

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_string(self):
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Iron deficiency is common in vegetarians."

        with patch("app.rag.pipeline.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_node]
            mock_index.as_retriever.return_value = mock_retriever
            mock_get_index.return_value = mock_index

            from app.rag.pipeline import retrieve_context
            result = await retrieve_context("iron deficiency symptoms")

            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_retrieve_returns_top_k_results(self):
        mock_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(mock_nodes):
            node.get_content.return_value = f"Medical fact number {i + 1}"

        with patch("app.rag.pipeline.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = mock_nodes
            mock_index.as_retriever.return_value = mock_retriever
            mock_get_index.return_value = mock_index

            from app.rag.pipeline import retrieve_context
            result = await retrieve_context("diabetes symptoms", top_k=3)

            # all 3 node contents should be joined in the result
            assert "Medical fact number 1" in result
            assert "Medical fact number 2" in result
            assert "Medical fact number 3" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_joins_with_double_newline(self):
        mock_nodes = [MagicMock(), MagicMock()]
        mock_nodes[0].get_content.return_value = "Fact A"
        mock_nodes[1].get_content.return_value = "Fact B"

        with patch("app.rag.pipeline.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = mock_nodes
            mock_index.as_retriever.return_value = mock_retriever
            mock_get_index.return_value = mock_index

            from app.rag.pipeline import retrieve_context
            result = await retrieve_context("test query")

            assert "Fact A\n\nFact B" == result

    @pytest.mark.asyncio
    async def test_retrieve_context_handles_empty_results(self):
        with patch("app.rag.pipeline.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_index.as_retriever.return_value = mock_retriever
            mock_get_index.return_value = mock_index

            from app.rag.pipeline import retrieve_context
            result = await retrieve_context("unknown query")

            assert result == ""

    def test_get_index_is_singleton(self):
        # index should only be built once and cached
        with patch("app.rag.pipeline.chromadb") as mock_chroma:
            with patch("app.rag.pipeline.VectorStoreIndex") as mock_vsi:
                mock_chroma.PersistentClient.return_value \
                    .get_or_create_collection.return_value = MagicMock()

                from app.rag import pipeline
                pipeline._index = None  # reset singleton

                pipeline.get_index()
                pipeline.get_index()  # call twice

                # VectorStoreIndex should only be constructed once
                assert mock_vsi.from_vector_store.call_count == 1

    @pytest.mark.asyncio
    async def test_retriever_called_with_correct_top_k(self):
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Some content"

        with patch("app.rag.pipeline.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_node]
            mock_index.as_retriever.return_value = mock_retriever
            mock_get_index.return_value = mock_index

            from app.rag.pipeline import retrieve_context
            await retrieve_context("headache remedies", top_k=5)

            # confirm retriever was configured with the right top_k
            mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)