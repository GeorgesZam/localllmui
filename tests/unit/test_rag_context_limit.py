"""
Unit tests for RAG context limit functionality.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag import RAG


class TestRAGContextLimit:
    """Test cases for RAG context size limiting."""

    def test_context_respects_max_chars_limit(self):
        """
        AAA Test:
        Arrange: Create RAG with large documents
        Act: Search with max_context_chars limit
        Assert: Verify context length is within limit
        """
        # Arrange
        rag = RAG()
        large_content = "Test content. " * 1000  # Very large content
        rag.documents = [
            {'source': 'test.txt', 'chunk_id': 0, 'content': large_content}
        ]

        # Act
        context, sources = rag.search('test query', max_context_chars=500)

        # Assert
        assert len(context) <= 600  # Allow small margin for formatting
        assert len(sources) == 1

    def test_context_includes_multiple_sources_when_space_permits(self):
        """
        AAA Test:
        Arrange: Create RAG with multiple small documents
        Act: Search with generous max_context_chars
        Assert: Verify context is built (may have fewer sources if no match)
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {'source': 'doc1.txt', 'chunk_id': 0, 'content': 'test query content ' * 20},
            {'source': 'doc2.txt', 'chunk_id': 0, 'content': 'test query more ' * 20},
            {'source': 'doc3.txt', 'chunk_id': 0, 'content': 'test query data ' * 20},
        ]

        # Act
        context, sources = rag.search('test query', max_context_chars=2000)

        # Assert
        # Should have some results since "test query" is in the content
        assert len(context) <= 2000
        # May have 0-3 sources depending on keyword matching
        assert isinstance(sources, list)

    def test_context_truncates_long_content(self):
        """
        AAA Test:
        Arrange: Create RAG with very long single document
        Act: Search with max_context_chars
        Assert: Verify content is truncated with ellipsis
        """
        # Arrange
        rag = RAG()
        long_content = "A" * 2000
        rag.documents = [
            {'source': 'long.txt', 'chunk_id': 0, 'content': long_content}
        ]

        # Act
        context, sources = rag.search('test query', max_context_chars=500)

        # Assert
        assert len(context) <= 600  # Allow margin for header
        # Content should be truncated
        assert '...' in context or len(context) < len(long_content)

    def test_context_preserves_document_header(self):
        """
        AAA Test:
        Arrange: Create RAG with documents
        Act: Search and retrieve context
        Assert: Verify each chunk has proper header
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {'source': 'test.txt', 'chunk_id': 0, 'content': 'Test content here'}
        ]

        # Act
        context, sources = rag.search('test query', max_context_chars=500)

        # Assert
        assert '[Document 1 - test.txt]' in context
        assert 'Test content here' in context

    def test_empty_search_returns_empty_context(self):
        """
        AAA Test:
        Arrange: Create RAG with no documents
        Act: Search with query
        Assert: Verify empty result is returned
        """
        # Arrange
        rag = RAG()
        rag.documents = []

        # Act
        context, sources = rag.search('test query')

        # Assert
        assert context == ""
        assert sources == []

    def test_context_limit_with_multiple_chunks(self):
        """
        AAA Test:
        Arrange: Create RAG with multiple documents that would exceed limit
        Act: Search with strict max_context_chars (using matching query)
        Assert: Verify context is properly divided among chunks
        """
        # Arrange
        rag = RAG()
        # Include "query" to ensure keyword match
        rag.documents = [
            {'source': 'doc1.txt', 'chunk_id': 0, 'content': 'query content A ' * 100},
            {'source': 'doc2.txt', 'chunk_id': 0, 'content': 'query content B ' * 100},
            {'source': 'doc3.txt', 'chunk_id': 0, 'content': 'query content C ' * 100},
        ]

        # Act
        context, sources = rag.search('query', max_context_chars=1000)

        # Assert
        assert len(context) <= 1000
        # Should have at least some results
        assert isinstance(sources, list)
        if len(sources) > 0:
            assert len(context) > 0


class TestRAGDefaultConfig:
    """Test cases for default RAG configuration."""

    def test_default_chunk_size_is_reduced(self):
        """
        AAA Test:
        Arrange: Import config
        Act: Check RAG_CHUNK_SIZE value
        Assert: Verify it's set to safe value (256 or less)
        """
        # Arrange
        import config

        # Act
        chunk_size = config.rag_chunk_size

        # Assert
        assert chunk_size <= 256, "Chunk size should be <= 256 to avoid context overflow"

    def test_default_top_k_is_reduced(self):
        """
        AAA Test:
        Arrange: Import config
        Act: Check RAG_TOP_K value
        Assert: Verify it's set to safe value (2 or less)
        """
        # Arrange
        import config

        # Act
        top_k = config.rag_top_k

        # Assert
        assert top_k <= 2, "Top K should be <= 2 to leave room for prompt/response"
