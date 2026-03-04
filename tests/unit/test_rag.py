"""
Unit tests for RAG module classes.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag import RAG, EmbeddingModel, DocumentParser


class TestEmbeddingModel:
    """Test cases for EmbeddingModel class."""

    def test_initializes_with_unloaded_state(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create EmbeddingModel instance
        Assert: Verify initial unloaded state
        """
        # Arrange - None needed

        # Act
        model = EmbeddingModel()

        # Assert
        assert model.is_loaded is False
        assert model.model is None

    def test_encode_returns_empty_when_model_not_loaded(self):
        """
        AAA Test:
        Arrange: Create EmbeddingModel without loading
        Act: Call encode
        Assert: Verify empty array is returned
        """
        # Arrange
        model = EmbeddingModel()

        # Act
        result = model.encode(["test"])

        # Assert
        assert result.size == 0


class TestDocumentParser:
    """Test cases for DocumentParser class."""

    def test_supported_extensions(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access SUPPORTED_EXTENSIONS
        Assert: Verify common extensions are included
        """
        # Arrange - None needed

        # Act
        extensions = DocumentParser.SUPPORTED_EXTENSIONS

        # Assert
        assert '.txt' in extensions
        assert '.pdf' in extensions
        assert '.docx' in extensions

    def test_text_extensions(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access TEXT_EXTENSIONS
        Assert: Verify text extensions are correct
        """
        # Arrange - None needed

        # Act
        extensions = DocumentParser.TEXT_EXTENSIONS

        # Assert
        assert '.txt' in extensions
        assert '.md' in extensions
        assert '.py' in extensions
        assert '.json' in extensions

    def test_parse_text_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a test text file
        Act: Parse the file
        Assert: Verify content is extracted
        """
        # Arrange
        test_content = "Test content for parsing"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        # Mock OCR processor
        mock_ocr = Mock()

        # Act
        parser = DocumentParser(mock_ocr)
        result = parser.parse(str(test_file))

        # Assert
        assert result == test_content

    def test_parse_markdown_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a markdown file
        Act: Parse the file
        Assert: Verify content is extracted
        """
        # Arrange
        test_content = "# Test\n\nSome content"
        test_file = tmp_path / "test.md"
        test_file.write_text(test_content)

        mock_ocr = Mock()

        # Act
        parser = DocumentParser(mock_ocr)
        result = parser.parse(str(test_file))

        # Assert
        assert "# Test" in result
        assert "Some content" in result

    def test_parse_json_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a JSON file
        Act: Parse the file
        Assert: Verify content is extracted
        """
        # Arrange
        test_content = '{"key": "value"}'
        test_file = tmp_path / "test.json"
        test_file.write_text(test_content)

        mock_ocr = Mock()

        # Act
        parser = DocumentParser(mock_ocr)
        result = parser.parse(str(test_file))

        # Assert
        assert '"key": "value"' in result


class TestRAGInit:
    """Test cases for RAG initialization."""

    def test_initializes_with_empty_state(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create RAG instance
        Assert: Verify initial empty state
        """
        # Arrange - None needed

        # Act
        rag = RAG()

        # Assert
        assert rag.documents == []
        assert rag.last_sources == []
        assert rag._embeddings is None
        assert rag.embedding_model is not None
        assert rag.parser is not None

    def test_has_embedding_model(self):
        """
        AAA Test:
        Arrange: Create RAG instance
        Act: Access embedding_model property
        Assert: Verify EmbeddingModel instance exists
        """
        # Arrange
        rag = RAG()

        # Act
        model = rag.embedding_model

        # Assert
        assert isinstance(model, EmbeddingModel)

    def test_has_document_parser(self):
        """
        AAA Test:
        Arrange: Create RAG instance
        Act: Access parser property
        Assert: Verify DocumentParser instance exists
        """
        # Arrange
        rag = RAG()

        # Act
        parser = rag.parser

        # Assert
        assert isinstance(parser, DocumentParser)


class TestRAGSearch:
    """Test cases for RAG search functionality."""

    def test_search_returns_empty_when_no_documents(self):
        """
        AAA Test:
        Arrange: Create RAG with no documents
        Act: Call search
        Assert: Verify empty results
        """
        # Arrange
        rag = RAG()
        rag.documents = []

        # Act
        context, sources = rag.search("test query")

        # Assert
        assert context == ""
        assert sources == []

    def test_search_with_documents(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with test documents
        Act: Call search
        Assert: Verify results are returned
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "test.txt", "chunk_id": 0, "content": "Test content about cats"},
            {"source": "test.txt", "chunk_id": 1, "content": "Test content about dogs"}
        ]

        # Mock embeddings to be None for keyword search
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act
        context, sources = rag.search("cats")

        # Assert
        assert len(context) > 0
        assert len(sources) > 0

    def test_search_updates_last_sources(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with documents
        Act: Call search
        Assert: Verify last_sources is updated
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "test.txt", "chunk_id": 0, "content": "Python programming"}
        ]
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act
        rag.search("Python")

        # Assert
        assert len(rag.last_sources) > 0
        assert "source" in rag.last_sources[0]


class TestRAGClearCache:
    """Test cases for RAG cache clearing."""

    def test_clear_cache_resets_documents(self, tmp_path):
        """
        AAA Test:
        Arrange: Create RAG with cache files
        Act: Clear cache
        Assert: Verify documents and embeddings are reset
        """
        # Arrange
        rag = RAG()
        rag.documents = [{"source": "test", "chunk_id": 0, "content": "test"}]

        # Create mock cache files
        index_file = tmp_path / "index.json"
        index_file.write_text("[]")

        # Mock the cache file paths to use temp directory
        import rag as rag_module
        original_index_file = rag_module.get_writable_path
        monkeypatch = pytest.MonkeyPatch()

        def mock_writable_path(filename):
            if filename == "index.json":
                return str(index_file)
            return str(tmp_path / filename)

        monkeypatch.setattr(rag_module, 'get_writable_path', mock_writable_path)

        try:
            # Act
            rag.clear_cache()

            # Assert
            assert rag.documents == []
            assert rag._embeddings is None
        finally:
            # Cleanup
            monkeypatch.undo()


class TestRAGFormatSources:
    """Test cases for formatting sources for display."""

    def test_format_empty_sources(self):
        """
        AAA Test:
        Arrange: Create RAG with no sources
        Act: Format sources
        Assert: Verify empty string is returned
        """
        # Arrange
        rag = RAG()
        rag.last_sources = []

        # Act
        result = rag.format_sources_for_display()

        # Assert
        assert result == ""

    def test_format_sources_with_data(self):
        """
        AAA Test:
        Arrange: Create RAG with sources
        Act: Format sources
        Assert: Verify formatted output
        """
        # Arrange
        rag = RAG()
        rag.last_sources = [
            {
                "index": 1,
                "source": "test.txt",
                "chunk_id": 0,
                "score": 0.85,
                "preview": "This is a preview of the content"
            }
        ]

        # Act
        result = rag.format_sources_for_display()

        # Assert
        assert "📚 Sources:" in result
        assert "test.txt" in result
        assert "0.85" in result


class TestRAGConversationIsolation:
    """Test cases for conversation isolation via allowed_sources parameter."""

    def test_search_with_allowed_sources_filters_results(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with multiple documents from different sources
        Act: Call search with allowed_sources filtering to one file
        Assert: Verify only results from allowed file are returned
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "conversation1.txt", "chunk_id": 0, "content": "Content about cats"},
            {"source": "conversation1.txt", "chunk_id": 1, "content": "Content about dogs"},
            {"source": "conversation2.txt", "chunk_id": 0, "content": "Content about birds"},
        ]
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act - search only in conversation1.txt
        context, sources = rag.search("cats", allowed_sources=["conversation1.txt"])

        # Assert - only sources from conversation1.txt should be returned
        assert len(sources) > 0
        for source in sources:
            assert source["source"] == "conversation1.txt"

    def test_search_with_empty_allowed_sources_returns_nothing(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with documents
        Act: Call search with empty allowed_sources list
        Assert: Verify no results are returned (new conversation with no docs)
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "conversation1.txt", "chunk_id": 0, "content": "Content about cats"},
        ]
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act - search with empty list (no documents in current conversation)
        context, sources = rag.search("cats", allowed_sources=[])

        # Assert
        assert context == ""
        assert sources == []

    def test_search_with_none_allowed_sources_searches_all(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with multiple documents
        Act: Call search with allowed_sources=None
        Assert: Verify results from all documents are returned
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "file1.txt", "chunk_id": 0, "content": "Python programming"},
            {"source": "file2.txt", "chunk_id": 0, "content": "JavaScript programming"},
        ]
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act - search with None (search all documents)
        context, sources = rag.search("programming", allowed_sources=None)

        # Assert - results from both files should be returned
        assert len(sources) > 0

    def test_search_with_multiple_allowed_sources(self, monkeypatch):
        """
        AAA Test:
        Arrange: Create RAG with documents from multiple conversations
        Act: Call search with multiple allowed sources
        Assert: Verify results from all allowed sources are returned
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "doc1.txt", "chunk_id": 0, "content": "Machine learning content"},
            {"source": "doc2.txt", "chunk_id": 0, "content": "Deep learning content"},
            {"source": "doc3.txt", "chunk_id": 0, "content": "Neural networks content"},
        ]
        monkeypatch.setattr(rag, '_embeddings', None)

        # Act - search allowing only doc1.txt and doc2.txt
        context, sources = rag.search("learning", allowed_sources=["doc1.txt", "doc2.txt"])

        # Assert
        assert len(sources) > 0
        for source in sources:
            assert source["source"] in ["doc1.txt", "doc2.txt"]
