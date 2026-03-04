"""
Functional tests for RAG workflow.
These tests verify end-to-end workflows and integrations.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag import RAG, EmbeddingModel, DocumentParser


class TestDocumentProcessingWorkflow:
    """Functional tests for document processing workflow."""

    def test_end_to_end_txt_processing(self):
        """
        AAA Test:
        Arrange: Create a test text file
        Act: Process the file through DocumentParser
        Assert: Verify text is extracted correctly
        """
        # Arrange
        test_content = """
        This is a test document.
        It contains multiple lines.
        And some important information.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file = f.name

        try:
            # Act
            mock_ocr = Mock()
            parser = DocumentParser(mock_ocr)
            extracted = parser.parse(test_file)

            # Assert
            assert "test document" in extracted
            assert "multiple lines" in extracted
            assert "important information" in extracted
        finally:
            # Cleanup
            os.unlink(test_file)

    def test_end_to_end_json_processing(self):
        """
        AAA Test:
        Arrange: Create a test JSON file
        Act: Process the file through DocumentParser
        Assert: Verify JSON data is extracted
        """
        # Arrange
        test_content = '{"name": "Test", "value": 123}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(test_content)
            test_file = f.name

        try:
            # Act
            mock_ocr = Mock()
            parser = DocumentParser(mock_ocr)
            extracted = parser.parse(test_file)

            # Assert
            assert '"name": "Test"' in extracted
            assert '"value": 123' in extracted or '"value":123' in extracted
        finally:
            # Cleanup
            os.unlink(test_file)

    def test_multiple_file_types_processing(self):
        """
        AAA Test:
        Arrange: Create multiple test files of different types
        Act: Process all files
        Assert: Verify all are processed successfully
        """
        # Arrange
        files_created = []
        try:
            # Create test files
            txt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            txt_file.write("Text file content")
            txt_file.close()
            files_created.append(txt_file.name)

            md_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
            md_file.write("# Markdown content")
            md_file.close()
            files_created.append(md_file.name)

            # Act
            mock_ocr = Mock()
            parser = DocumentParser(mock_ocr)
            results = []
            for file_path in files_created:
                extracted = parser.parse(file_path)
                results.append(extracted)

            # Assert
            assert len(results) == 2
            assert "Text file content" in results[0]
            assert "Markdown content" in results[1]
        finally:
            # Cleanup
            for f in files_created:
                os.unlink(f)


class TestRAGWorkflow:
    """Functional tests for RAG workflow."""

    def test_initialization_workflow(self):
        """
        AAA Test:
        Arrange: Create RAG instance
        Act: Initialize RAG
        Assert: Verify RAG is ready
        """
        # Arrange
        status_messages = []
        def status_callback(msg):
            status_messages.append(msg)

        # Act
        rag = RAG()

        # Assert
        assert rag is not None
        assert rag.embedding_model is not None
        assert rag.parser is not None

    def test_search_without_documents(self):
        """
        AAA Test:
        Arrange: Create RAG with no documents
        Act: Try to search
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

    def test_search_with_mock_documents(self):
        """
        AAA Test:
        Arrange: Create RAG with mock documents
        Act: Search for content
        Assert: Verify search returns results
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "test.txt", "chunk_id": 0, "content": "Python is a programming language"},
            {"source": "test.txt", "chunk_id": 1, "content": "JavaScript is used for web development"}
        ]

        # Act
        context, sources = rag.search("Python")

        # Assert
        assert len(context) > 0
        assert len(sources) > 0
        assert "Python" in context

    def test_format_sources_workflow(self):
        """
        AAA Test:
        Arrange: Create RAG with search results
        Act: Format sources for display
        Assert: Verify formatted output
        """
        # Arrange
        rag = RAG()
        rag.last_sources = [
            {
                "index": 1,
                "source": "document.txt",
                "chunk_id": 0,
                "score": 0.92,
                "preview": "This is a preview of the document content"
            }
        ]

        # Act
        formatted = rag.format_sources_for_display()

        # Assert
        assert "📚 Sources:" in formatted
        assert "document.txt" in formatted
        assert "0.92" in formatted


class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""

    def test_document_to_search_workflow(self):
        """
        AAA Test:
        Arrange: Create test document and RAG instance
        Act: Add document and search it
        Assert: Verify end-to-end workflow
        """
        # Arrange
        rag = RAG()

        # Create test document
        test_content = """
        Python is a programming language created by Guido van Rossum.
        It is widely used for web development, data science, and automation.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file = f.name

        try:
            # Act - Simulate having documents in RAG
            rag.documents = [
                {"source": "test.txt", "chunk_id": 0, "content": test_content.strip()}
            ]

            context, sources = rag.search("Guido")

            # Assert
            assert len(context) > 0
            assert len(sources) > 0
        finally:
            # Cleanup
            os.unlink(test_file)

    def test_multiple_searches_workflow(self):
        """
        AAA Test:
        Arrange: Create RAG with multiple documents
        Act: Perform multiple searches
        Assert: Verify each search works
        """
        # Arrange
        rag = RAG()
        rag.documents = [
            {"source": "doc1.txt", "chunk_id": 0, "content": "Content about cats and dogs"},
            {"source": "doc2.txt", "chunk_id": 0, "content": "Content about birds and fish"},
            {"source": "doc3.txt", "chunk_id": 0, "content": "Content about reptiles"}
        ]

        # Act
        results_cats = rag.search("cats")
        results_birds = rag.search("birds")
        results_all = rag.search("animals")

        # Assert
        assert len(results_cats[1]) > 0 or len(results_cats[0]) > 0
        assert len(results_birds[1]) > 0 or len(results_birds[0]) > 0


class TestEmbeddingModelWorkflow:
    """Functional tests for embedding model workflow."""

    def test_model_initial_state(self):
        """
        AAA Test:
        Arrange: Create EmbeddingModel
        Act: Check initial state
        Assert: Verify model is unloaded
        """
        # Arrange - None needed

        # Act
        model = EmbeddingModel()

        # Assert
        assert model.is_loaded is False
        assert model.model is None

    def test_encode_without_loaded_model(self):
        """
        AAA Test:
        Arrange: Create EmbeddingModel without loading
        Act: Try to encode text
        Assert: Verify empty array is returned
        """
        # Arrange
        model = EmbeddingModel()

        # Act
        result = model.encode(["test text"])

        # Assert
        assert result.size == 0


class TestErrorHandlingWorkflow:
    """Functional tests for error handling in workflows."""

    def test_non_existent_file_handling(self):
        """
        AAA Test:
        Arrange: Create parser
        Act: Try to parse non-existent file
        Assert: Verify graceful handling
        """
        # Arrange
        mock_ocr = Mock()
        parser = DocumentParser(mock_ocr)
        non_existent_file = "/tmp/non_existent_file_12345.txt"

        # Act
        result = parser.parse(non_existent_file)

        # Assert
        # Should return empty string or handle gracefully
        assert result == "" or result is None

    def test_unsupported_file_type(self):
        """
        AAA Test:
        Arrange: Create parser and unsupported file
        Act: Try to parse unsupported file type
        Assert: Verify graceful handling
        """
        # Arrange
        mock_ocr = Mock()
        parser = DocumentParser(mock_ocr)

        with tempfile.NamedTemporaryFile(suffix='.xyz123', delete=False) as f:
            f.write(b"some content")
            test_file = f.name

        try:
            # Act
            # Should fallback to text parsing
            result = parser.parse(test_file)

            # Assert
            # Should handle gracefully (either parse as text or return empty)
            assert isinstance(result, str)
        finally:
            # Cleanup
            os.unlink(test_file)

    def test_empty_file_handling(self):
        """
        AAA Test:
        Arrange: Create empty file
        Act: Parse the file
        Assert: Verify empty result is returned
        """
        # Arrange
        mock_ocr = Mock()
        parser = DocumentParser(mock_ocr)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            test_file = f.name

        try:
            # Act
            result = parser.parse(test_file)

            # Assert
            assert result == ""
        finally:
            # Cleanup
            os.unlink(test_file)
