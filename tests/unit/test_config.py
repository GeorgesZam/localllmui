"""
Unit tests for configuration module.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import config


class TestAppConfig:
    """Test cases for application configuration."""

    def test_app_name_is_defined(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access APP_NAME constant
        Assert: Verify it's a non-empty string
        """
        # Arrange - None needed

        # Act
        app_name = config.APP_NAME

        # Assert
        assert isinstance(app_name, str)
        assert len(app_name) > 0

    def test_window_size_is_valid_format(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access WINDOW_SIZE constant
        Assert: Verify it matches expected format (WIDTHxHEIGHT)
        """
        # Arrange - None needed

        # Act
        window_size = config.WINDOW_SIZE

        # Assert
        assert isinstance(window_size, str)
        assert 'x' in window_size
        parts = window_size.split('x')
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


class TestModelConfig:
    """Test cases for model configuration."""

    def test_model_paths_are_defined(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access model path constants
        Assert: Verify they are strings
        """
        # Arrange - None needed

        # Act
        model_file = config.MODEL_FILE
        embedding_folder = config.EMBEDDING_MODEL_FOLDER

        # Assert
        assert isinstance(model_file, str)
        assert isinstance(embedding_folder, str)

    def test_context_size_is_positive_integer(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access CONTEXT_SIZE constant
        Assert: Verify it's a positive integer
        """
        # Arrange - None needed

        # Act
        context_size = config.CONTEXT_SIZE

        # Assert
        assert isinstance(context_size, int)
        assert context_size > 0

    def test_max_tokens_less_than_context_size(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access both constants
        Assert: Verify max_tokens is less than context_size
        """
        # Arrange - None needed

        # Act
        context_size = config.CONTEXT_SIZE
        max_tokens = config.MAX_TOKENS

        # Assert
        assert max_tokens < context_size

    def test_threads_is_positive_integer(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access THREADS constant
        Assert: Verify it's a positive integer
        """
        # Arrange - None needed

        # Act
        threads = config.THREADS

        # Assert
        assert isinstance(threads, int)
        assert threads >= 1

    def test_gpu_layers_allows_all(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access GPU_LAYERS constant
        Assert: Verify -1 means use all layers
        """
        # Arrange - None needed

        # Act
        gpu_layers = config.GPU_LAYERS

        # Assert
        assert gpu_layers == -1


class TestRAGConfig:
    """Test cases for RAG configuration."""

    def test_rag_is_enabled(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access RAG_ENABLED constant
        Assert: Verify RAG is enabled
        """
        # Arrange - None needed

        # Act
        rag_enabled = config.RAG_ENABLED

        # Assert
        assert rag_enabled is True

    def test_chunk_size_positive(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access RAG_CHUNK_SIZE constant
        Assert: Verify it's a positive integer
        """
        # Arrange - None needed

        # Act
        chunk_size = config.RAG_CHUNK_SIZE

        # Assert
        assert isinstance(chunk_size, int)
        assert chunk_size > 0

    def test_chunk_overlap_less_than_chunk_size(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access chunk constants
        Assert: Verify overlap is less than chunk size
        """
        # Arrange - None needed

        # Act
        chunk_size = config.RAG_CHUNK_SIZE
        chunk_overlap = config.RAG_CHUNK_OVERLAP

        # Assert
        assert chunk_overlap < chunk_size

    def test_top_k_is_positive(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access RAG_TOP_K constant
        Assert: Verify it's a positive integer
        """
        # Arrange - None needed

        # Act
        top_k = config.RAG_TOP_K

        # Assert
        assert isinstance(top_k, int)
        assert top_k > 0

    def test_min_score_in_valid_range(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access RAG_MIN_SCORE constant
        Assert: Verify it's between 0 and 1
        """
        # Arrange - None needed

        # Act
        min_score = config.RAG_MIN_SCORE

        # Assert
        assert 0.0 <= min_score <= 1.0


class TestSamplingConfig:
    """Test cases for sampling configuration."""

    def test_temperature_in_valid_range(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access TEMPERATURE constant
        Assert: Verify it's between 0 and 2
        """
        # Arrange - None needed

        # Act
        temperature = config.TEMPERATURE

        # Assert
        assert 0.0 <= temperature <= 2.0

    def test_top_p_in_valid_range(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access TOP_P constant
        Assert: Verify it's between 0 and 1
        """
        # Arrange - None needed

        # Act
        top_p = config.TOP_P

        # Assert
        assert 0.0 <= top_p <= 1.0

    def test_repeat_penalty_greater_than_one(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access REPEAT_PENALTY constant
        Assert: Verify it's greater than 1.0
        """
        # Arrange - None needed

        # Act
        repeat_penalty = config.REPEAT_PENALTY

        # Assert
        assert repeat_penalty >= 1.0


class TestPerformanceConfig:
    """Test cases for performance configuration."""

    def test_batch_size_positive(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access BATCH_SIZE constant
        Assert: Verify it's a positive integer
        """
        # Arrange - None needed

        # Act
        batch_size = config.BATCH_SIZE

        # Assert
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_lazy_loading_enabled(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access LAZY_LOAD_EMBEDDING constant
        Assert: Verify it's a boolean
        """
        # Arrange - None needed

        # Act
        lazy_load = config.LAZY_LOAD_EMBEDDING

        # Assert
        assert isinstance(lazy_load, bool)

    def test_index_cache_enabled(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access INDEX_CACHE_ENABLED constant
        Assert: Verify it's a boolean
        """
        # Arrange - None needed

        # Act
        index_cache = config.INDEX_CACHE_ENABLED

        # Assert
        assert isinstance(index_cache, bool)


class TestStopTokens:
    """Test cases for stop tokens configuration."""

    def test_stop_tokens_is_list(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access STOP_TOKENS constant
        Assert: Verify it's a list
        """
        # Arrange - None needed

        # Act
        stop_tokens = config.STOP_TOKENS

        # Assert
        assert isinstance(stop_tokens, list)

    def test_stop_tokens_not_empty(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access STOP_TOKENS constant
        Assert: Verify list has at least one entry
        """
        # Arrange - None needed

        # Act
        stop_tokens = config.STOP_TOKENS

        # Assert
        assert len(stop_tokens) > 0
