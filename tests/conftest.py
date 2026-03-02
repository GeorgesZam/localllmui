"""
Shared fixtures and configuration for pytest.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text file for testing.\nIt has multiple lines.")
    return str(file_path)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    file_path = temp_dir / "sample.csv"
    file_path.write_text("Name,Age,City\nAlice,30,NYC\nBob,25,LA")
    return str(file_path)


@pytest.fixture
def sample_md_file(temp_dir):
    """Create a sample markdown file for testing."""
    content = """# Test Document

## Section 1

This is a test paragraph.

## Section 2

- Item 1
- Item 2
"""
    file_path = temp_dir / "sample.md"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    import numpy as np
    from unittest.mock import Mock

    mock = Mock()
    mock.encode = Mock(return_value=np.array([[0.1] * 384] * 2))
    return mock


@pytest.fixture
def sample_document_chunks():
    """Provide sample document chunks for testing."""
    return [
        "This is the first chunk of text.",
        "This is the second chunk with different content.",
        "The third chunk contains more information."
    ]
