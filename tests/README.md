# Test Suite for Local LLM UI

This directory contains the test suite for the Local LLM UI application, organized into unit tests and functional tests.

## Directory Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared pytest fixtures
├── pytest.ini                  # Pytest configuration
├── README.md                   # This file
├── unit/                       # Unit tests (AAA pattern)
│   ├── __init__.py
│   ├── test_utils.py          # Tests for utility functions
│   ├── test_config.py         # Tests for configuration
│   ├── test_document_processor.py  # Tests for DocumentProcessor
│   └── test_vector_store.py   # Tests for VectorStore
└── functional/                # Functional tests
    ├── __init__.py
    └── test_rag_workflow.py   # End-to-end RAG workflow tests
```

## Test Types

### Unit Tests (`tests/unit/`)
Individual component tests following the **AAA (Arrange-Act-Assert)** pattern:
- **Arrange**: Set up the test environment and preconditions
- **Act**: Execute the function/method being tested
- **Assert**: Verify the expected outcomes

Each test focuses on a single function or method in isolation.

### Functional Tests (`tests/functional/`)
End-to-end workflow tests that verify:
- Document processing workflows
- Vector store operations
- RAG engine integration
- Complete user scenarios

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest tests/unit/
```

### Run Only Functional Tests
```bash
pytest tests/functional/
```

### Run Specific Test File
```bash
pytest tests/unit/test_utils.py
```

### Run Specific Test Class
```bash
pytest tests/unit/test_utils.py::TestGetResourcePath
```

### Run Specific Test
```bash
pytest tests/unit/test_utils.py::TestGetResourcePath::test_returns_relative_path_when_not_bundled
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Verbose Output
```bash
pytest -v
```

### Run with Markers
```bash
pytest -m unit          # Only unit tests
pytest -m functional    # Only functional tests
pytest -m "not slow"    # Skip slow tests
```

## Test Organization

### Unit Tests (`test_*.py` in `unit/`)
Each test file corresponds to a source module:
- `test_utils.py` → `src/utils.py`
- `test_config.py` → `src/config.py`
- `test_document_processor.py` → `main.py:DocumentProcessor`
- `test_vector_store.py` → `main.py:VectorStore`
- `test_rag.py` → `src/rag.py` (RAG search, embedding model, document parser)
- `test_rag_context_limit.py` → `src/rag.py` (Context size limit testing)

### Functional Tests (`test_*.py` in `functional/`)
Organized by workflow:
- `test_rag_workflow.py` → Complete RAG workflows

## Writing New Tests

### Unit Test Template (AAA Pattern)

```python
class TestMyFunction:
    """Test cases for my_function."""

    def test_descriptive_name(self):
        """
        AAA Test:
        Arrange: Set up test data and preconditions
        Act: Call the function being tested
        Assert: Verify expected results
        """
        # Arrange
        input_data = "test input"
        expected = "expected output"

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected
```

### Functional Test Template

```python
class TestMyWorkflow:
    """Functional tests for my workflow."""

    def test_end_to_end_scenario(self):
        """
        AAA Test:
        Arrange: Set up complete test environment
        Act: Execute the workflow
        Assert: Verify expected end-to-end behavior
        """
        # Arrange
        engine = setup_engine()
        test_data = create_test_data()

        # Act
        result = execute_workflow(engine, test_data)

        # Assert
        assert result.is_complete()
        assert result.has_expected_properties()
```

## Fixtures

Shared fixtures in `conftest.py`:
- `temp_dir`: Temporary directory for test files
- `sample_text_file`: Sample .txt file
- `sample_csv_file`: Sample .csv file
- `sample_md_file`: Sample .md file
- `mock_embedding_model`: Mock embedding model
- `sample_document_chunks`: Sample text chunks

## Best Practices

1. **AAA Pattern**: All unit tests follow Arrange-Act-Assert
2. **Descriptive Names**: Test names clearly describe what is being tested
3. **Isolation**: Each test should be independent
4. **Cleanup**: Use `finally` blocks or fixtures for cleanup
5. **Mocking**: Mock external dependencies (file system, network, models)
6. **Documentation**: Add docstrings explaining test purpose
7. **One Assert Per Test**: Prefer multiple focused tests over one large test

## Dependencies

Install test dependencies:
```bash
pip install pytest pytest-cov numpy
```

## CI/CD Integration

Tests are configured to run in CI/CD via GitHub Actions (`.github/workflows/build.yml`).

## Notes

- Tests use mocking to avoid requiring actual ML models
- File operations use temporary directories
- Tests are designed to run quickly and in parallel when possible
- Configuration tests verify values without requiring running application

## Conversation Isolation Tests

The RAG module includes conversation isolation via the `allowed_sources` parameter:

### Behavior
- `allowed_sources=None`: Search all documents (default, backward compatible)
- `allowed_sources=[]`: Search no documents (new conversation with no docs)
- `allowed_sources=["file1.txt"]`: Search only in specified documents

### Test Coverage
- `TestRAGConversationIsolation` class in `tests/unit/test_rag.py`
- Tests filtering by single/multiple sources
- Tests empty list returns no results
- Tests None searches all documents

This ensures that documents from one conversation don't leak into other conversations.
