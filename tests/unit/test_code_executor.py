"""
Unit tests for code executor module.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from code_executor import (
    ResourceLimits,
    ExecutionResult,
    DownloadableFile,
    CodeDetector,
    ResourceMonitor,
    NetworkIsolator,
    EnhancedSandboxedCodeExecutor,
    format_code_output,
    execute_code,
)
from datetime import datetime


class TestResourceLimits:
    """Test cases for ResourceLimits dataclass."""

    def test_initializes_with_default_values(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create ResourceLimits instance
        Assert: Verify default values are set correctly
        """
        # Arrange - None needed

        # Act
        limits = ResourceLimits()

        # Assert
        assert limits.max_cpu_time == 30
        assert limits.max_memory_mb == 512
        assert limits.max_file_size_mb == 100
        assert limits.max_processes == 1
        assert limits.max_open_files == 64
        assert limits.allow_network is False
        assert limits.max_disk_write_mb == 500

    def test_initializes_with_custom_values(self):
        """
        AAA Test:
        Arrange: Define custom limit values
        Act: Create ResourceLimits with custom values
        Assert: Verify custom values are set correctly
        """
        # Arrange
        custom_limits = {
            'max_cpu_time': 60,
            'max_memory_mb': 1024,
            'max_file_size_mb': 200,
            'max_processes': 2,
            'max_open_files': 128,
            'allow_network': True,
            'max_disk_write_mb': 1000
        }

        # Act
        limits = ResourceLimits(**custom_limits)

        # Assert
        assert limits.max_cpu_time == 60
        assert limits.max_memory_mb == 1024
        assert limits.max_file_size_mb == 200
        assert limits.max_processes == 2
        assert limits.max_open_files == 128
        assert limits.allow_network is True
        assert limits.max_disk_write_mb == 1000


class TestExecutionResult:
    """Test cases for ExecutionResult dataclass."""

    def test_initializes_success_result(self):
        """
        AAA Test:
        Arrange: Define success result values
        Act: Create ExecutionResult for successful execution
        Assert: Verify result properties
        """
        # Arrange
        result_data = {
            'success': True,
            'output': 'Test output',
            'error': '',
            'files_created': [{'name': 'test.txt'}],
            'execution_time': 1.5,
            'stdout': 'Test output',
            'stderr': '',
            'resources_used': {'peak_memory_mb': 100},
            'exit_code': 0
        }

        # Act
        result = ExecutionResult(**result_data)

        # Assert
        assert result.success is True
        assert result.output == 'Test output'
        assert result.error == ''
        assert len(result.files_created) == 1
        assert result.execution_time == 1.5
        assert result.exit_code == 0

    def test_initializes_failure_result(self):
        """
        AAA Test:
        Arrange: Define failure result values
        Act: Create ExecutionResult for failed execution
        Assert: Verify error properties
        """
        # Arrange
        error_data = {
            'success': False,
            'output': '',
            'error': 'Syntax error',
            'exit_code': 1
        }

        # Act
        result = ExecutionResult(**error_data)

        # Assert
        assert result.success is False
        assert result.output == ''
        assert result.error == 'Syntax error'
        assert result.exit_code == 1


class TestDownloadableFile:
    """Test cases for DownloadableFile dataclass."""

    def test_initializes_with_all_fields(self):
        """
        AAA Test:
        Arrange: Define file metadata
        Act: Create DownloadableFile instance
        Assert: Verify all fields are set correctly
        """
        # Arrange
        file_info = {
            'filename': 'test.pdf',
            'filepath': '/tmp/test.pdf',
            'size': 1024,
            'mime_type': 'application/pdf',
            'created_at': datetime(2025, 1, 1, 12, 0, 0),
            'file_hash': 'abc123def456'
        }

        # Act
        downloadable = DownloadableFile(**file_info)

        # Assert
        assert downloadable.filename == 'test.pdf'
        assert downloadable.filepath == '/tmp/test.pdf'
        assert downloadable.size == 1024
        assert downloadable.mime_type == 'application/pdf'
        assert downloadable.file_hash == 'abc123def456'


class TestCodeDetector:
    """Test cases for CodeDetector class."""

    def test_detect_code_request_with_create_pdf(self):
        """
        AAA Test:
        Arrange: Create a message asking to create PDF
        Act: Call detect_code_request
        Assert: Verify code request is detected
        """
        # Arrange
        message = "Can you create a PDF report for me?"

        # Act
        detected, pattern = CodeDetector.detect_code_request(message)

        # Assert
        assert detected is True
        assert "pdf" in pattern.lower()

    def test_detect_code_request_with_plot(self):
        """
        AAA Test:
        Arrange: Create a message asking to plot data
        Act: Call detect_code_request
        Assert: Verify code request is detected
        """
        # Arrange
        message = "Please plot this data"

        # Act
        detected, pattern = CodeDetector.detect_code_request(message)

        # Assert
        assert detected is True
        assert "plot" in pattern.lower()

    def test_detect_code_request_with_analyze(self):
        """
        AAA Test:
        Arrange: Create a message asking to analyze data
        Act: Call detect_code_request
        Assert: Verify code request is detected
        """
        # Arrange
        # The pattern expects "analyze data" or "analyze csv" directly
        message = "Analyze csv data"

        # Act
        detected, pattern = CodeDetector.detect_code_request(message)

        # Assert
        assert detected is True
        assert "analyze" in pattern.lower()

    def test_detect_code_request_negative_case(self):
        """
        AAA Test:
        Arrange: Create a normal conversation message
        Act: Call detect_code_request
        Assert: Verify no code request is detected
        """
        # Arrange
        message = "Hello, how are you today?"

        # Act
        detected, pattern = CodeDetector.detect_code_request(message)

        # Assert
        assert detected is False
        assert pattern == ""

    def test_detect_code_in_response_with_python_markdown(self):
        """
        AAA Test:
        Arrange: Create a response with Python code block
        Act: Call detect_code_in_response
        Assert: Verify Python code is detected
        """
        # Arrange
        response = "Here's the code:\n```python\nprint('Hello')\n```"

        # Act
        has_code, language, code = CodeDetector.detect_code_in_response(response)

        # Assert
        assert has_code is True
        assert language == 'python'
        assert "print('Hello')" in code

    def test_detect_code_in_response_with_javascript(self):
        """
        AAA Test:
        Arrange: Create a response with JavaScript code block
        Act: Call detect_code_in_response
        Assert: Verify JavaScript code is detected
        """
        # Arrange
        response = "```javascript\nconsole.log('test');\n```"

        # Act
        has_code, language, code = CodeDetector.detect_code_in_response(response)

        # Assert
        assert has_code is True
        assert language == 'javascript'
        assert "console.log" in code

    def test_detect_code_in_response_negative_case(self):
        """
        AAA Test:
        Arrange: Create a response without code
        Act: Call detect_code_in_response
        Assert: Verify no code is detected
        """
        # Arrange
        response = "This is just plain text with no code blocks."

        # Act
        has_code, language, code = CodeDetector.detect_code_in_response(response)

        # Assert
        assert has_code is False
        assert language == ''
        assert code == ''

    def test_detect_code_with_inline_python(self):
        """
        AAA Test:
        Arrange: Create a response with inline Python code
        Act: Call detect_code_in_response
        Assert: Verify Python code is detected
        """
        # Arrange
        response = "You can use import numpy as np to work with arrays."

        # Act
        has_code, language, code = CodeDetector.detect_code_in_response(response)

        # Assert
        assert has_code is True
        assert language == 'python'
        assert "import numpy" in code


class TestNetworkIsolator:
    """Test cases for NetworkIsolator class."""

    def test_create_unshare_script_returns_string(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Call create_unshare_script
        Assert: Verify script is returned as string
        """
        # Arrange - None needed

        # Act
        script = NetworkIsolator.create_unshare_script()

        # Assert
        assert isinstance(script, str)
        assert "unshare" in script
        assert "#!/bin/bash" in script

    def test_is_available_on_linux(self):
        """
        AAA Test:
        Arrange: Mock Linux platform and existing unshare command
        Act: Call is_available
        Assert: Verify returns True on Linux with unshare
        """
        # Arrange
        with patch('sys.platform', 'linux'), \
             patch('os.path.exists', return_value=True):

            # Act
            available = NetworkIsolator.is_available()

            # Assert
            assert available is True

    def test_is_available_on_macos(self):
        """
        AAA Test:
        Arrange: Mock macOS platform
        Act: Call is_available
        Assert: Verify returns False on macOS
        """
        # Arrange
        with patch('sys.platform', 'darwin'):

            # Act
            available = NetworkIsolator.is_available()

            # Assert
            assert available is False


class TestEnhancedSandboxedCodeExecutorInit:
    """Test cases for EnhancedSandboxedCodeExecutor initialization."""

    def test_initializes_with_default_limits(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create executor without arguments
        Assert: Verify default limits are used
        """
        # Arrange - None needed

        # Act
        executor = EnhancedSandboxedCodeExecutor()

        # Assert
        assert executor.limits.max_cpu_time == 30
        assert executor.limits.max_memory_mb == 512
        assert executor.temp_dir is None
        assert executor._monitor is None
        assert len(executor._downloadable_files) == 0

    def test_initializes_with_custom_limits(self):
        """
        AAA Test:
        Arrange: Create custom resource limits
        Act: Create executor with custom limits
        Assert: Verify custom limits are used
        """
        # Arrange
        custom_limits = ResourceLimits(max_cpu_time=10, max_memory_mb=256)

        # Act
        executor = EnhancedSandboxedCodeExecutor(custom_limits)

        # Assert
        assert executor.limits.max_cpu_time == 10
        assert executor.limits.max_memory_mb == 256

    def test_blocked_modules_is_defined(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access BLOCKED_MODULES
        Assert: Verify dangerous modules are blocked
        """
        # Arrange - None needed

        # Act
        blocked = EnhancedSandboxedCodeExecutor.BLOCKED_MODULES

        # Assert
        assert 'os' in blocked
        assert 'sys' in blocked
        assert 'subprocess' in blocked
        assert 'eval' in blocked
        assert 'exec' in blocked

    def test_allowed_modules_is_defined(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Access ALLOWED_MODULES
        Assert: Verify safe modules are allowed
        """
        # Arrange - None needed

        # Act
        allowed = EnhancedSandboxedCodeExecutor.ALLOWED_MODULES

        # Assert
        assert 'json' in allowed
        assert 'math' in allowed
        assert 'datetime' in allowed
        assert 'pandas' in allowed
        assert 'numpy' in allowed


class TestEnhancedSandboxedCodeExecutorValidation:
    """Test cases for code validation in executor."""

    def test_validate_code_with_blocked_module_os(self):
        """
        AAA Test:
        Arrange: Create executor and code with os import
        Act: Call _validate_code
        Assert: Verify os module is blocked
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "import os\nos.system('ls')"

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is not None
        assert "os" in error.lower()
        assert "not allowed" in error.lower()

    def test_validate_code_with_blocked_module_subprocess(self):
        """
        AAA Test:
        Arrange: Create executor and code with subprocess import
        Act: Call _validate_code
        Assert: Verify subprocess is blocked
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "import subprocess"

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is not None
        assert "subprocess" in error.lower()

    def test_validate_code_with_eval(self):
        """
        AAA Test:
        Arrange: Create executor and code with eval
        Act: Call _validate_code
        Assert: Verify eval is blocked
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "x = eval('1 + 1')"

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is not None
        assert "eval" in error.lower()

    def test_validate_code_with_exec(self):
        """
        AAA Test:
        Arrange: Create executor and code with exec
        Act: Call _validate_code
        Assert: Verify exec is blocked
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "exec('print(\"test\")')"

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is not None
        assert "exec" in error.lower()

    def test_validate_code_with_safe_imports(self):
        """
        AAA Test:
        Arrange: Create executor and code with safe imports
        Act: Call _validate_code
        Assert: Verify code passes validation
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "import json\nimport math\nfrom datetime import datetime"

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is None

    def test_validate_code_with_python_file_open(self):
        """
        AAA Test:
        Arrange: Create executor and code opening .py file
        Act: Call _validate_code
        Assert: Verify .py file opening is blocked
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = 'with open("script.py", "r") as f: pass'

        # Act
        error = executor._validate_code(code)

        # Assert
        assert error is not None
        assert ".py" in error


class TestEnhancedSandboxedCodeExecutorFileOperations:
    """Test cases for file operations in executor."""

    def test_get_downloadable_files_empty_initially(self):
        """
        AAA Test:
        Arrange: Create executor
        Act: Call get_downloadable_files
        Assert: Verify empty list is returned
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()

        # Act
        files = executor.get_downloadable_files()

        # Assert
        assert files == []

    def test_save_file_to_returns_false_when_no_files(self):
        """
        AAA Test:
        Arrange: Create executor with no downloadable files
        Act: Call save_file_to
        Assert: Verify returns False
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()

        # Act
        result = executor.save_file_to("test.pdf", "/tmp/test.pdf")

        # Assert
        assert result is False

    def test_get_file_content_returns_none_when_no_files(self):
        """
        AAA Test:
        Arrange: Create executor with no downloadable files
        Act: Call get_file_content
        Assert: Verify returns None
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()

        # Act
        content = executor.get_file_content("test.txt")

        # Assert
        assert content is None

    def test_cleanup_temp_files_when_none(self):
        """
        AAA Test:
        Arrange: Create executor without running
        Act: Call cleanup_temp_files
        Assert: Verify no errors occur
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()

        # Act & Assert - should not raise
        executor.cleanup_temp_files()
        assert executor.temp_dir is None


class TestEnhancedSandboxedCodeExecutorExecutionLog:
    """Test cases for execution logging."""

    def test_get_execution_log_empty_initially(self):
        """
        AAA Test:
        Arrange: Create executor
        Act: Call get_execution_log
        Assert: Verify empty list is returned
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()

        # Act
        log = executor.get_execution_log()

        # Assert
        assert log == []

    def test_indent_code_with_spaces(self):
        """
        AAA Test:
        Arrange: Create multi-line code and executor
        Act: Call _indent_code
        Assert: Verify code is indented correctly
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "line1\nline2\nline3"

        # Act
        indented = executor._indent_code(code, 4)

        # Assert
        assert indented == "    line1\n    line2\n    line3"

    def test_indent_code_preserves_empty_lines(self):
        """
        AAA Test:
        Arrange: Create code with empty lines
        Act: Call _indent_code
        Assert: Verify empty lines are preserved
        """
        # Arrange
        executor = EnhancedSandboxedCodeExecutor()
        code = "line1\n\nline3"

        # Act
        indented = executor._indent_code(code, 2)

        # Assert
        assert indented == "  line1\n\n  line3"


class TestFormatCodeOutput:
    """Test cases for format_code_output function."""

    def test_format_successful_execution(self):
        """
        AAA Test:
        Arrange: Create successful execution result
        Act: Call format_code_output
        Assert: Verify success message is formatted
        """
        # Arrange
        result = ExecutionResult(
            success=True,
            output="Execution completed",
            stdout="Output here",
            execution_time=1.5
        )

        # Act
        formatted = format_code_output(result)

        # Assert
        assert "✅" in formatted or "successful" in formatted.lower()
        assert "Output here" in formatted
        assert "1.5" in formatted or "1" in formatted

    def test_format_failed_execution(self):
        """
        AAA Test:
        Arrange: Create failed execution result
        Act: Call format_code_output
        Assert: Verify error message is formatted
        """
        # Arrange
        result = ExecutionResult(
            success=False,
            output="",
            error="SyntaxError: invalid syntax",
            execution_time=0.1
        )

        # Act
        formatted = format_code_output(result)

        # Assert
        assert "❌" in formatted or "failed" in formatted.lower()
        assert "SyntaxError" in formatted

    def test_format_with_files_created(self):
        """
        AAA Test:
        Arrange: Create result with created files
        Act: Call format_code_output
        Assert: Verify files are listed
        """
        # Arrange
        result = ExecutionResult(
            success=True,
            output="Done",
            files_created=[
                {'filename': 'report.pdf', 'size': 1024, 'mime_type': 'application/pdf', 'hash': 'abc123'}
            ]
        )

        # Act
        formatted = format_code_output(result)

        # Assert
        assert "report.pdf" in formatted
        assert "1024" in formatted or "bytes" in formatted

    def test_format_with_resources_used(self):
        """
        AAA Test:
        Arrange: Create result with resource usage
        Act: Call format_code_output
        Assert: Verify resources are displayed
        """
        # Arrange
        result = ExecutionResult(
            success=True,
            output="Done",
            resources_used={
                'peak_memory_mb': 128.5,
                'cpu_time': 2.3,
                'disk_write_mb': 1.2
            }
        )

        # Act
        formatted = format_code_output(result)

        # Assert
        assert "128" in formatted or "memory" in formatted.lower()
        assert "2.3" in formatted or "cpu" in formatted.lower()


class TestExecuteCodeConvenienceFunction:
    """Test cases for execute_code convenience function."""

    def test_execute_code_returns_execution_result(self):
        """
        AAA Test:
        Arrange: Mock the executor
        Act: Call execute_code
        Assert: Verify ExecutionResult is returned
        """
        # Arrange
        mock_result = ExecutionResult(success=True, output="test")
        with patch('code_executor.EnhancedSandboxedCodeExecutor') as MockExecutor:
            mock_instance = Mock()
            mock_instance.execute.return_value = mock_result
            MockExecutor.return_value = mock_instance

            # Act
            result = execute_code("print('test')")

            # Assert
            assert isinstance(result, ExecutionResult)
            assert result.success is True

    def test_execute_code_passes_resource_limits(self):
        """
        AAA Test:
        Arrange: Create custom resource limits
        Act: Call execute_code with limits
        Assert: Verify limits are passed to executor
        """
        # Arrange
        custom_limits = ResourceLimits(max_cpu_time=10)
        with patch('code_executor.EnhancedSandboxedCodeExecutor') as MockExecutor:
            mock_instance = Mock()
            mock_instance.execute.return_value = ExecutionResult(success=True, output="")
            MockExecutor.return_value = mock_instance

            # Act
            execute_code("print('test')", custom_limits)

            # Assert
            MockExecutor.assert_called_once_with(custom_limits)


class TestMimeTypes:
    """Test cases for MIME type mapping."""

    def test_mime_type_for_pdf(self):
        """
        AAA Test:
        Arrange: Access MIME_TYPES
        Act: Check .pdf extension
        Assert: Verify correct MIME type
        """
        # Arrange & Act
        mime_type = EnhancedSandboxedCodeExecutor.MIME_TYPES.get('.pdf')

        # Assert
        assert mime_type == 'application/pdf'

    def test_mime_type_for_docx(self):
        """
        AAA Test:
        Arrange: Access MIME_TYPES
        Act: Check .docx extension
        Assert: Verify correct MIME type
        """
        # Arrange & Act
        mime_type = EnhancedSandboxedCodeExecutor.MIME_TYPES.get('.docx')

        # Assert
        assert mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

    def test_mime_type_for_png(self):
        """
        AAA Test:
        Arrange: Access MIME_TYPES
        Act: Check .png extension
        Assert: Verify correct MIME type
        """
        # Arrange & Act
        mime_type = EnhancedSandboxedCodeExecutor.MIME_TYPES.get('.png')

        # Assert
        assert mime_type == 'image/png'
