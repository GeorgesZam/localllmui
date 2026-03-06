"""
Functional tests for code execution workflow.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from code_executor import (
    CodeDetector,
    EnhancedSandboxedCodeExecutor,
    execute_code,
    ResourceLimits,
)


class TestCodeDetectionWorkflow:
    """Functional tests for code request detection workflow."""

    def test_full_workflow_detect_and_execute(self):
        """
        AAA Test:
        Arrange: User message requesting data analysis
        Act: Detect code request and validate workflow
        Assert: Verify code execution request is identified
        """
        # Arrange
        user_messages = [
            "Can you create a PDF report from this data?",
            "Please plot the sales figures",
            "Analyze csv data and show statistics",
            "Can you help me create charts",
        ]

        # Act
        for message in user_messages:
            detected, pattern = CodeDetector.detect_code_request(message)

            # Assert
            assert detected is True, f"Failed for: {message}"
            assert len(pattern) > 0

    def test_workflow_with_normal_conversation(self):
        """
        AAA Test:
        Arrange: Normal conversation messages
        Act: Check for code requests
        Assert: Verify no false positives
        """
        # Arrange
        normal_messages = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "What's the capital of France?",
            "How do I cook pasta?",
        ]

        # Act
        for message in normal_messages:
            detected, pattern = CodeDetector.detect_code_request(message)

            # Assert
            assert detected is False, f"False positive for: {message}"
            assert pattern == ""

    def test_workflow_extract_python_code_from_response(self):
        """
        AAA Test:
        Arrange: LLM response with Python code
        Act: Extract code from response
        Assert: Verify code is correctly extracted
        """
        # Arrange
        llm_responses = [
            'Here is the code:\n```python\nprint("Hello, World!")\n```',
            'You can use this:\n```python\nimport pandas as pd\ndf = pd.read_csv("data.csv")\n```',
            '```python\nfor i in range(10):\n    print(i)\n```',
        ]

        # Act
        for response in llm_responses:
            has_code, language, code = CodeDetector.detect_code_in_response(response)

            # Assert
            assert has_code is True
            assert language == 'python'
            assert len(code) > 0

    def test_workflow_with_mixed_code_blocks(self):
        """
        AAA Test:
        Arrange: Response with multiple code blocks
        Act: Detect code
        Assert: Verify first executable block is extracted
        """
        # Arrange
        response = '''Here is a Python solution:

```python
def calculate_sum(a, b):
    return a + b
```

And here is JavaScript:

```javascript
function add(a, b) {
    return a + b;
}
```
'''

        # Act
        has_code, language, code = CodeDetector.detect_code_in_response(response)

        # Assert
        assert has_code is True
        assert language == 'python'  # Python should be prioritized
        assert 'calculate_sum' in code


class TestCodeExecutionWorkflow:
    """Functional tests for code execution workflow."""

    def test_workflow_execute_simple_python(self):
        """
        AAA Test:
        Arrange: Simple Python code
        Act: Execute in sandbox
        Assert: Verify output is captured
        """
        # Arrange
        code = '''print("Hello from sandbox!")
result = 2 + 2
print(f"2 + 2 = {result}")
'''

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_result.stdout = "Hello from sandbox!\n2 + 2 = 4"
            mock_result.stderr = ""
            mock_result.exit_code = 0
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor()
            result = executor.execute(code)

            # Assert
            assert result.success is True
            assert "Hello from sandbox!" in result.stdout

    def test_workflow_with_resource_limits(self):
        """
        AAA Test:
        Arrange: Create executor with custom limits
        Act: Execute code
        Assert: Verify limits are applied
        """
        # Arrange
        custom_limits = ResourceLimits(
            max_cpu_time=5,
            max_memory_mb=128,
            max_processes=1
        )
        code = 'print("Quick execution")'

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_result.stdout = "Quick execution"
            mock_result.stderr = ""
            mock_result.exit_code = 0
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor(custom_limits)

            # Assert
            assert executor.limits.max_cpu_time == 5
            assert executor.limits.max_memory_mb == 128

    def test_workflow_detects_blocked_code(self):
        """
        AAA Test:
        Arrange: Code with blocked modules
        Act: Try to execute
        Assert: Verify code is rejected
        """
        # Arrange
        dangerous_codes = [
            'import os\nos.system("ls")',
            'import subprocess\nsubprocess.run(["ls"])',
            'eval("1 + 1")',
            'exec("print(\'test\')")',
        ]

        # Act & Assert
        executor = EnhancedSandboxedCodeExecutor()
        for code in dangerous_codes:
            error = executor._validate_code(code)

            # Assert
            assert error is not None, f"Should block: {code}"
            assert "not allowed" in error.lower() or "blocked" in error.lower()

    def test_workflow_allows_safe_code(self):
        """
        AAA Test:
        Arrange: Code with safe modules
        Act: Validate code
        Assert: Verify safe code passes validation
        """
        # Arrange
        safe_codes = [
            'import json\ndata = json.dumps({"key": "value"})',
            'import math\nresult = math.sqrt(16)',
            'from datetime import datetime\nnow = datetime.now()',
        ]

        # Act & Assert
        executor = EnhancedSandboxedCodeExecutor()
        for code in safe_codes:
            error = executor._validate_code(code)

            # Assert
            assert error is None, f"Should allow: {code}"


class TestCodeGenerationWorkflow:
    """Functional tests for code generation and execution workflow."""

    def test_workflow_generate_and_execute_plot(self):
        """
        AAA Test:
        Arrange: Code to generate a plot
        Act: Simulate execution workflow
        Assert: Verify plot generation code is detected
        """
        # Arrange
        user_request = "Build a chart showing sales data"
        llm_response = '''I'll create a bar chart for you:

```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C']
values = [10, 20, 15]

plt.bar(categories, values)
plt.title('Sales Data')
plt.savefig('sales_chart.png')
print('Chart saved as sales_chart.png')
```
'''

        # Act
        code_detected, _ = CodeDetector.detect_code_request(user_request)
        has_code, language, code = CodeDetector.detect_code_in_response(llm_response)

        # Assert
        assert code_detected is True, "Should detect code request"
        assert has_code is True, "Should detect code in response"
        assert language == 'python'
        assert 'matplotlib' in code

    def test_workflow_generate_pdf_document(self):
        """
        AAA Test:
        Arrange: Request to create PDF
        Act: Simulate workflow
        Assert: Verify PDF creation is detected
        """
        # Arrange
        user_request = "Generate a PDF invoice"
        llm_response = '''Here's code to create a PDF invoice:

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("invoice.pdf", pagesize=letter)
c.drawString(100, 750, "INVOICE")
c.drawString(100, 700, "Item 1: $10.00")
c.save()
print("Invoice created successfully")
```
'''

        # Act
        code_detected, _ = CodeDetector.detect_code_request(user_request)
        has_code, language, code = CodeDetector.detect_code_in_response(llm_response)

        # Assert
        assert code_detected is True
        assert has_code is True
        assert 'reportlab' in code

    def test_workflow_data_analysis_pipeline(self):
        """
        AAA Test:
        Arrange: Request for data analysis
        Act: Simulate full workflow
        Assert: Verify analysis steps are detected
        """
        # Arrange
        user_request = "Calculate statistics"
        llm_response = '''Here's the complete analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Calculate statistics
stats = df.describe()
print(stats)

# Create visualization
df.plot(kind='bar')
plt.savefig('analysis.png')
```
'''

        # Act
        request_detected, _ = CodeDetector.detect_code_request(user_request)
        has_code, language, code = CodeDetector.detect_code_in_response(llm_response)

        # Assert
        assert request_detected is True
        assert has_code is True
        assert 'pandas' in code
        assert 'pd.read_csv' in code


class TestCodeErrorHandlingWorkflow:
    """Functional tests for error handling in code execution."""

    def test_workflow_handles_syntax_error(self):
        """
        AAA Test:
        Arrange: Code with syntax error
        Act: Execute code
        Assert: Verify error is captured
        """
        # Arrange
        invalid_code = '''print("Missing quote)
x = 1 +  # Missing operand
'''

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "SyntaxError"
            mock_result.stderr = "SyntaxError: invalid syntax"
            mock_result.exit_code = 1
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor()
            result = executor.execute(invalid_code)

            # Assert
            assert result.success is False
            assert "SyntaxError" in result.error or "SyntaxError" in result.stderr

    def test_workflow_handles_runtime_error(self):
        """
        AAA Test:
        Arrange: Code that causes runtime error
        Act: Execute code
        Assert: Verify runtime error is captured
        """
        # Arrange
        runtime_error_code = '''x = 10
y = 0
result = x / y  # Division by zero
'''

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "ZeroDivisionError"
            mock_result.stderr = "ZeroDivisionError: division by zero"
            mock_result.exit_code = 1
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor()
            result = executor.execute(runtime_error_code)

            # Assert
            assert result.success is False

    def test_workflow_handles_timeout(self):
        """
        AAA Test:
        Arrange: Code that would run indefinitely
        Act: Execute with short timeout
        Assert: Verify timeout is enforced
        """
        # Arrange
        infinite_loop_code = '''while True:
    pass
'''

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Execution timeout exceeded"
            mock_result.exit_code = -1
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor(
                ResourceLimits(max_cpu_time=1)
            )
            result = executor.execute(infinite_loop_code)

            # Assert
            assert result.success is False
            assert "timeout" in result.error.lower()


class TestCodeExecutionWithFiles:
    """Functional tests for file operations in code execution."""

    def test_workflow_creates_output_file(self):
        """
        AAA Test:
        Arrange: Code that creates a file
        Act: Execute code
        Assert: Verify execution result structure
        """
        # Arrange
        file_creation_code = '''with open('output.txt', 'w') as f:
    f.write('Test content')
print('File created')
'''

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            from code_executor import ExecutionResult
            mock_result = ExecutionResult(
                success=True,
                output="File created",
                stdout="File created",
                exit_code=0
            )
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor()
            result = executor.execute(file_creation_code)

            # Assert
            assert result.success is True
            assert result.stdout == "File created"

    def test_workflow_downloads_generated_file(self):
        """
        AAA Test:
        Arrange: Execution that creates downloadable file
        Act: Get downloadable files
        Assert: Verify file is available for download
        """
        # Arrange
        with patch('code_executor.EnhancedSandboxedCodeExecutor._run_subprocess') as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_result.files_created = [
                {'filename': 'report.pdf', 'filepath': '/tmp/report.pdf',
                 'size': 1024, 'mime_type': 'application/pdf', 'hash': 'def456'}
            ]
            mock_result.exit_code = 0
            mock_run.return_value = mock_result

            executor = EnhancedSandboxedCodeExecutor()

            # Mock the downloadable files
            from code_executor import DownloadableFile
            from datetime import datetime

            executor._downloadable_files = [
                DownloadableFile(
                    filename='report.pdf',
                    filepath='/tmp/report.pdf',
                    size=1024,
                    mime_type='application/pdf',
                    created_at=datetime.now(),
                    file_hash='def456'
                )
            ]

            # Act
            files = executor.get_downloadable_files()

            # Assert
            assert len(files) > 0
            assert files[0].filename == 'report.pdf'
            assert files[0].mime_type == 'application/pdf'


class TestExecuteCodeConvenienceWorkflow:
    """Functional tests for convenience function workflow."""

    def test_workflow_simple_execution(self):
        """
        AAA Test:
        Arrange: Simple code
        Act: Use execute_code convenience function
        Assert: Verify execution succeeds
        """
        # Arrange
        code = 'print("Convenience test")'

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor') as MockExecutor:
            mock_instance = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.stdout = "Convenience test"
            mock_instance.execute.return_value = mock_result
            MockExecutor.return_value = mock_instance

            result = execute_code(code)

            # Assert
            assert result.success is True

    def test_workflow_with_custom_limits(self):
        """
        AAA Test:
        Arrange: Custom limits
        Act: Execute with limits
        Assert: Verify limits are passed through
        """
        # Arrange
        code = 'print("Limited execution")'
        limits = ResourceLimits(max_cpu_time=5)

        # Act
        with patch('code_executor.EnhancedSandboxedCodeExecutor') as MockExecutor:
            mock_instance = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_instance.execute.return_value = mock_result
            MockExecutor.return_value = mock_instance

            execute_code(code, limits)

            # Assert
            MockExecutor.assert_called_once_with(limits)
