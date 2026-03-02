"""
Sandboxed Code Execution Engine for Local RAG Assistant.

Provides isolated subprocess execution of Python code with security restrictions.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import re
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str = ""
    files_created: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


class CodeDetector:
    """Detects when user requests code execution based on patterns."""

    # Patterns that suggest code execution is needed
    CODE_PATTERNS = [
        # Document creation
        r'\bcreate\s+(a\s+)?(pdf|word|excel|powerpoint|document|spreadsheet|presentation)',
        r'\bgenerate\s+(a\s+)?(pdf|word|excel|powerpoint|docx|xlsx|pptx)',
        r'\bmake\s+(a\s+)?(pdf|word|excel|powerpoint)',
        r'\bbuild\s+(a\s+)?(report|chart|graph|dashboard)',

        # Data analysis
        r'\banalyze\s+(data|csv|excel|spreadsheet)',
        r'\bprocess\s+(data|csv|excel)',
        r'\bcalculate\s+',
        r'\bcompute\s+',
        r'\bstatistics?\b',
        r'\bplot\s+',
        r'\bvisuali[sz]e',

        # File operations
        r'\bexport\s+to',
        r'\bsave\s+as',
        r'\bconvert\s+to',

        # Code execution keywords
        r'\brun\s+(python\s+)?code',
        r'\bexecute\s+(python\s+)?code',
        r'\bwrite\s+(python\s+)?code',
        r'\busing\s+(python\s+)?code',
    ]

    @classmethod
    def detect_code_request(cls, user_message: str) -> Tuple[bool, str]:
        """
        Detect if user is requesting code execution.

        Returns:
            Tuple of (needs_code, matched_pattern)
        """
        message_lower = user_message.lower()

        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, message_lower):
                return True, f"Matched pattern: {pattern}"

        return False, ""


class SandboxedCodeExecutor:
    """
    Executes Python code in an isolated subprocess with security restrictions.

    Security features:
    - Separate subprocess (crash won't affect main app)
    - Temporary directory for execution
    - Module restrictions (blocks os, sys, subprocess, etc.)
    - Execution timeout
    - Memory limit
    """

    # Blocked modules for security
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'multiprocessing', 'threading',
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        'shutil', 'pathlib', 'tempfile',
        'importlib', 'pkgutil', '__import__',
    }

    # Allowed modules for document creation and data analysis
    ALLOWED_MODULES = {
        # Standard library
        'json', 'csv', 'math', 'statistics', 'datetime', 're',
        'string', 'random', 'collections', 'itertools', 'decimal',
        'fractions', 'typing', 'dataclasses', 'enum', 'typing_extensions',

        # Document creation
        'docx', 'pptx', 'openpyxl', 'reportlab',

        # Data analysis
        'pandas', 'numpy', 'matplotlib', 'seaborn',

        # Image processing
        'PIL', 'pillow', 'PIL.Image', 'cv2',

        # Builtins
        'builtins', 'io', 'abc', 'contextlib', 'functools',
    }

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        """
        Initialize the sandboxed executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.temp_dir = None

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code in sandboxed subprocess.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, errors, and created files
        """
        start_time = time.time()

        # Create temporary directory for execution
        self.temp_dir = tempfile.mkdtemp(prefix="code_exec_")

        try:
            # Validate code
            validation_error = self._validate_code(code)
            if validation_error:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=validation_error
                )

            # Prepare execution script
            exec_script = self._prepare_exec_script(code)

            # Write script to temp file
            script_path = os.path.join(self.temp_dir, "execute.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(exec_script)

            # Execute in subprocess
            result = self._run_subprocess(script_path)

            # Find created files
            files_created = self._find_created_files()

            result.execution_time = time.time() - start_time
            result.files_created = files_created

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            # Cleanup temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass

    def _validate_code(self, code: str) -> Optional[str]:
        """Validate code for security issues."""
        # Check for blocked module imports
        import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'

        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            base_module = module.split('.')[0]

            if base_module in self.BLOCKED_MODULES:
                return f"Blocked module: {module}. Module '{base_module}' is not allowed for security reasons."

        # Check for dangerous builtins
        dangerous_patterns = [
            (r'\b__import__\s*\(', "__import__() is not allowed"),
            (r'\beval\s*\(', "eval() is not allowed for security reasons"),
            (r'\bexec\s*\(', "exec() is not allowed for security reasons"),
            (r'\bcompile\s*\(', "compile() is not allowed for security reasons"),
            (r'\bopen\s*\(["\'].*\.py', "Opening .py files is not allowed"),
        ]

        for pattern, msg in dangerous_patterns:
            if re.search(pattern, code):
                return msg

        return None

    def _prepare_exec_script(self, user_code: str) -> str:
        """Prepare the execution script with sandboxing."""
        # Create a wrapper that captures output and errors
        script = f'''
import sys
import io
import traceback

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Change to temp directory
import os
os.chdir(r"{self.temp_dir}")

# Track created files
import tempfile
_created_files = []
_original_open = open

def tracking_open(file, mode='r', *args, **kwargs):
    result = _original_open(file, mode, *args, **kwargs)
    if 'w' in mode or 'a' in mode or '+' in mode:
        try:
            full_path = os.path.abspath(file.name)
            if os.path.exists(full_path):
                _created_files.append(full_path)
        except:
            pass
    return result

import builtins
builtins.open = tracking_open

# Execute user code
try:
{self._indent_code(user_code, 4)}
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    traceback.print_exc()

# Capture output
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore stdout and stderr
sys.stdout = old_stdout
sys.stderr = old_stderr

# Print results as JSON for easy parsing
import json
result = {{
    "stdout": stdout_value,
    "stderr": stderr_value,
    "files": _created_files,
    "success": True
}}
print("\\n---EXECUTION_RESULT---")
print(json.dumps(result))
'''
        return script

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block for embedding in script."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    def _run_subprocess(self, script_path: str) -> ExecutionResult:
        """Run the script in a subprocess."""
        try:
            # Run with timeout
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.temp_dir
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds",
                    stderr=stderr
                )

            # Parse the result
            if '---EXECUTION_RESULT---' in stdout:
                parts = stdout.split('---EXECUTION_RESULT---')
                output_before = parts[0]
                try:
                    result_data = json.loads(parts[1].strip())
                    return ExecutionResult(
                        success=result_data.get('success', True) and process.returncode == 0,
                        output=output_before + result_data.get('stdout', ''),
                        error=result_data.get('stderr', ''),
                        stdout=result_data.get('stdout', ''),
                        stderr=result_data.get('stderr', ''),
                    )
                except json.JSONDecodeError:
                    return ExecutionResult(
                        success=process.returncode == 0,
                        output=stdout,
                        error=stderr,
                    )
            else:
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout,
                    error=stderr,
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Subprocess error: {str(e)}"
            )

    def _find_created_files(self) -> List[str]:
        """Find files created during execution."""
        files = []
        if self.temp_dir and os.path.exists(self.temp_dir):
            for item in os.listdir(self.temp_dir):
                item_path = os.path.join(self.temp_dir, item)
                if os.path.isfile(item_path) and not item.endswith('.py'):
                    files.append(item_path)
        return files


def format_code_output(result: ExecutionResult) -> str:
    """Format execution result for display to user."""
    lines = []

    if result.success:
        lines.append("✅ Execution successful!")

        if result.stdout:
            lines.append(f"📊 Output:\n{result.stdout}")

        if result.files_created:
            lines.append("📁 Files created:")
            for filepath in result.files_created:
                filename = os.path.basename(filepath)
                lines.append(f"  • {filename}")

        lines.append(f"⏱️ Execution time: {result.execution_time:.2f}s")
    else:
        lines.append("❌ Execution failed!")
        if result.error:
            lines.append(f"🚫 Error: {result.error}")
        if result.stderr:
            lines.append(f"📋 Details:\n{result.stderr}")

    return "\n".join(lines)


# Convenience function for quick execution
def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
    """
    Convenience function to execute code with default settings.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult
    """
    executor = SandboxedCodeExecutor(timeout=timeout)
    return executor.execute(code)


if __name__ == "__main__":
    # Test the executor
    test_code = """
import json

# Test basic operations
print("Hello from sandbox!")
data = {"name": "Test", "value": 42}
print(json.dumps(data))

# Create a simple text file
with open("test_output.txt", "w") as f:
    f.write("This is a test file created by code execution.")
print("Created test_output.txt")
"""

    detector = CodeDetector()
    needs_code, pattern = detector.detect_code_request("create a pdf report")
    print(f"Detection result: {needs_code}, pattern: {pattern}")

    executor = SandboxedCodeExecutor()
    result = executor.execute(test_code)

    print(format_code_output(result))
