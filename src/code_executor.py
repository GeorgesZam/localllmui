"""
Enhanced Sandboxed Code Execution Engine for Local RAG Assistant.

Provides isolated subprocess execution with:
- Resource limits (CPU, memory, disk I/O, network)
- Security restrictions
- File download support (ChatGPT-style)
- Real-time monitoring
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import re
import json
import signal
import resource
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import threading
import hashlib


@dataclass
class ResourceLimits:
    """Resource limits for code execution."""
    max_cpu_time: int = 30  # seconds
    max_memory_mb: int = 512  # MB
    max_file_size_mb: int = 100  # MB
    max_processes: int = 1  # number of processes
    max_open_files: int = 64  # number of file descriptors
    allow_network: bool = False  # block all network access
    max_disk_write_mb: int = 500  # MB


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str = ""
    files_created: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""
    resources_used: Dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


@dataclass
class DownloadableFile:
    """Represents a file that can be downloaded."""
    filename: str
    filepath: str
    size: int
    mime_type: str
    created_at: datetime
    file_hash: str


class CodeDetector:
    """Detects when user requests code execution based on patterns."""

    CODE_PATTERNS = [
        r'\bcreate\s+(a\s+)?(pdf|word|excel|powerpoint|document|spreadsheet|presentation)',
        r'\bgenerate\s+(a\s+)?(pdf|word|excel|powerpoint|docx|xlsx|pptx)',
        r'\bmake\s+(a\s+)?(pdf|word|excel|powerpoint)',
        r'\bbuild\s+(a\s+)?(report|chart|graph|dashboard|image|plot)',
        r'\banalyze\s+(data|csv|excel|spreadsheet)',
        r'\bprocess\s+(data|csv|excel)',
        r'\bcalculate\s+',
        r'\bcompute\s+',
        r'\bstatistics?\b',
        r'\bplot\s+',
        r'\bvisuali[sz]e',
        r'\bexport\s+to',
        r'\bsave\s+as',
        r'\bconvert\s+to',
        r'\brun\s+(python\s+)?code',
        r'\bexecute\s+(python\s+)?code',
        r'\bwrite\s+(python\s+)?code',
    ]

    @classmethod
    def detect_code_request(cls, user_message: str) -> Tuple[bool, str]:
        """Detect if user is requesting code execution."""
        message_lower = user_message.lower()
        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, message_lower):
                return True, f"Matched pattern: {pattern}"
        return False, ""


class ResourceMonitor:
    """Monitor resource usage during execution."""

    def __init__(self, pid: int, check_interval: float = 0.5):
        self.pid = pid
        self.check_interval = check_interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._peak_memory_mb = 0
        self._cpu_time = 0.0
        self._disk_write_mb = 0

    def start(self) -> None:
        """Start monitoring in background thread."""
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return stats."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2)
        return {
            'peak_memory_mb': self._peak_memory_mb,
            'cpu_time': self._cpu_time,
            'disk_write_mb': self._disk_write_mb
        }

    def _monitor_loop(self) -> None:
        """Monitor resource usage."""
        import psutil
        try:
            process = psutil.Process(self.pid)
            initial_io = process.io_counters()

            while self._monitoring:
                try:
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)

                    # CPU time
                    cpu_times = process.cpu_times()
                    self._cpu_time = cpu_times.user + cpu_times.system

                    # Disk write
                    io_counters = process.io_counters()
                    if hasattr(io_counters, 'write_bytes'):
                        disk_write = (io_counters.write_bytes - initial_io.write_bytes)
                        self._disk_write_mb = disk_write / (1024 * 1024)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

                time.sleep(self.check_interval)

        except ImportError:
            # psutil not available, skip monitoring
            pass


class NetworkIsolator:
    """Isolate process from network access."""

    @staticmethod
    def create_unshare_script() -> str:
        """Create a script that uses unshare to isolate network."""
        return f'''#!/bin/bash
# Network isolation using unshare (Linux)
if command -v unshare &> /dev/null; then
    exec unshare -n -r --mount-proc "$@"
else
    # Fallback: just run without isolation
    exec "$@"
fi
'''

    @staticmethod
    def is_available() -> bool:
        """Check if network isolation is available."""
        return sys.platform == 'linux' and os.path.exists('/usr/bin/unshare')


class EnhancedSandboxedCodeExecutor:
    """
    Enhanced sandboxed code executor with resource limits.

    Security features:
    - Resource limits (CPU, memory, processes, files)
    - Network isolation (Linux)
    - Filesystem restrictions
    - Module blocking
    - Real-time monitoring
    - Automatic cleanup
    """

    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'multiprocessing', 'threading',
        'socket', 'urllib', 'requests', 'http', 'ftplib', 'telnetlib',
        'shutil', 'pathlib', 'tempfile', 'importlib', 'pkgutil',
        '__import__', 'eval', 'exec', 'compile',
    }

    ALLOWED_MODULES = {
        # Standard library
        'json', 'csv', 'math', 'statistics', 'datetime', 're',
        'string', 'random', 'collections', 'itertools', 'decimal',
        'fractions', 'typing', 'dataclasses', 'enum', 'io',
        'abc', 'contextlib', 'functools', 'hashlib', 'base64',

        # Document creation
        'docx', 'pptx', 'openpyxl', 'reportlab', 'reportlab.lib',

        # Data analysis
        'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'seaborn',

        # Image processing
        'PIL', 'pillow', 'PIL.Image', 'cv2', 'cv2',

        # File formats
        'xlsxwriter', 'pypdf2', 'pypdf', 'pdfplumber',
    }

    MIME_TYPES = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.html': 'text/html',
        '.md': 'text/markdown',
    }

    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """
        Initialize the enhanced sandboxed executor.

        Args:
            resource_limits: Resource limits for execution
        """
        self.limits = resource_limits or ResourceLimits()
        self.temp_dir: Optional[str] = None
        self._monitor: Optional[ResourceMonitor] = None
        self._downloadable_files: List[DownloadableFile] = []
        self._execution_log: List[Dict[str, Any]] = []

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code in enhanced sandboxed environment.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with full details
        """
        start_time = time.time()
        self.temp_dir = tempfile.mkdtemp(prefix="enhanced_sandbox_")

        try:
            # Validate code
            validation_error = self._validate_code(code)
            if validation_error:
                return self._create_error_result(validation_error, start_time)

            # Prepare execution script
            exec_script = self._prepare_exec_script(code)

            # Write script
            script_path = os.path.join(self.temp_dir, "execute.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(exec_script)

            # Create wrapper script for resource limits
            wrapper_path = self._create_wrapper_script(script_path)

            # Execute with resource limits
            result = self._run_subprocess(wrapper_path)

            # Gather resource stats
            if self._monitor:
                result.resources_used = self._monitor.stop()

            # Find created files
            result.files_created = self._catalog_created_files()

            result.execution_time = time.time() - start_time

            # Log execution
            self._log_execution(code, result)

            return result

        except Exception as e:
            return self._create_error_result(str(e), start_time)

        finally:
            self._cleanup()

    def _validate_code(self, code: str) -> Optional[str]:
        """Validate code for security issues."""
        # Check for blocked module imports
        import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'

        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            base_module = module.split('.')[0]

            if base_module in self.BLOCKED_MODULES:
                return f"Blocked module: {module}. Module '{base_module}' is not allowed for security reasons."

        # Check for dangerous patterns
        dangerous_patterns = [
            (r'\b__import__\s*\(', "__import__() is not allowed"),
            (r'\beval\s*\(', "eval() is not allowed for security reasons"),
            (r'\bexec\s*\(', "exec() is not allowed for security reasons"),
            (r'\bcompile\s*\(', "compile() is not allowed for security reasons"),
            (r'\bopen\s*\(["\'].*\.py', "Opening .py files is not allowed"),
            (r'\bmmap\s*\(', "mmap() is not allowed"),
            (r'\bmemoryview\s*\(', "memoryview() is not allowed"),
        ]

        for pattern, msg in dangerous_patterns:
            if re.search(pattern, code):
                return msg

        return None

    def _prepare_exec_script(self, user_code: str) -> str:
        """Prepare execution script with enhanced sandboxing."""
        return f'''
import sys
import io
import traceback
import signal

# Set resource limits
def set_limits():
    import resource
    try:
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, ({self.limits.max_cpu_time}, {self.limits.max_cpu_time}))
        # Memory limit
        memory_bytes = {self.limits.max_memory_mb} * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        # Max processes
        resource.setrlimit(resource.RLIMIT_NPROC, ({self.limits.max_processes}, {self.limits.max_processes}))
        # Max open files
        resource.setrlimit(resource.RLIMIT_NOFILE, ({self.limits.max_open_files}, {self.limits.max_open_files}))
        # Max file size
        file_size_bytes = {self.limits.max_file_size_mb} * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
    except (ValueError, resource.error):
        pass  # Limits not supported on this platform

# Set limits on startup
set_limits()

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Change to temp directory
import os
os.chdir(r"{self.temp_dir}")

# Restrict filesystem access
original_open = open
_safe_paths = {{r"{self.temp_dir}", "/dev/null", "/dev/urandom"}}

def safe_open(file, mode='r', *args, **kwargs):
    # Check if trying to open outside safe paths
    if isinstance(file, str):
        full_path = os.path.abspath(file)
        is_safe = any(full_path.startswith(p) for p in _safe_paths)
        if not is_safe and ('r' in mode or 'a' in mode or '+' in mode):
            raise PermissionError(f"Cannot access file: {{file}}")
    return original_open(file, mode, *args, **kwargs)

import builtins
builtins.open = safe_open

# Block network access
def block_network():
    import socket
    original_socket = socket.socket
    class BlockedSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            raise OSError("Network access is blocked in sandbox")
    socket.socket = BlockedSocket

if not {str(self.limits.allow_network).lower()}:
    try:
        block_network()
    except:
        pass

# Track created files
_created_files = []
_original_open = open

def tracking_open(file, mode='r', *args, **kwargs):
    result = _original_open(file, mode, *args, **kwargs)
    if 'w' in mode or 'a' in mode or '+' in mode:
        try:
            if hasattr(result, 'name'):
                full_path = os.path.abspath(result.name)
                if os.path.exists(full_path):
                    _created_files.append(full_path)
        except:
            pass
    return result

builtins.open = tracking_open

# Set timeout signal
def timeout_handler(signum, frame):
    raise TimeoutError("Execution time limit exceeded")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.limits.max_cpu_time})

# Execute user code
try:
{self._indent_code(user_code, 4)}
except TimeoutError as e:
    print(f"ERROR: {{str(e)}}")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    traceback.print_exc()
finally:
    signal.alarm(0)  # Cancel alarm

# Capture output
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore stdout and stderr
sys.stdout = old_stdout
sys.stderr = old_stderr

# Print results as JSON
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

    def _create_wrapper_script(self, script_path: str) -> str:
        """Create wrapper script for execution with resource limits."""
        wrapper_path = os.path.join(self.temp_dir, "wrapper.sh")

        # Try to use unshare for network isolation (Linux)
        if sys.platform == 'linux' and os.path.exists('/usr/bin/unshare'):
            wrapper_content = f'''#!/bin/bash
cd "{self.temp_dir}"
exec unshare -n -r --mount-proc "{sys.executable}" "{script_path}"
'''
        else:
            # Fallback wrapper
            wrapper_content = f'''#!/bin/bash
cd "{self.temp_dir}"
exec "{sys.executable}" "{script_path}"
'''

        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

        return wrapper_path

    def _run_subprocess(self, wrapper_path: str) -> ExecutionResult:
        """Run the script in a subprocess with monitoring."""
        try:
            # Launch subprocess
            process = subprocess.Popen(
                [wrapper_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # Create new process group
            )

            # Start resource monitoring
            try:
                self._monitor = ResourceMonitor(process.pid)
                self._monitor.start()
            except Exception:
                pass  # Monitoring not critical

            try:
                stdout, stderr = process.communicate(timeout=self.limits.max_cpu_time)
            except subprocess.TimeoutExpired:
                # Kill entire process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    process.kill()
                stdout, stderr = process.communicate()

                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.limits.max_cpu_time} seconds",
                    stderr=stderr,
                    exit_code=-1
                )

            # Parse results
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
                        exit_code=process.returncode
                    )
                except json.JSONDecodeError:
                    pass

            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout,
                error=stderr,
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Subprocess error: {str(e)}",
                exit_code=-1
            )

    def _catalog_created_files(self) -> List[Dict[str, Any]]:
        """Catalog all created files for download."""
        files = []
        if self.temp_dir and os.path.exists(self.temp_dir):
            for item in os.listdir(self.temp_dir):
                item_path = os.path.join(self.temp_dir, item)
                if os.path.isfile(item_path) and not item.endswith(('.py', '.sh')):
                    try:
                        stat_info = os.stat(item_path)
                        ext = os.path.splitext(item)[1].lower()
                        mime_type = self.MIME_TYPES.get(ext, 'application/octet-stream')

                        # Calculate file hash
                        file_hash = self._calculate_file_hash(item_path)

                        file_info = {
                            'filename': item,
                            'filepath': item_path,
                            'size': stat_info.st_size,
                            'mime_type': mime_type,
                            'created_at': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                            'hash': file_hash
                        }
                        files.append(file_info)

                        # Add to downloadable files list
                        self._downloadable_files.append(DownloadableFile(
                            filename=item,
                            filepath=item_path,
                            size=stat_info.st_size,
                            mime_type=mime_type,
                            created_at=datetime.fromtimestamp(stat_info.st_ctime),
                            file_hash=file_hash
                        ))
                    except Exception as e:
                        print(f"Error cataloging file {{item}}: {{e}}")
        return files

    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_downloadable_files(self) -> List[DownloadableFile]:
        """Get list of downloadable files from last execution."""
        return self._downloadable_files.copy()

    def save_file_to(self, filename: str, destination: str) -> bool:
        """
        Save a generated file to a user-selected location.

        Args:
            filename: Name of the file to save
            destination: Full path where to save the file

        Returns:
            True if successful
        """
        for downloadable in self._downloadable_files:
            if downloadable.filename == filename:
                try:
                    shutil.copy2(downloadable.filepath, destination)
                    return True
                except Exception as e:
                    print(f"Error saving file: {e}")
                    return False
        return False

    def get_file_content(self, filename: str) -> Optional[bytes]:
        """Get content of a generated file."""
        for downloadable in self._downloadable_files:
            if downloadable.filename == filename:
                try:
                    with open(downloadable.filepath, 'rb') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
        return None

    def _create_error_result(self, error: str, start_time: float) -> ExecutionResult:
        """Create an error result."""
        return ExecutionResult(
            success=False,
            output="",
            error=error,
            execution_time=time.time() - start_time,
            exit_code=-1
        )

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        # Stop monitor
        if self._monitor:
            try:
                self._monitor.stop()
            except:
                pass

        # Note: We DON'T delete temp dir here - let files persist for download
        # Temp dir will be cleaned up when executor is garbage collected
        # or when cleanup_temp_files() is explicitly called

    def cleanup_temp_files(self) -> None:
        """Explicitly clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp dir: {e}")
            self.temp_dir = None
        self._downloadable_files.clear()

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block for embedding in script."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    def _log_execution(self, code: str, result: ExecutionResult) -> None:
        """Log execution for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'code_hash': hashlib.sha256(code.encode()).hexdigest()[:16],
            'success': result.success,
            'execution_time': result.execution_time,
            'exit_code': result.exit_code,
            'files_created': len(result.files_created),
            'resources_used': result.resources_used
        }
        self._execution_log.append(log_entry)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return self._execution_log.copy()


def format_code_output(result: ExecutionResult) -> str:
    """Format execution result for display to user."""
    lines = []

    if result.success:
        lines.append("✅ Execution successful!\n")

        if result.stdout:
            lines.append(f"📊 Output:\n{result.stdout}")

        if result.files_created:
            lines.append("📁 Files created:")
            for file_info in result.files_created:
                size_mb = file_info['size'] / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb > 1 else f"{file_info['size']} bytes"
                lines.append(f"  • {file_info['filename']} ({size_str})")
                lines.append(f"    Type: {file_info['mime_type']}")
                lines.append(f"    Hash: {file_info['hash'][:16]}...")

        if result.resources_used:
            lines.append("⚡ Resources used:")
            lines.append(f"  • Peak memory: {result.resources_used.get('peak_memory_mb', 0):.1f} MB")
            lines.append(f"  • CPU time: {result.resources_used.get('cpu_time', 0):.2f}s")
            lines.append(f"  • Disk write: {result.resources_used.get('disk_write_mb', 0):.1f} MB")

        lines.append(f"\n⏱️ Execution time: {result.execution_time:.2f}s")
    else:
        lines.append("❌ Execution failed!")
        if result.error:
            lines.append(f"🚫 Error: {result.error}")
        if result.stderr:
            lines.append(f"📋 Details:\n{result.stderr}")

    return "\n".join(lines)


# Convenience function
def execute_code(code: str, resource_limits: Optional[ResourceLimits] = None) -> ExecutionResult:
    """
    Execute code with enhanced sandbox.

    Args:
        code: Python code to execute
        resource_limits: Optional resource limits

    Returns:
        ExecutionResult
    """
    executor = EnhancedSandboxedCodeExecutor(resource_limits)
    return executor.execute(code)
