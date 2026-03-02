#!/usr/bin/env python3
"""Debug the sandbox executor to see what's actually happening."""

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=== Debugging Sandbox Executor ===\n")

# Mock psutil
try:
    import psutil
except ImportError:
    class MockPsUtil:
        class Process:
            def __init__(self, pid): pass
            def memory_info(self):
                class M: rss = 1000000
                return M()
            def cpu_times(self):
                class T: user = 0.1; system = 0.05
                return T()
            def io_counters(self):
                class I: write_bytes = 0
                return I()
    sys.modules['psutil'] = MockPsUtil()

from code_executor import EnhancedSandboxedCodeExecutor, ResourceLimits

# Simple test code
code = '''
print("Hello from sandbox!")
x = 1 + 1
print(f"1 + 1 = {x}")
'''

print("Creating executor...")
executor = EnhancedSandboxedCodeExecutor()

print("\nExecuting code...")
print(f"Code: {code[:100]}")

result = executor.execute(code)

print(f"\n--- RESULT ---")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Exit code: {result.exit_code}")
print(f"\nStdout:\n{result.stdout}")
print(f"\nStderr:\n{result.stderr}")
if result.error:
    print(f"\nError:\n{result.error}")

# Check if temp dir still exists
if executor.temp_dir and os.path.exists(executor.temp_dir):
    print(f"\nTemp dir: {executor.temp_dir}")
    files = os.listdir(executor.temp_dir)
    print(f"Files in temp dir: {files}")
else:
    print(f"\nTemp dir was cleaned up")

print("\n=== Debug Complete ===")
