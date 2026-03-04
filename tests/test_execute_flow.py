#!/usr/bin/env python3
"""Test the actual execute() method flow."""

import sys
import os
import tempfile

sys.path.insert(0, 'src')

# Mock psutil
try:
    import psutil
except ImportError:
    class MockPsUtil:
        class Process:
            def __init__(self, pid): pass
    sys.modules['psutil'] = MockPsUtil()

from code_executor import EnhancedSandboxedCodeExecutor

code = '''print("Hello from sandbox!")
x = 1 + 1
print(f"x = {x}")'''

print("=== Creating executor and calling execute() ===")
executor = EnhancedSandboxedCodeExecutor()

# Check temp_dir before execute
print(f"temp_dir before execute: {executor.temp_dir}")

result = executor.execute(code)

print(f"\n=== Result ===")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
print(f"Error: {result.error}")

if result.files_created:
    print(f"\nFiles created: {len(result.files_created)}")
    for f in result.files_created:
        print(f"  - {f['filename']}")
