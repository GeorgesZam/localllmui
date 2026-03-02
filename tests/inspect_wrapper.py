#!/usr/bin/env python3
"""Inspect the wrapper files that are created."""

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock psutil
try:
    import psutil
except ImportError:
    class MockPsUtil:
        class Process:
            def __init__(self, pid): pass
    sys.modules['psutil'] = MockPsUtil()

from code_executor import EnhancedSandboxedCodeExecutor

code = '''
print("Hello!")
x = 1 + 1
print(f"x = {x}")
'''

print("Creating executor and inspecting files...")
executor = EnhancedSandboxedCodeExecutor()

# Manually create the files
executor.temp_dir = tempfile.mkdtemp(prefix="inspect_")

# Create exec script
exec_script = executor._prepare_exec_script(code)

script_path = os.path.join(executor.temp_dir, "execute.py")
with open(script_path, 'w') as f:
    f.write(exec_script)

# Create wrapper script
wrapper_path = executor._create_wrapper_script(script_path)

print(f"\nTemp dir: {executor.temp_dir}")
print(f"\n--- execute.py (first 60 lines) ---")
with open(script_path, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:60], 1):
        print(f"{i:3}: {line.rstrip()}")

print(f"\n--- wrapper.sh ---")
with open(wrapper_path, 'r') as f:
    print(f.read())

print(f"\n--- Running wrapper directly ---")
result = subprocess.run(
    ["bash", wrapper_path],
    capture_output=True,
    text=True,
    timeout=10
)
print(f"Exit code: {result.returncode}")
print(f"\nStdout:\n{result.stdout[:500]}")
print(f"\nStderr:\n{result.stderr[:500]}")

# Keep files for inspection
print(f"\n⚠️ Files preserved at: {executor.temp_dir}")
print("   (You can inspect them manually - they won't be auto-cleaned)")
