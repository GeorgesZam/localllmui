#!/usr/bin/env python3
"""
Simple working sandbox test that you can run immediately.
This bypasses the complex resource monitoring to test core functionality.
"""

import sys
import os
import subprocess
import tempfile
import json
import time


def execute_code_simple(code: str, timeout: int = 10):
    """Execute code in a simple sandbox (no resource limits)."""
    temp_dir = tempfile.mkdtemp(prefix="simple_sandbox_")
    start_time = time.time()

    try:
        # Create a simpler execution script that just works
        # Indent user code for the try block
        indented_code = '\n'.join('    ' + line for line in code.split('\n'))

        script = f'''
import sys
import io

# Redirect output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Change to temp dir
os.chdir(r"{temp_dir}")

# Track created files
_created_files = []
_original_open = open

def tracking_open(file, mode='r', *args, **kwargs):
    result = _original_open(file, mode, *args, **kwargs)
    if 'w' in mode or 'a' in mode:
        try:
            if hasattr(result, 'name'):
                full_path = os.path.abspath(result.name)
                _created_files.append(full_path)
        except:
            pass
    return result

import builtins
builtins.open = tracking_open

# Execute user code
try:
{indented_code}
except Exception as e:
    print(f"ERROR: {{e}}")

# Capture output
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore
sys.stdout = old_stdout
sys.stderr = old_stderr

# Output results
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

        # Write and execute
        script_path = os.path.join(temp_dir, "execute.py")
        with open(script_path, 'w') as f:
            f.write(script)

        # Run
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=temp_dir
        )

        # Parse result
        if '---EXECUTION_RESULT---' in result.stdout:
            parts = result.stdout.split('---EXECUTION_RESULT---')
            output_before = parts[0]
            try:
                data = json.loads(parts[1].strip())
                return {
                    'success': data.get('success', True) and result.returncode == 0,
                    'output': output_before + data.get('stdout', ''),
                    'error': data.get('stderr', ''),
                    'files': data.get('files', []),
                    'execution_time': time.time() - start_time,
                    'stdout': data.get('stdout', ''),
                    'stderr': data.get('stderr', ''),
                }
            except json.JSONDecodeError:
                pass

        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr,
            'files': [],
            'execution_time': time.time() - start_time,
        }

    finally:
        # Don't clean up so we can inspect files
        print(f"\n📁 Temp dir preserved: {temp_dir}")


def test_basic():
    """Test basic code execution."""
    print("="*60)
    print("TEST: Basic Code Execution")
    print("="*60)

    code = '''print("Hello from sandbox!")
x = 1 + 1
print(f"1 + 1 = {x}")
import json
data = {"test": "success"}
print(json.dumps(data, indent=2))'''

    result = execute_code_simple(code)
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    print(f"\nOutput:\n{result['output']}")
    return result['success']


def test_file_creation():
    """Test file creation."""
    print("\n" + "="*60)
    print("TEST: File Creation")
    print("="*60)

    code = '''# Create a text file
with open("hello.txt", "w") as f:
    f.write("Hello from sandbox!\\n")
    f.write("This file was created safely.\\n")

print("✅ Created hello.txt")

# Create a JSON file
import json
with open("data.json", "w") as f:
    json.dump({"test": True, "value": 123}, f, indent=2)

print("✅ Created data.json")
'''

    result = execute_code_simple(code)
    print(f"Success: {result['success']}")
    print(f"Files created: {len(result['files'])}")

    for f in result['files']:
        filename = os.path.basename(f)
        size = os.path.getsize(f)
        print(f"  📄 {filename} ({size} bytes)")

    return result['success'] and len(result['files']) > 0


def test_csv():
    """Test CSV processing."""
    print("\n" + "="*60)
    print("TEST: CSV Processing")
    print("="*60)

    code = '''import csv

# Create CSV
with open("sales.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Month", "Sales", "Growth"])
    writer.writerow(["Jan", "50000", "5%"])
    writer.writerow(["Feb", "55000", "10%"])

print("✅ CSV created!")

# Read it back
with open("sales.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['Month']}: ${row['Sales']} (growth: {row['Growth']})")
'''

    result = execute_code_simple(code)
    print(f"\nOutput:\n{result['output']}")
    return result['success']


def main():
    print("\n" + "="*60)
    print("🧪 SIMPLE SANDBOX TEST")
    print("="*60)

    tests = [
        ("Basic Execution", test_basic),
        ("File Creation", test_file_creation),
        ("CSV Processing", test_csv),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅" if p else "❌"
        print(f"{status} {name}")

    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
