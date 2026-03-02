#!/usr/bin/env python3
"""Diagnose sandbox execution issues."""

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=== Sandbox Diagnostic ===\n")

# Test 1: Can we run subprocess at all?
print("1. Testing subprocess...")
try:
    result = subprocess.run(
        [sys.executable, "-c", "print('Hello')"],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"   ✅ subprocess works: {result.stdout.strip()}")
except Exception as e:
    print(f"   ❌ subprocess failed: {e}")

# Test 2: Can we create and run a temp script?
print("\n2. Testing temp script execution...")
try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Temp script works!")')
        temp_path = f.name

    result = subprocess.run(
        [sys.executable, temp_path],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"   ✅ Temp script: {result.stdout.strip()}")
    os.unlink(temp_path)
except Exception as e:
    print(f"   ❌ Temp script failed: {e}")

# Test 3: Test the actual sandbox code
print("\n3. Testing sandbox directly...")
code = '''
print("Test from sandbox!")
import json
print(json.dumps({"test": 123}))
'''

# Create temp dir
temp_dir = tempfile.mkdtemp(prefix="sandbox_test_")
print(f"   Temp dir: {temp_dir}")

try:
    # Write the code
    script_path = os.path.join(temp_dir, "test.py")
    with open(script_path, 'w') as f:
        f.write(code)

    # Run it
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=temp_dir
    )

    print(f"   Exit code: {result.returncode}")
    print(f"   stdout: {result.stdout[:200]}")
    if result.stderr:
        print(f"   stderr: {result.stderr[:200]}")

finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

# Test 4: Test wrapper script
print("\n4. Testing wrapper script...")
try:
    temp_dir = tempfile.mkdtemp(prefix="wrapper_test_")

    # Create test script
    test_py = os.path.join(temp_dir, "test.py")
    with open(test_py, 'w') as f:
        f.write('print("Wrapper test works!")')

    # Create wrapper
    wrapper_sh = os.path.join(temp_dir, "wrapper.sh")
    with open(wrapper_sh, 'w') as f:
        f.write(f'''#!/bin/bash
cd "{temp_dir}"
exec "{sys.executable}" "{test_py}"
''')
    os.chmod(wrapper_sh, 0o755)

    # Run wrapper
    result = subprocess.run(
        [wrapper_sh],
        capture_output=True,
        text=True,
        timeout=5
    )

    print(f"   Exit code: {result.returncode}")
    print(f"   stdout: {result.stdout[:200]}")
    if result.stderr:
        print(f"   stderr: {result.stderr[:200]}")

except Exception as e:
    print(f"   ❌ Wrapper failed: {e}")
finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

# Test 5: Simple executor without fancy features
print("\n5. Testing simple executor...")

try:
    temp_dir = tempfile.mkdtemp(prefix="simple_test_")

    code = '''
print("Simple test!")
import json
data = {"key": "value"}
print(json.dumps(data))
'''

    script_path = os.path.join(temp_dir, "run.py")
    with open(script_path, 'w') as f:
        # Simple wrapper without resource limits
        f.write(f'''
import sys
import io
import json

# Capture output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Change to temp dir
import os
os.chdir(r"{temp_dir}")

# Execute
try:
{code}
except Exception as e:
    print(f"ERROR: {{e}}")

# Capture
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore
sys.stdout = old_stdout
sys.stderr = old_stderr

# Output
print(json.dumps({{"stdout": stdout_value, "stderr": stderr_value}}))
''')

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=temp_dir
    )

    print(f"   Exit code: {result.returncode}")
    print(f"   Output preview: {result.stdout[:150]}...")

    # Try to parse JSON
    try:
        import json as json_mod
        output = json_mod.loads(result.stdout.strip().split('\n')[-1])
        print(f"   ✅ Parsed JSON response")
        print(f"   Captured stdout: {output.get('stdout', '')[:100]}")
    except Exception as e:
        print(f"   ❌ JSON parse failed: {e}")

finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

print("\n=== Diagnostic Complete ===")
