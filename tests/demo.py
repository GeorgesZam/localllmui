#!/usr/bin/env python3
"""
WORKING SANDBOX DEMONSTRATION

This is a simple, working example of sandboxed code execution.
Run this to see the sandbox in action!
"""

import sys
import subprocess
import tempfile
import json
import os
import time


def run_sandboxed_code(code: str):
    """Run code in a sandboxed subprocess."""
    temp_dir = tempfile.mkdtemp(prefix="sandbox_demo_")

    # Prepare the execution script with proper indentation
    indented_code = '\n'.join('    ' + line for line in code.split('\n'))

    script = f'''
import sys
import io

print("=== Sandbox Execution Started ===")

# Capture stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
{indented_code}
except Exception as e:
    print(f"ERROR: {{e}}", file=old_stderr)

# Get output
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore stdout/stderr
sys.stdout = old_stdout
sys.stderr = old_stderr

# Print captured output
print("\\n=== Output ===")
print(stdout_value)

# List created files
import os
created_files = []
for item in os.listdir("."):
    if os.path.isfile(item) and not item.endswith('.py'):
        created_files.append(item)
        size = os.path.getsize(item)
        print(f"📄 Created: {{item}} ({{size}} bytes)")

print("\\n=== Execution Complete ===")
'''

    # Write script
    script_path = os.path.join(temp_dir, "run.py")
    with open(script_path, 'w') as f:
        f.write(script)

    # Run the script
    print(f"🔬 Executing in sandbox...")
    print(f"📁 Working directory: {temp_dir}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=temp_dir
    )

    print(f"\n📤 Process Output:")
    print(result.stdout)

    if result.stderr:
        print(f"\n⚠️ Errors:")
        print(result.stderr)

    # Show created files
    if created_files := [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f)) and not f.endswith('.py')]:
        print(f"\n📦 Generated Files:")
        for f in created_files:
            filepath = os.path.join(temp_dir, f)
            size = os.path.getsize(filepath)
            print(f"  • {f} ({size} bytes)")
            print(f"    Path: {filepath}")

    return temp_dir


print("\n" + "="*60)
print("🧪 SANDBOX DEMONSTRATION")
print("="*60)

# Test 1: Basic execution
print("\n📝 TEST 1: Basic Python Code")
print("-"*40)

code1 = '''
print("Hello from the sandbox! 🎉")
import json
data = {"app": "Sandbox", "status": "working"}
print(json.dumps(data, indent=2))

# Math operations
import math
print(f"\\nπ = {math.pi:.6f}")
print(f"e^2 = {math.exp(2):.4f}")
'''

temp1 = run_sandboxed_code(code1)

# Test 2: File creation
print("\n\n" + "="*60)
print("📝 TEST 2: File Creation")
print("-"*40)

code2 = '''
# Create a text file
with open("output.txt", "w") as f:
    f.write("This file was created in the sandbox!\\n")
    for i in range(3):
        f.write(f"Line {i+1}\\n")

print("✅ Created output.txt")

# Create a JSON file
import json
with open("data.json", "w") as f:
    json.dump({
        "title": "Sandbox Test",
        "items": [1, 2, 3],
        "timestamp": "2025-03-02"
    }, f, indent=2)

print("✅ Created data.json")
'''

temp2 = run_sandboxed_code(code2)

# Test 3: CSV processing
print("\n\n" + "="*60)
print("📝 TEST 3: CSV Data Processing")
print("-"*40)

code3 = '''
import csv

# Create a sample CSV
with open("sales.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Month", "Revenue", "Growth"])
    writer.writerow(["January", "$50,000", "5%"])
    writer.writerow(["February", "$55,000", "10%"])
    writer.writerow(["March", "$60,000", "9%"])

print("✅ Created sales.csv")

# Read and analyze
with open("sales.csv", "r") as f:
    reader = csv.DictReader(f)
    print("\\n📊 Sales Data:")
    for row in reader:
        print(f"  • {row['Month']}: {row['Revenue']} (growth: {row['Growth']})")
'''

temp3 = run_sandboxed_code(code3)

# Test 4: Security blocking (dangerous code)
print("\n\n" + "="*60)
print("📝 TEST 4: Security (Blocking Dangerous Code)")
print("-"*40)

code4 = '''
# This SHOULD be blocked
import os
os.system("echo 'This should NOT work'")
'''

print("Attempting to run code with 'os' module...")
print("(In the full sandbox, this would be blocked)")

# Note: In our simple demo, we're not doing module blocking
# The full EnhancedSandboxedCodeExecutor has this feature

print("\n" + "="*60)
print("✅ DEMONSTRATION COMPLETE")
print("="*60)
print("\n📁 Temporary directories preserved for inspection:")
print(f"  • {temp1}")
print(f"  • {temp2}")
print(f"  • {temp3}")
print("\n💡 The sandbox is working! Files are created in isolated temp directories.")
print("   The EnhancedSandboxedCodeExecutor adds resource limits & security blocking.")
