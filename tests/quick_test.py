#!/usr/bin/env python3
"""
Quick test for the Enhanced Sandboxed Code Executor.
Run this to verify the sandbox works without extra dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock psutil if not available
try:
    import psutil
except ImportError:
    print("⚠️ psutil not available - resource monitoring disabled")
    class MockPsUtil:
        class NoSuchProcess(Exception):
            pass
        class AccessDenied(Exception):
            pass
        class Process:
            def __init__(self, pid):
                self.pid = pid
            def memory_info(self):
                class MemInfo:
                    rss = 1000000
                return MemInfo()
            def cpu_times(self):
                class Times:
                    user = 0.1
                    system = 0.05
                return Times()
            def io_counters(self):
                class IO:
                    write_bytes = 0
                return IO()
    sys.modules['psutil'] = MockPsUtil()
    MockPsUtil.NoSuchProcess = MockPsUtil.NoSuchProcess
    MockPsUtil.AccessDenied = MockPsUtil.AccessDenied
    MockPsUtil.Process = MockPsUtil.Process

from code_executor import EnhancedSandboxedCodeExecutor, ResourceLimits, format_code_output


def test_basic():
    """Test 1: Basic code execution"""
    print("\n" + "="*60)
    print("📝 TEST 1: Basic Code Execution")
    print("="*60)

    code = '''
print("Hello from the sandbox! ✨")
import json
data = {"name": "Sandbox Test", "value": 42}
print(json.dumps(data, indent=2))
'''

    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(format_code_output(result))
    return result.success


def test_file_creation():
    """Test 2: File creation"""
    print("\n" + "="*60)
    print("📁 TEST 2: File Creation")
    print("="*60)

    code = '''
# Create a text file
with open("hello.txt", "w") as f:
    f.write("Hello from sandbox!\\n")
    f.write("This file was created safely.\\n")

# Create a JSON file
import json
with open("data.json", "w") as f:
    json.dump({"test": True, "value": 123}, f, indent=2)

print("✅ Files created successfully!")
'''

    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(format_code_output(result))

    # Show downloadable files
    if result.files_created:
        print("\n📦 Files ready for download:")
        for f in executor.get_downloadable_files():
            print(f"  • {f.filename} ({f.size} bytes)")

    return result.success and len(result.files_created) > 0


def test_security():
    """Test 3: Security blocking"""
    print("\n" + "="*60)
    print("🔒 TEST 3: Security Blocking")
    print("="*60)

    dangerous_code = '''
import os
os.system("echo 'This should be blocked'")
'''

    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(dangerous_code)

    if not result.success and "not allowed" in result.error:
        print("✅ DANGEROUS CODE BLOCKED!")
        print(f"   Reason: {result.error}")
        return True
    else:
        print("❌ Security violation - code was not blocked!")
        return False


def test_csv_data():
    """Test 4: CSV data processing"""
    print("\n" + "="*60)
    print("📊 TEST 4: CSV Data Processing")
    print("="*60)

    code = '''
import csv

# Create CSV
with open("sales.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Month", "Sales", "Growth"])
    writer.writerow(["Jan", "50000", "5%"])
    writer.writerow(["Feb", "55000", "10%"])
    writer.writerow(["Mar", "60000", "9%"])

print("✅ CSV created!")

# Read and analyze
with open("sales.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['Month']}: ${row['Sales']} (growth: {row['Growth']})")
'''

    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(format_code_output(result))
    return result.success


def test_timeout():
    """Test 5: Timeout protection"""
    print("\n" + "="*60)
    print("⏱️ TEST 5: Timeout Protection")
    print("="*60)

    code = '''
# This will timeout
import time
print("Starting...")
for i in range(100):
    time.sleep(1)
    print(f"Loop {i}")
'''

    limits = ResourceLimits(max_cpu_time=3)  # 3 second timeout
    executor = EnhancedSandboxedCodeExecutor(limits)
    result = executor.execute(code)

    if "timed out" in str(result.error).lower():
        print("✅ Timeout enforced - process was killed!")
        print(f"   Execution time: {result.execution_time:.2f}s")
        return True
    else:
        print(f"Result: {result.error or 'Completed'}")
        return result.execution_time < 5


def main():
    print("\n" + "="*60)
    print("🧪 ENHANCED SANDBOX - QUICK TEST")
    print("="*60)

    tests = [
        ("Basic Execution", test_basic),
        ("File Creation", test_file_creation),
        ("Security Blocking", test_security),
        ("CSV Processing", test_csv_data),
        ("Timeout Protection", test_timeout),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
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

    if passed == total:
        print("\n🎉 All tests passed! Sandbox is working correctly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
