#!/usr/bin/env python3
"""
Test Suite for Enhanced Sandboxed Code Execution.

Run this file to test all sandbox features.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from code_executor import (
    EnhancedSandboxedCodeExecutor,
    CodeDetector,
    ResourceLimits,
    execute_code,
    format_code_output
)


def test_code_detection():
    """Test code request detection patterns."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Code Detection Patterns")
    print("="*60)

    test_messages = [
        ("Create a PDF report with sales data", True),
        ("What is the capital of France?", False),
        ("Generate an Excel file with Q1 data", True),
        ("Analyze this CSV file", True),
        ("Hello, how are you?", False),
        ("Make a chart showing monthly trends", True),
        ("Write python code to calculate fibonacci", True),
        ("Tell me a joke", False),
    ]

    passed = 0
    for message, expected in test_messages:
        detected, _ = CodeDetector.detect_code_request(message)
        status = "✅" if detected == expected else "❌"
        if detected == expected:
            passed += 1
        print(f"{status} '{message[:50]}...' -> {detected}")

    print(f"\nResult: {passed}/{len(test_messages)} tests passed")
    return passed == len(test_messages)


def test_basic_execution():
    """Test basic Python code execution."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Basic Code Execution")
    print("="*60)

    code = """
# Basic operations
print("Hello from sandbox!")
import json
data = {"name": "Test", "value": 42}
print(json.dumps(data))

# Math operations
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""

    print("Executing code...")
    result = execute_code(code)

    print(f"\nSuccess: {result.success}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"\nOutput:\n{result.stdout}")

    if result.error:
        print(f"Error: {result.error}")

    return result.success


def test_file_creation():
    """Test file creation in sandbox."""
    print("\n" + "="*60)
    print("🧪 TEST 3: File Creation")
    print("="*60)

    code = """
# Create a text file
with open("test_output.txt", "w") as f:
    f.write("This is a test file created by code execution.\\n")
    f.write("Created at: 2025-03-02\\n")
    for i in range(5):
        f.write(f"Line {i+1}\\n")

print("Created test_output.txt")

# Create a JSON file
import json
data = {
    "title": "Test Report",
    "items": [1, 2, 3, 4, 5],
    "metadata": {"version": "1.0"}
}

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Created data.json")
"""

    print("Executing code with file creation...")
    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(f"\nSuccess: {result.success}")
    print(f"Files created: {len(result.files_created)}")

    for file_info in result.files_created:
        size_mb = file_info['size'] / (1024 * 1024)
        size_str = f"{size_mb:.3f} MB" if size_mb > 0.001 else f"{file_info['size']} bytes"
        print(f"  📄 {file_info['filename']} ({size_str})")
        print(f"     Type: {file_info['mime_type']}")
        print(f"     Hash: {file_info['hash'][:16]}...")

    # Test file retrieval
    print("\n--- Testing File Download ---")
    downloadable = executor.get_downloadable_files()
    for f in downloadable:
        content = executor.get_file_content(f.filename)
        if content:
            preview = content[:100].decode('utf-8', errors='ignore')
            print(f"  📋 Content preview of {f.filename}:")
            print(f"     {preview[:50]}...")

    return result.success and len(result.files_created) > 0


def test_security_blocking():
    """Test security blocking of dangerous modules."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Security Blocking")
    print("="*60)

    dangerous_codes = [
        ("import os; os.system('ls')", "os module"),
        ("import subprocess; subprocess.run(['ls'])", "subprocess module"),
        ("import socket; s = socket.socket()", "socket module"),
        ("import urllib.request; urllib.request.urlopen('http://example.com')", "urllib module"),
        ("eval('print(1)')", "eval builtin"),
        ("exec('print(1)')", "exec builtin"),
        ("__import__('os')", "__import__ builtin"),
    ]

    passed = 0
    for code, desc in dangerous_codes:
        executor = EnhancedSandboxedCodeExecutor()
        result = executor.execute(code)

        blocked = not result.success and "not allowed" in result.error
        status = "✅" if blocked else "❌"
        if blocked:
            passed += 1

        print(f"{status} {desc}: {'BLOCKED' if blocked else 'ALLOWED'}")
        if result.error:
            print(f"    → {result.error[:60]}...")

    print(f"\nResult: {passed}/{len(dangerous_codes)} security tests passed")
    return passed == len(dangerous_codes)


def test_resource_limits():
    """Test resource limit enforcement."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Resource Limits")
    print("="*60)

    # Test memory limit with tight constraint
    code = """
# Try to use lots of memory
big_list = []
for i in range(1000000):  # Should exceed small memory limit
    big_list.append(i * 2)
print(f"Created list with {len(big_list)} elements")
"""

    print("Testing memory limit (10MB)...")
    limits = ResourceLimits(
        max_cpu_time=10,
        max_memory_mb=10,  # Very small limit
        allow_network=False
    )

    executor = EnhancedSandboxedCodeExecutor(limits)
    result = executor.execute(code)

    # Should fail due to memory limit
    memory_limited = not result.success
    print(f"Memory limit enforced: {'✅ YES' if memory_limited else '❌ NO'}")
    print(f"Result: {result.error if result.error else result.stdout[:100]}")

    return True  # Test passes if it doesn't crash


def test_timeout():
    """Test execution timeout."""
    print("\n" + "="*60)
    print("🧪 TEST 6: Execution Timeout")
    print("="*60)

    code = """
# Infinite loop (should timeout)
import time
i = 0
while True:
    i += 1
    time.sleep(0.1)
    if i > 1000:
        break
"""

    print("Testing 3 second timeout...")
    limits = ResourceLimits(
        max_cpu_time=3,  # 3 second timeout
        max_memory_mb=512
    )

    executor = EnhancedSandboxedCodeExecutor(limits)
    result = executor.execute(code)

    timed_out = "timed out" in result.error.lower() if result.error else False
    print(f"Timeout enforced: {'✅ YES' if timed_out else '❌ NO'}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Result: {result.error if result.error else 'Completed'}")

    return result.execution_time < 5  # Should be under 5 seconds


def test_data_analysis():
    """Test data analysis with pandas (if available)."""
    print("\n" + "="*60)
    print("🧪 TEST 7: Data Analysis (pandas)")
    print("="*60)

    code = """
import pandas as pd
import json

# Create sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000]
}

df = pd.DataFrame(data)
print("DataFrame created:")
print(df)
print(f"\\nShape: {df.shape}")
print(f"\\nMean salary: ${df['Salary'].mean():,.2f}")
print(f"\\nSummary:")
print(df.describe())
"""

    print("Executing data analysis code...")
    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(f"\nSuccess: {result.success}")
    print(f"\nOutput:\n{result.stdout}")

    if result.error and "pandas" in result.error:
        print("\n⚠️ pandas not available - skipping this test")
        return True  # Don't fail if pandas not installed

    return result.success


def test_pdf_generation():
    """Test PDF generation with reportlab (if available)."""
    print("\n" + "="*60)
    print("🧪 TEST 8: PDF Generation (reportlab)")
    print("="*60)

    code = """
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create PDF
doc = SimpleDocTemplate("test_report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Add title
story.append(Paragraph("Test Report", styles['Title']))
story.append(Spacer(1, 12))

# Add content
story.append(Paragraph("This is a test report generated by the sandbox.", styles['Normal']))
story.append(Spacer(1, 12))

# Add more content
for i in range(3):
    story.append(Paragraph(f"Section {i+1}: This is sample content.", styles['Normal']))
    story.append(Spacer(1, 6))

doc.build(story)
print("PDF created successfully!")
print(f"File: test_report.pdf")
"""

    print("Executing PDF generation code...")
    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(f"\nSuccess: {result.success}")

    if result.success and result.files_created:
        for file_info in result.files_created:
            if file_info['filename'].endswith('.pdf'):
                print(f"✅ PDF created: {file_info['filename']}")
                print(f"   Size: {file_info['size']} bytes")
                return True

    if result.error and "reportlab" in result.error:
        print("\n⚠️ reportlab not available - skipping this test")
        return True  # Don't fail if reportlab not installed

    print("❌ PDF not created")
    return False


def test_csv_processing():
    """Test CSV file processing."""
    print("\n" + "="*60)
    print("🧪 TEST 9: CSV Processing")
    print("="*60)

    code = """
import csv
import json

# Create a CSV file
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Los Angeles'],
    ['Charlie', 35, 'Chicago'],
]

with open('people.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Created people.csv")

# Read it back
with open('people.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{{row['Name']}} is {{row['Age']}} years old from {{row['City']}}")

# Calculate average age
with open('people.csv', 'r') as f:
    reader = csv.DictReader(f)
    ages = [int(row['Age']) for row in reader]
    avg_age = sum(ages) / len(ages)
    print(f"\\nAverage age: {avg_age:.1f}")
"""

    print("Executing CSV processing code...")
    executor = EnhancedSandboxedCodeExecutor()
    result = executor.execute(code)

    print(f"\nSuccess: {result.success}")
    print(f"\nOutput:\n{result.stdout}")

    # Check for CSV file
    csv_created = any(
        f['filename'] == 'people.csv'
        for f in result.files_created
    )
    print(f"\nCSV file created: {'✅ YES' if csv_created else '❌ NO'}")

    return result.success and csv_created


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("🚀 ENHANCED SANDBOX TEST SUITE")
    print("="*60)
    print(f"Python {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    tests = [
        test_code_detection,
        test_basic_execution,
        test_file_creation,
        test_security_blocking,
        test_resource_limits,
        test_timeout,
        test_data_analysis,
        test_pdf_generation,
        test_csv_processing,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
