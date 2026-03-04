#!/usr/bin/env python3
"""
Diagnostic script for llama-cpp-python on Windows.

This script helps diagnose llama-cpp-python issues when running
in PyInstaller builds on Windows.
"""

import sys
import os

print("=" * 60)
print("llama-cpp-python Windows Diagnostic Tool")
print("=" * 60)

# Check if running in PyInstaller
is_frozen = hasattr(sys, '_MEIPASS')
print(f"\nRunning in PyInstaller: {is_frozen}")

if is_frozen:
    print(f"PyInstaller temp dir: {sys._MEIPASS}")
    print(f"Executable: {sys.executable}")

# Try to import llama_cpp
print("\n" + "-" * 60)
print("Testing llama_cpp import...")
print("-" * 60)

try:
    import llama_cpp
    print("✓ llama_cpp imported successfully")

    # Get llama_cpp info
    print(f"  Location: {llama_cpp.__file__}")
    print(f"  Version: {getattr(llama_cpp, '__version__', 'Unknown')}")

    # Check for _llama_cpp (the compiled extension)
    if hasattr(llama_cpp, '_llama_cpp'):
        print("  ✓ _llama_cpp extension found")

        # Try to get the DLL file
        ext_module = llama_cpp._llama_cpp
        if hasattr(ext_module, '__file__'):
            dll_path = ext_module.__file__
            print(f"    DLL: {dll_path}")
            print(f"    Exists: {os.path.exists(dll_path)}")

            if not os.path.exists(dll_path):
                print("    ✗ ERROR: DLL file doesn't exist!")

                # Try to find alternative locations
                if is_frozen:
                    print("    Searching for alternative locations...")

                    meipass = sys._MEIPASS
                    dll_name = os.path.basename(dll_path)

                    # Common locations
                    search_paths = [
                        os.path.join(meipass, dll_name),
                        os.path.join(meipass, 'llama_cpp', dll_name),
                        os.path.join(meipass, 'lib', dll_name),
                    ]

                    for alt_path in search_paths:
                        if os.path.exists(alt_path):
                            print(f"    ✓ Found at: {alt_path}")
    else:
        print("  ✗ _llama_cpp extension NOT found")

    # Test basic functionality
    print("\n" + "-" * 60)
    print("Testing basic llama_cpp functionality...")
    print("-" * 60)

    try:
        # Try to create a Llama instance (without loading a model)
        print("  Testing Llama class instantiation...")

        # Just check if we can access the class
        LlamaClass = getattr(llama_cpp, 'Llama', None)
        if LlamaClass:
            print("  ✓ Llama class accessible")

            # Check class methods
            methods = [m for m in dir(LlamaClass) if not m.startswith('_')]
            print(f"  Available methods: {len(methods)}")
        else:
            print("  ✗ Llama class NOT accessible")

    except Exception as e:
        print(f"  ✗ Error: {e}")

except ImportError as e:
    print(f"✗ Failed to import llama_cpp: {e}")
    print("\nThis usually means llama-cpp-python is not installed.")
    print("Install with: pip install llama-cpp-python")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

# Check system info
print("\n" + "-" * 60)
print("System Information")
print("-" * 60)
print(f"Platform: {sys.platform}")
print(f"Python version: {sys.version}")
print(f"Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")

# Check for common DLL dependencies
print("\n" + "-" * 60)
print("Checking DLL dependencies...")
print("-" * 60)

if sys.platform == "win32":
    common_dlls = [
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_2.dll',
    ]

    for dll in common_dlls:
        # Check in system directories
        found = False
        for system_dir in [
            os.environ.get('SystemRoot', r'C:\Windows') + r'\System32',
            os.environ.get('SystemRoot', r'C:\Windows') + r'\SysWOW64',
        ]:
            dll_path = os.path.join(system_dir, dll)
            if os.path.exists(dll_path):
                print(f"  ✓ {dll}: {system_dir}")
                found = True
                break

        if not found:
            print(f"  ✗ {dll}: NOT FOUND")

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)
