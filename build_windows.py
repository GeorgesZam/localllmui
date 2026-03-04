#!/usr/bin/env python3
"""
Improved build script for Windows PyInstaller builds.

This script provides better error handling and optimization
for building the application on Windows.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(f"STDERR: {result.stderr}", file=sys.stderr)

    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}", file=sys.stderr)
        return False

    return True


def clean_build_dirs():
    """Clean previous build directories."""
    print("\nCleaning previous build directories...")

    dirs_to_clean = ['build', 'dist', 'LocalRAG.spec']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
                print(f"Removed: {dir_name}/")
            else:
                os.remove(dir_name)
                print(f"Removed: {dir_name}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")

    required = ['PyInstaller', 'customtkinter', 'llama_cpp']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} - MISSING")

    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    return True


def build_app():
    """Build the application with PyInstaller."""
    print("\nBuilding application with PyInstaller...")

    # PyInstaller command with optimizations
    cmd = (
        "pyinstaller "
        "--clean "
        "--noconfirm "
        "LocalRAG.spec"
    )

    return run_command(cmd, "PyInstaller Build")


def test_executable():
    """Test the built executable."""
    exe_path = Path("dist/LocalRAG/LocalRAG.exe")

    if sys.platform == "win32" and exe_path.exists():
        print(f"\n{'='*60}")
        print(f"Executable created: {exe_path}")
        print(f"Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"{'='*60}")
        print("\nYou can test the application by running:")
        print(f"  {exe_path}")
        return True
    elif sys.platform != "win32":
        print("\nNOTE: Not on Windows, skipping executable test")
        return True
    else:
        print(f"\nERROR: Executable not found at {exe_path}")
        return False


def main():
    """Main build function."""
    print("="*60)
    print("LocalLLMUI Windows Build Script")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Clean previous builds
    if not run_command("echo Skipping clean", "Clean"):
        return 1

    # Build application
    if not build_app():
        print("\nERROR: Build failed!")
        return 1

    # Test executable
    if not test_executable():
        return 1

    print("\n" + "="*60)
    print("Build completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the executable in dist/LocalRAG/")
    print("2. If it works, you can distribute the LocalRAG folder")
    print("3. To debug issues, set console=True in LocalRAG.spec")

    return 0


if __name__ == "__main__":
    sys.exit(main())
