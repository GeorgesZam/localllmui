"""
Ollama Installer for Windows

This script handles downloading and installing Ollama on Windows.
It can be bundled with the application and run automatically if Ollama is not detected.

Usage:
    python install_ollama.py [--check] [--install] [--start]
"""

import os
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path


class OllamaWindowsInstaller:
    """Handles Ollama installation on Windows."""

    OLLAMA_VERSION = "0.5.7"
    OLLAMA_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/OllamaSetup.exe"
    INSTALLER_NAME = "OllamaSetup.exe"

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir())
        self.installer_path = self.temp_dir / self.INSTALLER_NAME
        self.ollama_path = self._find_ollama_executable()

    def _find_ollama_executable(self) -> Path:
        """Find where Ollama is installed."""
        # Check common installation paths
        common_paths = [
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Ollama",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama",
            Path(os.environ.get("APPDATA", "")) / "Ollama",
        ]

        for path in common_paths:
            exe = path / "ollama.exe"
            if exe.exists():
                return exe

        # Check if it's in PATH
        try:
            result = subprocess.run(
                ["where", "ollama"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass

        return None

    def is_installed(self) -> bool:
        """Check if Ollama is installed."""
        return self.ollama_path is not None and self.ollama_path.exists()

    def is_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_version(self) -> str:
        """Get Ollama version if installed."""
        if not self.is_installed():
            return "Not installed"

        try:
            result = subprocess.run(
                [str(self.ollama_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "Unknown"

    def download(self, on_progress=None) -> tuple[bool, str]:
        """
        Download Ollama installer.

        Args:
            on_progress: Callback function(downloaded_bytes, total_bytes)

        Returns:
            (success, message)
        """
        if self.installer_path.exists():
            return True, f"Installer already downloaded: {self.installer_path}"

        def report_progress(block_num, block_size, total_size):
            if on_progress and total_size > 0:
                downloaded = block_num * block_size
                on_progress(downloaded, total_size)
                if downloaded >= total_size or block_num % 10 == 0:
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end="")

        try:
            print(f"\nDownloading Ollama from {self.OLLAMA_URL}")
            print(f"Saving to: {self.installer_path}")

            urllib.request.urlretrieve(
                self.OLLAMA_URL,
                str(self.installer_path),
                reporthook=report_progress
            )

            file_size = self.installer_path.stat().st_size / (1024 * 1024)
            print(f"\nDownload complete: {file_size:.1f}MB")

            return True, str(self.installer_path)

        except Exception as e:
            return False, f"Download failed: {e}"

    def install(self, wait=False) -> tuple[bool, str]:
        """
        Run Ollama installer.

        Args:
            wait: If True, wait for installer to complete

        Returns:
            (success, message)
        """
        if not self.installer_path.exists():
            return False, f"Installer not found at {self.installer_path}"

        try:
            print(f"Running Ollama installer...")

            if wait:
                # Synchronous install - wait for completion
                result = subprocess.run(
                    [str(self.installer_path)],
                    timeout=300  # 5 minutes max
                )
                if result.returncode == 0:
                    return True, "Installation completed"
                else:
                    return False, f"Installer returned code {result.returncode}"
            else:
                # Asynchronous - launch and return
                subprocess.Popen(
                    [str(self.installer_path)],
                    shell=True
                )
                return True, "Installer launched. Please complete the installation and restart the application."

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Installation failed: {e}"

    def start_server(self, wait=False) -> tuple[bool, str]:
        """
        Start Ollama server.

        Args:
            wait: If True, wait for server to be ready

        Returns:
            (success, message)
        """
        if not self.is_installed():
            return False, "Ollama is not installed"

        # Check if already running
        if self.is_running():
            return True, "Ollama server is already running"

        try:
            print("Starting Ollama server...")

            # Start ollama serve
            subprocess.Popen(
                [str(self.ollama_path), "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS
            )

            if wait:
                # Wait for server to respond
                import time
                import requests

                for i in range(20):  # Wait up to 10 seconds
                    time.sleep(0.5)
                    try:
                        response = requests.get("http://localhost:11434/api/tags", timeout=1)
                        if response.status_code == 200:
                            return True, "Ollama server started successfully"
                    except Exception:
                        if i % 4 == 0:
                            print(f"Waiting for Ollama to start... ({i * 0.5:.1f}s)")

                return False, "Timeout waiting for Ollama to start"
            else:
                return True, "Ollama server starting..."

        except Exception as e:
            return False, f"Failed to start server: {e}"

    def pull_model(self, model: str) -> tuple[bool, str]:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name (e.g., "qwen2.5:0.5b")

        Returns:
            (success, message)
        """
        if not self.is_installed():
            return False, "Ollama is not installed"

        try:
            print(f"Pulling model {model}...")

            result = subprocess.run(
                [str(self.ollama_path), "pull", model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )

            if result.returncode == 0:
                return True, f"Model {model} pulled successfully"
            else:
                return False, f"Failed to pull model: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Model pull timed out"
        except Exception as e:
            return False, f"Failed to pull model: {e}"

    def get_installed_models(self) -> list:
        """Get list of installed models."""
        if not self.is_installed():
            return []

        try:
            result = subprocess.run(
                [str(self.ollama_path), "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse output
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models

        except Exception as e:
            print(f"Error getting models: {e}")

        return []


def print_status():
    """Print current Ollama status."""
    installer = OllamaWindowsInstaller()

    print("=" * 50)
    print("Ollama Status Check")
    print("=" * 50)

    if installer.is_installed():
        print(f"[OK] Ollama is installed")
        print(f"     Path: {installer.ollama_path}")
        print(f"     Version: {installer.get_version()}")

        if installer.is_running():
            print("[OK] Ollama server is running")

            models = installer.get_installed_models()
            if models:
                print(f"[OK] Installed models: {', '.join(models)}")
            else:
                print("[WARN] No models installed")
        else:
            print("[WARN] Ollama server is NOT running")
    else:
        print("[FAIL] Ollama is NOT installed")
        print("\nTo install:")
        print("  1. Visit https://ollama.com/download")
        print("  2. Download and run the installer")
        print("  3. Restart this application")

    print("=" * 50)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama Installer for Windows")
    parser.add_argument("--check", action="store_true", help="Check Ollama status")
    parser.add_argument("--install", action="store_true", help="Download and run installer")
    parser.add_argument("--start", action="store_true", help="Start Ollama server")
    parser.add_argument("--pull", metavar="MODEL", help="Pull a model")
    parser.add_argument("--list", action="store_true", help="List installed models")
    parser.add_argument("--wait", action="store_true", help="Wait for operations to complete")

    args = parser.parse_args()

    installer = OllamaWindowsInstaller()

    if args.check:
        print_status()
        return 0

    if args.install:
        # Download installer
        success, msg = installer.download()
        if not success:
            print(f"Error: {msg}")
            return 1

        # Run installer
        success, msg = installer.install(wait=args.wait)
        if success:
            print(f"Success: {msg}")
            return 0
        else:
            print(f"Error: {msg}")
            return 1

    if args.start:
        success, msg = installer.start_server(wait=args.wait)
        if success:
            print(f"Success: {msg}")
            return 0
        else:
            print(f"Error: {msg}")
            return 1

    if args.pull:
        success, msg = installer.pull_model(args.pull)
        if success:
            print(f"Success: {msg}")
            return 0
        else:
            print(f"Error: {msg}")
            return 1

    if args.list:
        models = installer.get_installed_models()
        if models:
            print("Installed models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models installed")
        return 0

    # Default: show status
    print_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
