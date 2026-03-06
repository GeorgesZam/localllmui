"""
Ollama Engine - LLM backend using Ollama instead of llama_cpp.

This module provides the same interface as llm.py but uses Ollama
which is better optimized and doesn't require bundling models.
"""

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import requests

try:
    from config import ConfigManager
    from patterns import Event, Observable, SingletonMeta, StateEvent
    from rag import RAG
except ImportError:
    # Fallback for when run standalone
    class SingletonMeta(type):
        _instances = {}

        def __call__(cls, *args, **kwargs):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]

    class Observable:
        def __init__(self):
            self._observers = []

        def attach(self, observer):
            self._observers.append(observer)

        def notify(self, event):
            for observer in self._observers:
                observer.update(event)

        def update(self, event):
            pass

    class Event:
        def __init__(self, event_type, data):
            self.type = event_type
            self.data = data

    class StateEvent:
        LOADING = "loading"
        READY = "ready"
        ERROR = "error"

        @staticmethod
        def create(event_type, data):
            return {"type": event_type, "data": data}

    class ConfigManager:
        CONTEXT_SIZE = 8192
        MAX_TOKENS = 2048
        TEMPERATURE = 0.7
        TOP_P = 0.9
        REPEAT_PENALTY = 1.1
        STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
        SYSTEM_PROMPT = "You are a helpful AI assistant."
        CODE_EXECUTION_ENABLED = False
        CODE_EXECUTION_AUTO_DETECT = False
        CODE_EXECUTION_TIMEOUT = 30
        CODE_EXECUTION_MAX_MEMORY_MB = 512
        RAG_ENABLED = True
        RAG_SHOW_SOURCES = True

        @staticmethod
        def get_instance():
            return ConfigManager()


@dataclass
class GenerationStats:
    """Statistics for a generation run."""

    tokens_generated: int = 0
    time_elapsed: float = 0.0
    tokens_per_second: float = 0.0
    has_code_execution: bool = False
    rag_results_used: int = 0


@dataclass
class OllamaModel:
    """Information about an Ollama model."""

    name: str
    size: int
    quantization: str
    family: str
    modified_at: str


class OllamaInstaller:
    """Handles Ollama installation on Windows."""

    OLLAMA_VERSION = "0.5.7"
    OLLAMA_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/OllamaSetup.exe"

    @staticmethod
    def is_ollama_installed() -> bool:
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_ollama_running() -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def start_ollama() -> Tuple[bool, str]:
        """Start Ollama server (requires Ollama to be installed)."""
        try:
            # Start ollama serve in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            # Wait for server to start
            for _ in range(20):  # Wait up to 10 seconds
                time.sleep(0.5)
                if OllamaInstaller.is_ollama_running():
                    return True, "Ollama server started"
            return False, "Timeout waiting for Ollama to start"
        except Exception as e:
            return False, f"Failed to start Ollama: {e}"

    @staticmethod
    def get_download_path() -> str:
        """Get path for Ollama installer download."""
        temp_dir = Path(os.environ.get("TEMP", "/tmp"))
        return str(temp_dir / "OllamaSetup.exe")

    @staticmethod
    def download_installer(
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, str]:
        """Download Ollama installer."""
        import urllib.request

        installer_path = OllamaInstaller.get_download_path()

        def report_progress(block_num, block_size, total_size):
            if on_progress and total_size > 0:
                downloaded = block_num * block_size
                on_progress(downloaded, total_size)

        try:
            print(f"[Ollama] Downloading from {OllamaInstaller.OLLAMA_URL}")
            urllib.request.urlretrieve(
                OllamaInstaller.OLLAMA_URL, installer_path, reporthook=report_progress
            )
            return True, installer_path
        except Exception as e:
            return False, f"Download failed: {e}"

    @staticmethod
    def run_installer(installer_path: str) -> Tuple[bool, str]:
        """Run Ollama installer (synchronous - waits for completion)."""
        try:
            print(f"[Ollama] Running installer: {installer_path}")
            result = subprocess.run(
                [installer_path], shell=True, timeout=300  # 5 minutes max
            )
            return True, "Installer completed"
        except subprocess.TimeoutExpired:
            return False, "Installer timed out"
        except Exception as e:
            return False, f"Installer failed: {e}"


class OllamaEngine(Observable, metaclass=SingletonMeta):
    """
    LLM engine using Ollama instead of llama_cpp.

    Benefits:
    - No need to bundle large model files
    - Better GPU optimization
    - Smaller executable size
    - Easy model switching via CLI

    Usage:
        engine = OllamaEngine()
        engine.load()
        for token in engine.generate("Hello"):
            print(token, end='')
    """

    DEFAULT_MODEL = "qwen2.5:0.5b"  # Small, fast model
    ALTERNATIVE_MODELS = [
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "phi3:mini",
        "gemma2:2b",
        "llama3.2:1b",
    ]

    def __init__(self):
        # Only initialize once (Singleton pattern)
        if hasattr(self, "_initialized"):
            return

        # Initialize Observable base class
        super().__init__()

        # Configuration
        self._config = ConfigManager.get_instance()

        # Core components
        self.rag: Optional[RAG] = None

        # Ollama connection
        self.base_url = "http://localhost:11434"
        self.model = self.DEFAULT_MODEL
        self._client = None

        # State
        self.is_ready = False
        self.error: Optional[str] = None
        self._max_history = 10  # Ollama handles history better
        self.history: list = []

        # Code execution (disabled by default for Ollama)
        self.code_executor = None
        self.code_execution_enabled = False

        # Skills integration
        self.skills_executor = None
        self._enabled_skills_content = ""

        # Statistics
        self._current_stats = GenerationStats()

        # Stop flag for generation cancellation
        self._stop_requested = threading.Event()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "OllamaEngine":
        """Get the singleton Ollama engine instance."""
        return cls()

    def stop_generation(self):
        """Request to stop the current generation."""
        self._stop_requested.set()

    def reset_stop_flag(self):
        """Reset stop flag for new generation."""
        self._stop_requested.clear()

    def _check_ollama(self) -> Tuple[bool, str]:
        """Check if Ollama is installed and running."""
        if not OllamaInstaller.is_ollama_installed():
            return False, "Ollama is not installed"

        if not OllamaInstaller.is_ollama_running():
            return False, "Ollama is not running"

        return True, "OK"

    def _ensure_ollama_running(
        self, on_progress: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Ensure Ollama is installed and running."""

        def log(msg: str) -> None:
            print(f"[Ollama] {msg}")
            if on_progress:
                on_progress(msg)

        # Check if running
        installed, status = self._check_ollama()

        if not installed:
            log("Ollama is not installed")
            log("Please install Ollama from: https://ollama.com/download")
            return False

        if not OllamaInstaller.is_ollama_running():
            log("Starting Ollama server...")
            success, msg = OllamaInstaller.start_ollama()
            if not success:
                log(f"Failed to start: {msg}")
                return False
            log("Ollama server started")

        return True

    def _pull_model(
        self, model: str, on_progress: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Pull a model from Ollama registry."""

        def log(msg: str) -> None:
            print(f"[Ollama] {msg}")
            if on_progress:
                on_progress(msg)

        try:
            log(f"Pulling model {model}...")

            # Use subprocess to show progress
            process = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    log(line)

            process.wait(timeout=300)  # 5 minutes max

            if process.returncode == 0:
                log(f"Model {model} pulled successfully")
                return True
            else:
                log(f"Failed to pull model: return code {process.returncode}")
                return False

        except subprocess.TimeoutExpired:
            log("Model pull timed out")
            return False
        except Exception as e:
            log(f"Error pulling model: {e}")
            return False

    def _get_available_models(self) -> list:
        """Get list of available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except requests.RequestException:
            pass
        return []

    def _test_model(self, model: str) -> bool:
        """Test if a model is working."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "test", "stream": False},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def load(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """
        Initialize Ollama engine and ensure model is available.

        Args:
            on_progress: Optional callback for progress updates
            model_path: Optional model name (overrides default)

        Returns:
            True if loading successful
        """

        def log(msg: str) -> None:
            print(f"[Ollama] {msg}")
            if on_progress:
                on_progress(msg)
            self.notify(StateEvent.create(StateEvent.LOADING, {"message": msg}))

        try:
            # Ensure Ollama is running
            if not self._ensure_ollama_running(on_progress):
                self.error = "Ollama is not available. Please install Ollama from https://ollama.com/download"
                self.notify(StateEvent.create(StateEvent.ERROR, {"error": self.error}))
                return False

            # Set model
            if model_path:
                self.model = model_path

            log(f"Using model: {self.model}")

            # Check if model exists
            models = self._get_available_models()
            model_names = [m.split(":")[0] for m in models]

            # Check if our model or its family is available
            model_family = self.model.split(":")[0]
            if model_family not in model_names:
                log(f"Model {self.model} not found locally")
                log(f"Pulling {self.model} from Ollama registry...")
                if not self._pull_model(self.model, on_progress):
                    # Try alternative models
                    for alt in self.ALTERNATIVE_MODELS:
                        alt_family = alt.split(":")[0]
                        if alt_family in model_names:
                            log(f"Using alternative model: {alt}")
                            self.model = alt
                            break
                        elif alt_family != model_family:
                            log(f"Trying to pull {alt}...")
                            if self._pull_model(alt, on_progress):
                                self.model = alt
                                break

            # Test the model
            log("Testing model...")
            if not self._test_model(self.model):
                log(f"Warning: Model {self.model} may not work properly")

            # Initialize RAG if not already initialized
            if self.rag is None:
                log("Initializing RAG...")
                self.rag = RAG()
                self.rag.initialize(log)
                log(f"RAG initialized with {len(self.rag.documents)} documents")
            else:
                log("RAG already initialized")

            # Clear history
            self.history = []
            self._stop_requested.clear()
            self.is_ready = True
            log("Ready!")

            self.notify(StateEvent.create(StateEvent.READY, {"model": self.model}))

            return True

        except Exception as e:
            self.error = str(e)
            import traceback

            traceback.print_exc()
            log(f"Error: {self.error}")

            self.notify(StateEvent.create(StateEvent.ERROR, {"error": self.error}))
            return False

    def set_skills_executor(self, skills_executor):
        """Set the skills executor for prompt enhancement."""
        self.skills_executor = skills_executor

    def update_skills_content(self, skills_manager):
        """Update the skills content from the skills manager."""
        if self.skills_executor and skills_manager:
            base_prompt = self._config.SYSTEM_PROMPT
            enhanced_prompt, skill_names = self.skills_executor.apply_skills_to_prompt(
                "", base_prompt
            )
            self._enabled_skills_content = enhanced_prompt.replace(
                base_prompt, ""
            ).strip()
        else:
            self._enabled_skills_content = ""

    def _build_prompt(self, message: str, rag_context: str = "") -> str:
        """Build prompt for Ollama /api/generate endpoint."""
        # Build system prompt
        base_system = self._config.SYSTEM_PROMPT
        if self._enabled_skills_content:
            system = f"{base_system}\n\n{self._enabled_skills_content}"
        else:
            system = base_system

        # Add RAG context if available
        if rag_context:
            system = f"""{system}

=== CONTEXT ===
{rag_context}
=== END ===

Answer based on context. If not found, say so."""

        # Build conversation prompt
        prompt = f"System: {system}\n\n"

        # Add history
        for h in self.history[-self._max_history :]:
            prompt += f"User: {h['user']}\n"
            prompt += f"Assistant: {h['assistant']}\n"

        # Add current message
        prompt += f"User: {message}\n"
        prompt += "Assistant:"

        return prompt

    def generate(
        self, message: str, allowed_document_sources: list = None
    ) -> Iterator[str]:
        """
        Generate response using Ollama.

        Args:
            message: User message
            allowed_document_sources: Optional list of document filenames for RAG filtering

        Yields:
            Generated tokens
        """
        if not self.is_ready:
            yield "Error: Model not ready"
            return

        self._current_stats = GenerationStats()

        # RAG generation
        rag_context = ""
        sources = []

        if self._config.RAG_ENABLED and self.rag and self.rag.documents:
            rag_context, sources = self.rag.search(
                message, allowed_sources=allowed_document_sources
            )
            self._current_stats.rag_results_used = len(sources)

        prompt = self._build_prompt(message, rag_context)

        full_response = ""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": self._config.MAX_TOKENS,
                        "temperature": self._config.TEMPERATURE,
                        "top_p": self._config.TOP_P,
                        "repeat_penalty": self._config.REPEAT_PENALTY,
                    },
                },
                stream=True,
                timeout=120,
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if self._stop_requested.is_set():
                    break

                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        token = data["response"]
                        full_response += token
                        self._current_stats.tokens_generated += 1
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        except requests.RequestException as e:
            print(f"[Ollama] Generation error: {e}")
            yield f"\n[Error: {e}]"

        elapsed = time.time() - start_time
        self._current_stats.time_elapsed = elapsed
        if elapsed > 0:
            self._current_stats.tokens_per_second = (
                self._current_stats.tokens_generated / elapsed
            )

        # Add sources if RAG was used
        if self._config.RAG_SHOW_SOURCES and sources:
            sources_text = self.rag.format_sources_for_display()
            yield sources_text
            full_response += sources_text

        # Store in history
        if full_response.strip():
            clean = full_response.split("Sources:")[0].strip()
            self.history.append({"user": message, "assistant": clean})

            if len(self.history) > self._max_history * 2:
                self.history = self.history[-self._max_history :]

        # Notify completion
        self.notify(
            Event(
                "generation_complete",
                {
                    "tokens": self._current_stats.tokens_generated,
                    "has_code": self._current_stats.has_code_execution,
                    "rag_results": self._current_stats.rag_results_used,
                },
            )
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        print("[Ollama] History cleared")
        self.notify(Event("history_cleared", {}))

    def get_stats(self) -> GenerationStats:
        """Get statistics from the last generation."""
        return self._current_stats

    def reset(self) -> None:
        """Reset the engine state."""
        self.history = []
        self.error = None
        self._current_stats = GenerationStats()
        self.notify(Event("engine_reset", {}))

    def switch_model(
        self,
        model_path: str,
        model_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Switch to a different model.

        Args:
            model_path: Model name (e.g., "qwen2.5:1.5b")
            model_id: ID of the model
            on_progress: Optional callback for progress updates

        Returns:
            True if successful
        """
        self.model = model_path
        return self.load(on_progress=on_progress, model_path=model_path)

    def get_current_model_id(self) -> str:
        """Get the current model ID."""
        return self.model


def get_ollama_engine() -> OllamaEngine:
    """Get the global Ollama engine instance."""
    return OllamaEngine.get_instance()
