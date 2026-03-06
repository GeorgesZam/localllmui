"""
Ollama Model Manager - Pull and manage Ollama models.

This module handles pulling Ollama models from the registry,
with progress tracking and error handling.
"""

import os
import threading
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
import json

from ollama_catalog import OllamaModelInfo, get_ollama_model_by_id, OLLAMA_MODEL_CATALOG, get_installed_ollama_models


@dataclass
class OllamaPullProgress:
    """Progress information for an Ollama model pull."""
    model_id: str
    ollama_name: str
    status: str  # "pulling", "verifying", "completed", "error"
    percentage: float
    downloaded_mb: float
    total_mb: float
    speed_mb_s: float
    error: Optional[str] = None


class OllamaModelManager:
    """
    Manager for pulling and maintaining Ollama models.

    Features:
    - Pull models from Ollama registry with progress tracking
    - Track installed models
    - Set active model
    - Clean up unused models (delete from Ollama)
    """

    def __init__(self):
        """Initialize the Ollama model manager."""
        self._installed_models: Dict[str, str] = {}  # model_id -> ollama_name
        self._active_model_id: Optional[str] = None
        self._current_pull: Optional[OllamaPullProgress] = None

        # Scan for installed models
        self._scan_installed_models()

    def _scan_installed_models(self):
        """Scan Ollama for installed models."""
        try:
            installed = get_installed_ollama_models()

            # Build mapping from our catalog IDs to Ollama names
            for model in OLLAMA_MODEL_CATALOG:
                if model.ollama_name in installed:
                    self._installed_models[model.id] = model.ollama_name

            print(f"[OllamaManager] Found {len(self._installed_models)} catalog models installed")
        except Exception as e:
            print(f"[OllamaManager] Error scanning models: {e}")

    def get_installed_models(self) -> List[str]:
        """Get list of installed model IDs."""
        return list(self._installed_models.keys())

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        return model_id in self._installed_models

    def get_active_model(self) -> Optional[str]:
        """Get the currently active model ID."""
        return self._active_model_id

    def set_active_model(self, model_id: str) -> bool:
        """
        Set the active model.

        Args:
            model_id: ID of model to activate

        Returns:
            True if successful
        """
        if model_id not in self._installed_models:
            return False

        self._active_model_id = model_id
        return True

    def get_model_ollama_name(self, model_id: str) -> Optional[str]:
        """Get the Ollama name for a model ID."""
        model_info = get_ollama_model_by_id(model_id)
        if model_info:
            return model_info.ollama_name
        return None

    def pull_model(
        self,
        model_id: str,
        on_progress: Optional[Callable[[OllamaPullProgress], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ) -> Optional[threading.Thread]:
        """
        Pull a model from Ollama registry.

        Args:
            model_id: ID of model to pull
            on_progress: Callback for progress updates
            on_complete: Callback when pull completes (success, message)

        Returns:
            Thread running the pull, or None if model not found
        """
        model_info = get_ollama_model_by_id(model_id)
        if not model_info:
            if on_complete:
                on_complete(False, f"Model {model_id} not found in catalog")
            return None

        def pull():
            try:
                # Initialize progress
                self._current_pull = OllamaPullProgress(
                    model_id=model_id,
                    ollama_name=model_info.ollama_name,
                    status="pulling",
                    percentage=0.0,
                    downloaded_mb=0.0,
                    total_mb=float(model_info.file_size_mb),
                    speed_mb_s=0.0
                )

                if on_progress:
                    on_progress(self._current_pull)

                # Run ollama pull
                print(f"[OllamaManager] Pulling {model_info.name} ({model_info.ollama_name})")

                process = subprocess.Popen(
                    ["ollama", "pull", model_info.ollama_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Parse output for progress
                import time
                start_time = time.time()

                for line in process.stdout:
                    line = line.strip()
                    if line:
                        print(f"[Ollama] {line}")

                        # Parse progress (ollama shows percentage or digest info)
                        if "%" in line:
                            try:
                                # Extract percentage
                                pct_str = line.split("%")[0].split()[-1]
                                self._current_pull.percentage = float(pct_str)
                                self._current_pull.downloaded_mb = (
                                    self._current_pull.total_mb * self._current_pull.percentage / 100
                                )
                            except (ValueError, IndexError):
                                pass

                        elif "verifying" in line.lower() or "digest" in line.lower():
                            self._current_pull.status = "verifying"

                        # Update speed estimate
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            self._current_pull.speed_mb_s = (
                                self._current_pull.downloaded_mb / elapsed
                            )

                        if on_progress:
                            on_progress(self._current_pull)

                # Wait for process to complete
                process.wait(timeout=600)  # 10 minutes max

                if process.returncode == 0:
                    # Success - add to installed models
                    self._installed_models[model_id] = model_info.ollama_name
                    self._current_pull.status = "completed"
                    self._current_pull.percentage = 100.0

                    if on_progress:
                        on_progress(self._current_pull)

                    if on_complete:
                        on_complete(True, f"Model {model_info.name} pulled successfully!")
                else:
                    raise Exception(f"Pull failed with return code {process.returncode}")

            except subprocess.TimeoutExpired:
                error_msg = "Pull timed out"
                print(f"[OllamaManager] {error_msg}")

                if self._current_pull:
                    self._current_pull.status = "error"
                    self._current_pull.error = error_msg
                    if on_progress:
                        on_progress(self._current_pull)

                if on_complete:
                    on_complete(False, error_msg)

            except Exception as e:
                error_msg = str(e)
                print(f"[OllamaManager] Pull error: {error_msg}")

                if self._current_pull:
                    self._current_pull.status = "error"
                    self._current_pull.error = error_msg
                    if on_progress:
                        on_progress(self._current_pull)

                if on_complete:
                    on_complete(False, f"Pull failed: {error_msg}")

            finally:
                self._current_pull = None

        thread = threading.Thread(target=pull, daemon=True)
        thread.start()
        return thread

    def delete_model(self, model_id: str) -> bool:
        """
        Delete an installed model from Ollama.

        Args:
            model_id: ID of model to delete

        Returns:
            True if successful
        """
        if model_id not in self._installed_models:
            return False

        # Don't delete active model
        if model_id == self._active_model_id:
            return False

        ollama_name = self._installed_models[model_id]

        try:
            result = subprocess.run(
                ["ollama", "rm", ollama_name],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                del self._installed_models[model_id]
                return True
            else:
                print(f"[OllamaManager] Delete failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"[OllamaManager] Error deleting model: {e}")
            return False

    def get_pull_progress(self) -> Optional[OllamaPullProgress]:
        """Get current pull progress."""
        return self._current_pull

    def get_model_info(self, model_id: str) -> Optional[OllamaModelInfo]:
        """Get model info from catalog."""
        return get_ollama_model_by_id(model_id)

    def get_catalog(self) -> List[OllamaModelInfo]:
        """Get full model catalog."""
        return OLLAMA_MODEL_CATALOG

    def get_recommended_catalog(self) -> List[OllamaModelInfo]:
        """Get recommended models for display."""
        from ollama_catalog import get_recommended_ollama_models
        return get_recommended_ollama_models()
