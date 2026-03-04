"""
Model Manager - Download and manage LLM models.

This module handles downloading, verifying, and managing LLM models
from the catalog, with progress tracking and error handling.
"""

import os
import threading
import hashlib
from pathlib import Path
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
import json

from model_catalog import ModelInfo, get_model_by_id, MODEL_CATALOG


@dataclass
class DownloadProgress:
    """Progress information for a model download."""
    model_id: str
    downloaded_bytes: int
    total_bytes: int
    percentage: float
    speed_mb_s: float
    status: str  # "downloading", "paused", "completed", "error"
    error: Optional[str] = None


@dataclass
class InstalledModel:
    """Information about an installed model."""
    model_id: str
    filepath: str
    file_size_bytes: int
    installed_date: str
    is_active: bool = False


class ModelManager:
    """
    Manager for downloading and maintaining LLM models.

    Features:
    - Download models from catalog with progress tracking
    - Verify downloaded files
    - Track installed models
    - Set active model
    - Clean up unused models
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        try:
            self.models_dir.mkdir(exist_ok=True)
        except PermissionError:
            print(f"[ModelManager] Permission denied creating models directory: {models_dir}")
            # Try using a fallback directory in user's home folder
            fallback_dir = Path.home() / ".localllm_models"
            fallback_dir.mkdir(exist_ok=True)
            self.models_dir = fallback_dir
            print(f"[ModelManager] Using fallback directory: {self.models_dir}")
        except Exception as e:
            print(f"[ModelManager] Error creating models directory: {e}")
            # Use fallback directory
            fallback_dir = Path.home() / ".localllm_models"
            fallback_dir.mkdir(exist_ok=True)
            self.models_dir = fallback_dir
            print(f"[ModelManager] Using fallback directory: {self.models_dir}")

        self._state_file = self.models_dir / "models_state.json"

        self._state_file = self.models_dir / "models_state.json"
        self._installed_models: Dict[str, InstalledModel] = {}
        self._active_model_id: Optional[str] = None
        self._current_download: Optional[DownloadProgress] = None

        # Load state
        self._load_state()

    def _load_state(self):
        """Load saved state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.get("installed", {}).items():
                        self._installed_models[model_id] = InstalledModel(**model_data)
                    self._active_model_id = data.get("active_model")
            except Exception as e:
                print(f"[ModelManager] Error loading state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            data = {
                "installed": {
                    model_id: {
                        "model_id": m.model_id,
                        "filepath": m.filepath,
                        "file_size_bytes": m.file_size_bytes,
                        "installed_date": m.installed_date,
                        "is_active": m.is_active
                    }
                    for model_id, m in self._installed_models.items()
                },
                "active_model": self._active_model_id
            }
            with open(self._state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ModelManager] Error saving state: {e}")

    def _normalize_path(self, path: Path) -> str:
        """
        Convert a path to absolute resolved path.

        Args:
            path: Path object to normalize

        Returns:
            String representation of absolute, resolved path
        """
        return str(path.resolve())

    def get_installed_models(self) -> List[InstalledModel]:
        """Get list of installed models."""
        return list(self._installed_models.values())

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        return model_id in self._installed_models

    def get_active_model(self) -> Optional[InstalledModel]:
        """Get the currently active model."""
        if self._active_model_id and self._active_model_id in self._installed_models:
            return self._installed_models[self._active_model_id]
        return None

    def get_model_path(self, model_id: str) -> Optional[str]:
        """
        Get the file path for an installed model.

        Returns an absolute path if the model exists, None otherwise.
        Will attempt to relocate the model if the stored path is invalid.
        """
        if model_id not in self._installed_models:
            print(f"[ModelManager] Model '{model_id}' not found in installed models")
            return None

        filepath = self._installed_models[model_id].filepath
        path = Path(filepath)

        # If relative, resolve against models_dir
        if not path.is_absolute():
            print(f"[ModelManager] Converting relative path to absolute for {model_id}")
            path = self.models_dir / path

        # Check if file exists at stored location
        if path.exists():
            return self._normalize_path(path)

        # File not found at stored path - try to relocate in models_dir
        print(f"[ModelManager] Model file not found at stored path: {path}, attempting to relocate...")
        for model_file in self.models_dir.glob("*.gguf"):
            if model_file.stem == model_id or model_id in model_file.name.lower():
                # Found it! Update the stored path
                normalized_path = self._normalize_path(model_file)
                self._installed_models[model_id].filepath = normalized_path
                self._save_state()
                print(f"[ModelManager] Relocated model '{model_id}' to: {normalized_path}")
                return normalized_path

        # Model file completely missing
        print(f"[ModelManager] Error: Model file not found for '{model_id}'. Checked: {path}")
        return None

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

        # Update is_active flags
        for model in self._installed_models.values():
            model.is_active = False

        self._installed_models[model_id].is_active = True
        self._active_model_id = model_id
        self._save_state()
        return True

    def scan_for_models(self):
        """Scan models directory for untracked models."""
        try:
            if not self.models_dir.exists():
                return
            for filepath in self.models_dir.glob("*.gguf"):
                # Try to match with catalog
                found = False
                for model in MODEL_CATALOG:
                    if filepath.name == model.filename or model.id in filepath.name.lower():
                        if model.id not in self._installed_models:
                            installed = InstalledModel(
                                model_id=model.id,
                                filepath=self._normalize_path(filepath),
                                file_size_bytes=filepath.stat().st_size,
                                installed_date="unknown",
                                is_active=False
                            )
                            self._installed_models[model.id] = installed
                        found = True
                        break

                # If not in catalog, create entry with filename as ID
                if not found:
                    model_id = filepath.stem
                    if model_id not in self._installed_models:
                        installed = InstalledModel(
                            model_id=model_id,
                            filepath=self._normalize_path(filepath),
                            file_size_bytes=filepath.stat().st_size,
                            installed_date="unknown",
                            is_active=False
                        )
                        self._installed_models[model_id] = installed

                self._save_state()
        except Exception as e:
            print(f"[ModelManager] Error scanning models: {e}")

    def download_model(
        self,
        model_id: str,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ) -> threading.Thread:
        """
        Download a model from the catalog.

        Args:
            model_id: ID of model to download
            on_progress: Callback for progress updates
            on_complete: Callback when download completes (success, message)

        Returns:
            Thread running the download
        """
        model_info = get_model_by_id(model_id)
        if not model_info:
            if on_complete:
                on_complete(False, f"Model {model_id} not found in catalog")
            return None

        def download():
            try:
                dest_path = self.models_dir / model_info.filename
                if not dest_path.parent.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Initialize progress
                total_bytes = model_info.file_size_mb * 1024 * 1024
                self._current_download = DownloadProgress(
                    model_id=model_id,
                    downloaded_bytes=0,
                    total_bytes=total_bytes,
                    percentage=0.0,
                    speed_mb_s=0.0,
                    status="downloading"
                )

                import time
                import urllib.request

                def progress_callback(block_num, block_size, total_size):
                    if self._current_download:
                        self._current_download.downloaded_bytes = block_num * block_size
                        self._current_download.total_bytes = total_size
                        self._current_download.percentage = (
                            (self._current_download.downloaded_bytes / total_size * 100)
                            if total_size > 0 else 0
                        )

                        if on_progress:
                            on_progress(self._current_download)

                # Download with progress
                print(f"[ModelManager] Downloading {model_info.name} from {model_info.download_url}")
                urllib.request.urlretrieve(
                    model_info.download_url,
                    dest_path,
                    reporthook=progress_callback
                )

                # Verify file exists
                if not dest_path.exists():
                    raise FileNotFoundError("Download failed - file not created")

                # Create installed model entry
                import datetime
                installed = InstalledModel(
                    model_id=model_id,
                    filepath=self._normalize_path(dest_path),
                    file_size_bytes=dest_path.stat().st_size,
                    installed_date=datetime.datetime.now().isoformat(),
                    is_active=False
                )
                self._installed_models[model_id] = installed
                self._save_state()

                self._current_download.status = "completed"
                if on_progress:
                    on_progress(self._current_download)

                if on_complete:
                    on_complete(True, f"Model {model_info.name} downloaded successfully!")

            except Exception as e:
                error_msg = str(e)
                print(f"[ModelManager] Download error: {error_msg}")

                if self._current_download:
                    self._current_download.status = "error"
                    self._current_download.error = error_msg
                    if on_progress:
                        on_progress(self._current_download)

                # Cleanup partial download
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except:
                        pass

                if on_complete:
                    on_complete(False, f"Download failed: {error_msg}")

            finally:
                self._current_download = None

        thread = threading.Thread(target=download, daemon=True)
        thread.start()
        return thread

    def delete_model(self, model_id: str) -> bool:
        """
        Delete an installed model.

        Args:
            model_id: ID of model to delete

        Returns:
            True if successful
        """
        if model_id not in self._installed_models:
            return False

        model = self._installed_models[model_id]

        # Don't delete active model
        if model.is_active:
            return False

        try:
            filepath = Path(model.filepath)
            if filepath.exists():
                filepath.unlink()

            del self._installed_models[model_id]
            self._save_state()
            return True
        except Exception as e:
            print(f"[ModelManager] Error deleting model: {e}")
            return False

    def get_download_progress(self) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self._current_download

    def cancel_download(self) -> bool:
        """Cancel current download."""
        # Note: Full cancellation support requires more complex implementation
        if self._current_download:
            self._current_download.status = "paused"
            return True
        return False

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info from catalog."""
        return get_model_by_id(model_id)

    def get_catalog(self) -> List[ModelInfo]:
        """Get full model catalog."""
        return MODEL_CATALOG

    def get_recommended_catalog(self) -> List[ModelInfo]:
        """Get recommended models for display."""
        from model_catalog import get_recommended_models
        return get_recommended_models()
