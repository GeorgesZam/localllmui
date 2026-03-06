"""
Unit tests for model manager module.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import json
import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from model_manager import (
    DownloadProgress,
    InstalledModel,
    ModelManager,
)


class TestDownloadProgress:
    """Test cases for DownloadProgress dataclass."""

    def test_initializes_with_all_fields(self):
        """
        AAA Test:
        Arrange: Define progress data
        Act: Create DownloadProgress
        Assert: Verify all fields are set correctly
        """
        # Arrange
        progress_data = {
            'model_id': 'test_model',
            'downloaded_bytes': 1024,
            'total_bytes': 2048,
            'percentage': 50.0,
            'speed_mb_s': 1.5,
            'status': 'downloading',
            'error': None
        }

        # Act
        progress = DownloadProgress(**progress_data)

        # Assert
        assert progress.model_id == 'test_model'
        assert progress.downloaded_bytes == 1024
        assert progress.total_bytes == 2048
        assert progress.percentage == 50.0
        assert progress.speed_mb_s == 1.5
        assert progress.status == 'downloading'
        assert progress.error is None

    def test_initializes_with_error(self):
        """
        AAA Test:
        Arrange: Define progress with error
        Act: Create DownloadProgress
        Assert: Verify error is stored
        """
        # Arrange
        progress_data = {
            'model_id': 'test',
            'downloaded_bytes': 0,
            'total_bytes': 1000,
            'percentage': 0.0,
            'speed_mb_s': 0.0,
            'status': 'error',
            'error': 'Network error'
        }

        # Act
        progress = DownloadProgress(**progress_data)

        # Assert
        assert progress.status == 'error'
        assert progress.error == 'Network error'


class TestInstalledModel:
    """Test cases for InstalledModel dataclass."""

    def test_initializes_with_all_fields(self):
        """
        AAA Test:
        Arrange: Define installed model data
        Act: Create InstalledModel
        Assert: Verify all fields are set correctly
        """
        # Arrange
        model_data = {
            'model_id': 'llama-2-7b',
            'filepath': '/models/llama.gguf',
            'file_size_bytes': 4096000000,
            'installed_date': '2025-01-01T12:00:00',
            'is_active': True
        }

        # Act
        model = InstalledModel(**model_data)

        # Assert
        assert model.model_id == 'llama-2-7b'
        assert model.filepath == '/models/llama.gguf'
        assert model.file_size_bytes == 4096000000
        assert model.installed_date == '2025-01-01T12:00:00'
        assert model.is_active is True

    def test_defaults_to_inactive(self):
        """
        AAA Test:
        Arrange: Create model without is_active
        Act: Create InstalledModel
        Assert: Verify defaults to inactive
        """
        # Arrange
        model_data = {
            'model_id': 'test',
            'filepath': '/path/to/file.gguf',
            'file_size_bytes': 1000,
            'installed_date': '2025-01-01'
        }

        # Act
        model = InstalledModel(**model_data)

        # Assert
        assert model.is_active is False


class TestModelManagerInit:
    """Test cases for ModelManager initialization."""

    def test_initializes_with_directory(self, tmp_path):
        """
        AAA Test:
        Arrange: Create temp directory
        Act: Initialize ModelManager
        Assert: Verify directory is created
        """
        # Arrange
        models_dir = tmp_path / 'models'

        # Act
        manager = ModelManager(str(models_dir))

        # Assert
        assert manager.models_dir.exists()
        assert manager.models_dir.is_dir()

    def test_uses_fallback_on_permission_error(self):
        """
        AAA Test:
        Arrange: Mock permission error on initial directory creation
        Act: Initialize ModelManager
        Assert: Verify fallback directory is used or error is handled
        """
        # Arrange
        original_mkdir = Path.mkdir
        call_count = [0]

        def selective_mkdir(self, *args, **kwargs):
            call_count[0] += 1
            # Fail first attempt (models_dir), succeed second (fallback)
            if call_count[0] == 1:
                raise PermissionError("Access denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, 'mkdir', selective_mkdir):
            # Act
            manager = ModelManager('/invalid/path')

            # Assert
            # Should have tried to create the fallback directory
            assert call_count[0] >= 1
            assert manager.models_dir is not None

    def test_loads_existing_state(self, tmp_path):
        """
        AAA Test:
        Arrange: Create existing state file
        Act: Initialize ModelManager
        Assert: Verify state is loaded
        """
        # Arrange
        models_dir = tmp_path / 'models'
        models_dir.mkdir()

        state_file = models_dir / 'models_state.json'
        state_data = {
            'installed': {
                'test_model': {
                    'model_id': 'test_model',
                    'filepath': '/models/test.gguf',
                    'file_size_bytes': 1000,
                    'installed_date': '2025-01-01',
                    'is_active': False
                }
            },
            'active_model': None
        }
        state_file.write_text(json.dumps(state_data))

        # Act
        manager = ModelManager(str(models_dir))

        # Assert
        assert 'test_model' in manager._installed_models
        assert manager._installed_models['test_model'].model_id == 'test_model'


class TestModelManagerGetInstalledModels:
    """Test cases for getting installed models."""

    def test_returns_empty_list_when_no_models(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without models
        Act: Get installed models
        Assert: Verify empty list
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        models = manager.get_installed_models()

        # Assert
        assert models == []

    def test_returns_list_of_installed_models(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with installed models
        Act: Get installed models
        Assert: Verify all models are returned
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._installed_models = {
            'model1': InstalledModel(
                model_id='model1',
                filepath='/path1.gguf',
                file_size_bytes=1000,
                installed_date='2025-01-01'
            ),
            'model2': InstalledModel(
                model_id='model2',
                filepath='/path2.gguf',
                file_size_bytes=2000,
                installed_date='2025-01-02'
            )
        }

        # Act
        models = manager.get_installed_models()

        # Assert
        assert len(models) == 2
        model_ids = [m.model_id for m in models]
        assert 'model1' in model_ids
        assert 'model2' in model_ids


class TestModelManagerIsModelInstalled:
    """Test cases for checking if model is installed."""

    def test_returns_true_when_installed(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with installed model
        Act: Check if model is installed
        Assert: Verify returns True
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._installed_models['test'] = InstalledModel(
            model_id='test',
            filepath='/test.gguf',
            file_size_bytes=1000,
            installed_date='2025-01-01'
        )

        # Act
        result = manager.is_model_installed('test')

        # Assert
        assert result is True

    def test_returns_false_when_not_installed(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without the model
        Act: Check if model is installed
        Assert: Verify returns False
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        result = manager.is_model_installed('nonexistent')

        # Assert
        assert result is False


class TestModelManagerGetActiveModel:
    """Test cases for getting active model."""

    def test_returns_active_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with active model
        Act: Get active model
        Assert: Verify correct model is returned
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._installed_models = {
            'model1': InstalledModel(
                model_id='model1',
                filepath='/path1.gguf',
                file_size_bytes=1000,
                installed_date='2025-01-01',
                is_active=False
            ),
            'model2': InstalledModel(
                model_id='model2',
                filepath='/path2.gguf',
                file_size_bytes=2000,
                installed_date='2025-01-02',
                is_active=True
            )
        }
        manager._active_model_id = 'model2'

        # Act
        active = manager.get_active_model()

        # Assert
        assert active is not None
        assert active.model_id == 'model2'
        assert active.is_active is True

    def test_returns_none_when_no_active_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without active model
        Act: Get active model
        Assert: Verify returns None
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._active_model_id = None

        # Act
        active = manager.get_active_model()

        # Assert
        assert active is None


class TestModelManagerSetActiveSheet:
    """Test cases for setting active model."""

    def test_sets_active_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with installed models
        Act: Set active model
        Assert: Verify model is activated
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._installed_models = {
            'model1': InstalledModel(
                model_id='model1',
                filepath='/path1.gguf',
                file_size_bytes=1000,
                installed_date='2025-01-01',
                is_active=False
            ),
            'model2': InstalledModel(
                model_id='model2',
                filepath='/path2.gguf',
                file_size_bytes=2000,
                installed_date='2025-01-02',
                is_active=False
            )
        }

        # Act
        result = manager.set_active_model('model2')

        # Assert
        assert result is True
        assert manager._active_model_id == 'model2'
        assert manager._installed_models['model2'].is_active is True
        assert manager._installed_models['model1'].is_active is False

    def test_returns_false_for_nonexistent_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager
        Act: Try to set non-existent model as active
        Assert: Verify returns False
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        result = manager.set_active_model('nonexistent')

        # Assert
        assert result is False


class TestModelManagerGetModelPath:
    """Test cases for getting model path."""

    def test_returns_none_for_nonexistent_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without model
        Act: Get model path
        Assert: Verify returns None
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        path = manager.get_model_path('nonexistent')

        # Assert
        assert path is None

    def test_returns_absolute_path_for_existing_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with existing model file
        Act: Get model path
        Assert: Verify absolute path is returned
        """
        # Arrange
        model_file = tmp_path / 'model.gguf'
        model_file.write_text('fake model')

        manager = ModelManager(str(tmp_path))
        manager._installed_models['test'] = InstalledModel(
            model_id='test',
            filepath=str(model_file),
            file_size_bytes=100,
            installed_date='2025-01-01'
        )

        # Act
        path = manager.get_model_path('test')

        # Assert
        assert path is not None
        assert Path(path).is_absolute()

    def test_relocates_missing_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with missing model file
        Act: Get model path with relocated file
        Assert: Verify file is relocated and path returned
        """
        # Arrange
        # Create model file with different name
        model_file = tmp_path / 'test_model.gguf'
        model_file.write_text('fake model')

        manager = ModelManager(str(tmp_path))
        manager._installed_models['test_model'] = InstalledModel(
            model_id='test_model',
            filepath='/wrong/path/test_model.gguf',  # Wrong path
            file_size_bytes=100,
            installed_date='2025-01-01'
        )

        # Act
        path = manager.get_model_path('test_model')

        # Assert
        assert path is not None
        # Path should be updated to actual location
        assert 'test_model' in path.lower()


class TestModelManagerDeleteModel:
    """Test cases for deleting models."""

    def test_deletes_installed_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with inactive model
        Act: Delete model
        Assert: Verify model is removed
        """
        # Arrange
        model_file = tmp_path / 'model.gguf'
        model_file.write_text('fake model')

        manager = ModelManager(str(tmp_path))
        manager._installed_models['test'] = InstalledModel(
            model_id='test',
            filepath=str(model_file),
            file_size_bytes=100,
            installed_date='2025-01-01',
            is_active=False
        )

        # Act
        result = manager.delete_model('test')

        # Assert
        assert result is True
        assert 'test' not in manager._installed_models
        assert not model_file.exists()

    def test_returns_false_for_active_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with active model
        Act: Try to delete active model
        Assert: Verify returns False
        """
        # Arrange
        model_file = tmp_path / 'model.gguf'
        model_file.write_text('fake model')

        manager = ModelManager(str(tmp_path))
        manager._installed_models['test'] = InstalledModel(
            model_id='test',
            filepath=str(model_file),
            file_size_bytes=100,
            installed_date='2025-01-01',
            is_active=True
        )

        # Act
        result = manager.delete_model('test')

        # Assert
        assert result is False
        assert model_file.exists()  # File should not be deleted

    def test_returns_false_for_nonexistent_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager
        Act: Try to delete non-existent model
        Assert: Verify returns False
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        result = manager.delete_model('nonexistent')

        # Assert
        assert result is False


class TestModelManagerDownloadProgress:
    """Test cases for download progress tracking."""

    def test_returns_none_when_no_download(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without active download
        Act: Get download progress
        Assert: Verify returns None
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        progress = manager.get_download_progress()

        # Assert
        assert progress is None

    def test_cancels_active_download(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with active download
        Act: Cancel download
        Assert: Verify download is paused
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._current_download = DownloadProgress(
            model_id='test',
            downloaded_bytes=100,
            total_bytes=1000,
            percentage=10.0,
            speed_mb_s=1.0,
            status='downloading'
        )

        # Act
        result = manager.cancel_download()

        # Assert
        assert result is True
        assert manager._current_download.status == 'paused'

    def test_returns_false_when_no_download_to_cancel(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager without active download
        Act: Try to cancel download
        Assert: Verify returns False
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        result = manager.cancel_download()

        # Assert
        assert result is False


class TestModelManagerSaveLoadState:
    """Test cases for state persistence."""

    def test_saves_state_to_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager with installed model
        Act: Save state
        Assert: Verify state file is created
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        manager._installed_models['test'] = InstalledModel(
            model_id='test',
            filepath='/test.gguf',
            file_size_bytes=1000,
            installed_date='2025-01-01',
            is_active=True
        )
        manager._active_model_id = 'test'

        # Act
        manager._save_state()

        # Assert
        state_file = tmp_path / 'models_state.json'
        assert state_file.exists()

        with open(state_file, 'r') as f:
            data = json.load(f)
        assert 'test' in data['installed']
        assert data['active_model'] == 'test'

    def test_loads_state_from_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create state file with data
        Act: Create manager (loads state)
        Assert: Verify state is restored
        """
        # Arrange
        state_file = tmp_path / 'models_state.json'
        state_data = {
            'installed': {
                'test': {
                    'model_id': 'test',
                    'filepath': '/test.gguf',
                    'file_size_bytes': 1000,
                    'installed_date': '2025-01-01',
                    'is_active': True
                }
            },
            'active_model': 'test'
        }
        state_file.write_text(json.dumps(state_data))

        # Act
        manager = ModelManager(str(tmp_path))

        # Assert
        assert 'test' in manager._installed_models
        assert manager._active_model_id == 'test'
        assert manager._installed_models['test'].is_active is True


class TestModelManagerGetModelInfo:
    """Test cases for getting model info from catalog."""

    def test_returns_none_for_nonexistent_model(self, tmp_path):
        """
        AAA Test:
        Arrange: Create manager
        Act: Get info for non-existent model
        Assert: Verify returns None
        """
        # Arrange
        manager = ModelManager(str(tmp_path))

        # Act
        info = manager.get_model_info('nonexistent_model_xyz')

        # Assert
        # Will return None if model not in catalog
        # (assuming get_model_by_id returns None for unknown IDs)
        assert info is None


class TestModelManagerNormalizePath:
    """Test cases for path normalization."""

    def test_converts_to_absolute_path(self, tmp_path):
        """
        AAA Test:
        Arrange: Create relative path
        Act: Normalize path
        Assert: Verify absolute path is returned
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        relative_path = Path('models/test.gguf')

        # Act
        normalized = manager._normalize_path(relative_path)

        # Assert
        assert Path(normalized).is_absolute()

    def test_handles_existing_absolute_path(self, tmp_path):
        """
        AAA Test:
        Arrange: Create absolute path
        Act: Normalize path
        Assert: Verify path remains absolute
        """
        # Arrange
        manager = ModelManager(str(tmp_path))
        absolute_path = tmp_path / 'test.gguf'

        # Act
        normalized = manager._normalize_path(absolute_path)

        # Assert
        assert Path(normalized).is_absolute()
