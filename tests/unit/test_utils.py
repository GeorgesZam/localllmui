"""
Unit tests for utility functions.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import utils
from utils import get_resource_path, get_writable_path, log_message, get_file_hash


class TestGetResourcePath:
    """Test cases for get_resource_path function."""

    def test_returns_absolute_path_when_not_bundled(self):
        """
        AAA Test:
        Arrange: Set up test environment without PyInstaller bundle
        Act: Call get_resource_path with a relative path
        Assert: Verify the returned path is absolute and contains input
        """
        # Arrange
        test_path = "config.json"
        original_meipass = getattr(sys, '_MEIPASS', None)
        if hasattr(sys, '_MEIPASS'):
            delattr(sys, '_MEIPASS')

        try:
            # Act
            result = get_resource_path(test_path)

            # Assert
            # Function returns absolute path from project root
            assert os.path.isabs(result)
            assert test_path in result or result.endswith(test_path)
        finally:
            # Cleanup
            if original_meipass:
                sys._MEIPASS = original_meipass

    def test_returns_bundle_path_when_bundled(self, monkeypatch):
        """
        AAA Test:
        Arrange: Mock PyInstaller bundle environment
        Act: Call get_resource_path with a relative path
        Assert: Verify the returned path uses bundle directory
        """
        # Arrange
        test_path = "models/model.gguf"
        bundle_path = "/tmp/myapp"

        # Clear the cache and mock _MEIPASS
        import utils
        utils.get_resource_path.cache_clear()
        monkeypatch.setattr(sys, '_MEIPASS', bundle_path, raising=False)

        # Act
        result = get_resource_path(test_path)

        # Assert
        assert result == os.path.join(bundle_path, test_path)

    def test_caches_results(self):
        """
        AAA Test:
        Arrange: Set up test environment
        Act: Call get_resource_path multiple times with same input
        Assert: Verify caching works (same result, performance check)
        """
        # Arrange
        test_path = "data/test.txt"
        original_meipass = getattr(sys, '_MEIPASS', None)
        if hasattr(sys, '_MEIPASS'):
            delattr(sys, '_MEIPASS')

        try:
            # Act
            result1 = get_resource_path(test_path)
            result2 = get_resource_path(test_path)

            # Assert
            assert result1 == result2
            # Result is absolute path
            assert os.path.isabs(result1)
            assert test_path in result1
        finally:
            # Cleanup
            if original_meipass:
                sys._MEIPASS = original_meipass


class TestGetWritablePath:
    """Test cases for get_writable_path function."""

    def test_creates_app_data_directory(self, tmp_path, monkeypatch):
        """
        AAA Test:
        Arrange: Mock non-bundled environment with temp directory
        Act: Call get_writable_path
        Assert: Verify directory is created and path is correct
        """
        # Arrange
        test_filename = "test_cache.json"
        original_meipass = getattr(sys, '_MEIPASS', None)
        if hasattr(sys, '_MEIPASS'):
            delattr(sys, '_MEIPASS')

        # Mock the app data dir creation
        monkeypatch.setattr(utils, '_APP_DATA_DIR', str(tmp_path))

        try:
            # Act
            result = get_writable_path(test_filename)

            # Assert
            assert result == os.path.join(str(tmp_path), test_filename)
            assert os.path.exists(tmp_path)
        finally:
            # Cleanup
            monkeypatch.setattr(utils, '_APP_DATA_DIR', None)
            if original_meipass:
                sys._MEIPASS = original_meipass

    def test_returns_cached_directory_on_second_call(self, tmp_path, monkeypatch):
        """
        AAA Test:
        Arrange: Initialize app data directory once
        Act: Call get_writable_path multiple times
        Assert: Verify same directory is used (caching works)
        """
        # Arrange
        monkeypatch.setattr(utils, '_APP_DATA_DIR', str(tmp_path))

        try:
            # Act
            result1 = get_writable_path("file1.txt")
            result2 = get_writable_path("file2.txt")

            # Assert
            assert os.path.dirname(result1) == str(tmp_path)
            assert os.path.dirname(result2) == str(tmp_path)
        finally:
            # Cleanup
            monkeypatch.setattr(utils, '_APP_DATA_DIR', None)


class TestLogMessage:
    """Test cases for log_message function."""

    def test_prints_message_with_prefix(self, capsys):
        """
        AAA Test:
        Arrange: Define prefix and message
        Act: Call log_message
        Assert: Verify output contains prefix and message
        """
        # Arrange
        prefix = "INFO"
        message = "Test message"

        # Act
        log_message(prefix, message)

        # Assert
        captured = capsys.readouterr()
        assert f"[{prefix}] {message}" in captured.out

    def test_calls_callback_if_provided(self):
        """
        AAA Test:
        Arrange: Create a mock callback function
        Act: Call log_message with callback
        Assert: Verify callback was invoked with correct message
        """
        # Arrange
        callback_called = []
        def test_callback(msg):
            callback_called.append(msg)

        prefix = "DEBUG"
        message = "Callback test"

        # Act
        log_message(prefix, message, callback=test_callback)

        # Assert
        assert len(callback_called) == 1
        assert callback_called[0] == message

    def test_works_without_callback(self, capsys):
        """
        AAA Test:
        Arrange: Set up test without callback
        Act: Call log_message without callback
        Assert: Verify message is still printed
        """
        # Arrange
        prefix = "ERROR"
        message = "No callback test"

        # Act
        log_message(prefix, message)

        # Assert
        captured = capsys.readouterr()
        assert f"[{prefix}] {message}" in captured.out


class TestGetFileHash:
    """Test cases for get_file_hash function."""

    def test_returns_hash_for_existing_file(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a temporary test file
        Act: Call get_file_hash
        Assert: Verify hash is returned and is consistent
        """
        # Arrange
        test_file = tmp_path / "test.txt"
        test_content = "Test content for hashing"
        test_file.write_text(test_content)

        # Act
        hash1 = get_file_hash(str(test_file))
        hash2 = get_file_hash(str(test_file))

        # Assert
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # MD5 hash truncated to 16 chars
        assert hash1 == hash2  # Same file should produce same hash

    def test_different_files_produce_different_hashes(self, tmp_path):
        """
        AAA Test:
        Arrange: Create two different test files
        Act: Call get_file_hash on both
        Assert: Verify hashes are different
        """
        # Arrange
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        # Act
        hash1 = get_file_hash(str(file1))
        hash2 = get_file_hash(str(file2))

        # Assert
        assert hash1 != hash2

    def test_hash_changes_when_file_is_modified(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a file and get its hash
        Act: Modify the file and get new hash
        Assert: Verify hash changes
        """
        # Arrange
        test_file = tmp_path / "mutable.txt"
        test_file.write_text("Original content")
        original_hash = get_file_hash(str(test_file))

        # Act
        import time
        time.sleep(0.01)  # Ensure mtime is different
        test_file.write_text("Modified content")
        new_hash = get_file_hash(str(test_file))

        # Assert
        assert original_hash != new_hash
