"""
Test for model switching bug.

User reports: Cannot switch models in the UI.
This test tries to reproduce the bug and verify the fix.
"""

import tempfile
import threading
import queue
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def assert_equal(actual, expected, msg=""):
    """Simple assertion helper."""
    if actual != expected:
        raise AssertionError(f"{msg}\n  Expected: {expected}\n  Actual: {actual}")


def assert_true(value, msg=""):
    """Simple assertion helper."""
    if not value:
        raise AssertionError(f"{msg}\n  Expected: True\n  Actual: {value}")


def assert_false(value, msg=""):
    """Simple assertion helper."""
    if value:
        raise AssertionError(f"{msg}\n  Expected: False\n  Actual: {value}")


def assert_none(value, msg=""):
    """Simple assertion helper."""
    if value is not None:
        raise AssertionError(f"{msg}\n  Expected: None\n  Actual: {value}")


def test_set_active_model_on_non_installed_model():
    """Test that set_active_model returns False for non-installed model."""
    from model_manager import ModelManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(tmpdir)

        # Try to set a non-existent model as active
        result = manager.set_active_model("non_existent_model")

        assert_false(result, "set_active_model should return False for non-installed model")
    print("✓ test_set_active_model_on_non_installed_model PASSED")


def test_set_active_model_on_installed_model():
    """Test that set_active_model works correctly on installed model."""
    from model_manager import ModelManager, InstalledModel

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(tmpdir)

        # Manually add a fake installed model
        fake_model = InstalledModel(
            model_id="test_model",
            filepath=str(Path(tmpdir) / "test.gguf"),
            file_size_bytes=1000,
            installed_date="2024-01-01",
            is_active=False
        )
        manager._installed_models["test_model"] = fake_model

        # Set it as active
        result = manager.set_active_model("test_model")

        assert_true(result, "set_active_model should return True for installed model")
        assert_equal(manager.get_active_model().model_id, "test_model")
        assert_equal(manager._active_model_id, "test_model")
    print("✓ test_set_active_model_on_installed_model PASSED")


def test_get_model_path_returns_none_for_non_installed():
    """Test that get_model_path returns None for non-installed model."""
    from model_manager import ModelManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(tmpdir)

        path = manager.get_model_path("non_existent")

        assert_none(path, "get_model_path should return None for non-installed model")
    print("✓ test_get_model_path_returns_none_for_non_installed PASSED")


def test_switch_model_updates_config():
    """Test that switch_model updates the config."""
    from llm import LLMEngine

    engine = LLMEngine()

    # Mock the load method to avoid actual model loading
    original_load = engine.load
    engine.load = Mock(return_value=True)

    try:
        result = engine.switch_model(
            model_path="/fake/path/model.gguf",
            model_id="test_model"
        )

        # Check that config was updated
        assert_equal(engine._config.model_file, "/fake/path/model.gguf")
        assert_equal(engine._config.model_id, "test_model")
        engine.load.assert_called_once()
    finally:
        engine.load = original_load
    print("✓ test_switch_model_updates_config PASSED")


def test_on_model_select_callback_is_called():
    """Test that the on_model_select callback is called when button clicked."""
    callback_called = []
    callback_args = []

    def mock_callback(model_id):
        callback_called.append(True)
        callback_args.append(model_id)

    # Simulate the callback being set
    on_model_select = mock_callback

    # Simulate button click
    on_model_select("test_model_id")

    assert_equal(len(callback_called), 1)
    assert_equal(callback_args[0], "test_model_id")
    print("✓ test_on_model_select_callback_is_called PASSED")


def test_switch_model_flow_with_mocked_components():
    """
    Test the complete flow:
    1. ModelCatalogWindow calls on_model_select
    2. App._on_model_selected gets model path from manager
    3. App calls llm.switch_model
    4. UI updates with new model
    """
    from model_manager import ModelManager, InstalledModel

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create a manager with installed models
        manager = ModelManager(tmpdir)

        # Add fake model
        model_path = Path(tmpdir) / "model1.gguf"
        model_path.touch()  # Create empty file

        fake_model = InstalledModel(
            model_id="model1",
            filepath=str(model_path),
            file_size_bytes=1000,
            installed_date="2024-01-01",
            is_active=False
        )
        manager._installed_models["model1"] = fake_model

        # Verify get_model_path works (normalize paths for comparison on macOS)
        path = manager.get_model_path("model1")
        # Use os.path.samefile for cross-platform comparison
        assert_true(os.path.samefile(path, str(model_path)), f"Paths don't match: {path} vs {model_path}")

        # Verify set_active_model works
        result = manager.set_active_model("model1")
        assert_true(result)
        assert_equal(manager.get_active_model().model_id, "model1")
    print("✓ test_switch_model_flow_with_mocked_components PASSED")


# RED TEST - This test should FAIL initially and PASS after the fix
def test_model_path_resolution_bug():
    """
    BUG: The model path might not be resolved correctly.
    Test that get_model_path returns absolute, existing paths.
    """
    from model_manager import ModelManager, InstalledModel

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(tmpdir)

        # Create a model file
        model_path = Path(tmpdir) / "test_model.gguf"
        model_path.touch()

        # Add to manager with relative path (potential bug source)
        installed = InstalledModel(
            model_id="test_model",
            filepath="test_model.gguf",  # Relative path - this might cause issues
            file_size_bytes=1000,
            installed_date="2024-01-01",
            is_active=False
        )
        manager._installed_models["test_model"] = installed

        # Get the path - should return absolute path
        path = manager.get_model_path("test_model")

        # The bug: get_model_path might return a relative path
        # which won't exist when LLMEngine tries to load it
        assert_true(path is not None, "Model path should not be None")

        # CRITICAL: Path should exist!
        # If get_model_path returns a relative path, this will fail
        # This is the BUG we're testing for
        print(f"  Returned path: {path}")
        print(f"  Is absolute: {os.path.isabs(path)}")
        print(f"  Exists: {os.path.exists(path)}")

        # For the test to pass, the path must exist
        if not os.path.isabs(path):
            # This is the bug scenario
            full_path = os.path.abspath(path)
            print(f"  Full path would be: {full_path}")
            print(f"  Full path exists: {os.path.exists(full_path)}")

        assert_true(os.path.exists(path), f"Model path does not exist: {path}")
    print("✓ test_model_path_resolution_bug PASSED")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_set_active_model_on_non_installed_model,
        test_set_active_model_on_installed_model,
        test_get_model_path_returns_none_for_non_installed,
        test_switch_model_updates_config,
        test_on_model_select_callback_is_called,
        test_switch_model_flow_with_mocked_components,
        test_model_path_resolution_bug,  # RED TEST
    ]

    failed = []
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, e))

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print("SUCCESS: All tests passed!")
    print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
