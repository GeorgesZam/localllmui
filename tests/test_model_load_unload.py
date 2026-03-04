"""
Test for model load/unload bug.

User reports: Cannot switch models - gets "Failed to load model from file" error.
This test tries to reproduce the bug where unloading a model doesn't properly
free resources before loading a new one.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_model_unload_creates_clean_state():
    """
    Test that unloading a model properly cleans up state.
    The bug: after del self.llm, some state remains that causes the next load to fail.
    """
    from llm import LLMEngine
    from config import Config

    engine = LLMEngine()

    # Mock the Llama class to avoid actual loading
    mock_llama_class = Mock()
    mock_llama_instance = Mock()
    mock_llama_class.return_value = mock_llama_instance

    with patch('llm.Llama', mock_llama_class):
        # Create a fake model file
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            fake_model_path = f.name
            # Write some dummy data
            f.write(b"FAKE_GGUF")

        try:
            # First load
            result1 = engine.load(model_path=fake_model_path)
            assert result1 is True, "First load should succeed"
            assert engine.llm is not None, "Engine should have llm instance"

            # Check initial state
            first_llm = engine.llm
            first_is_ready = engine.is_ready

            print(f"After first load: llm={first_llm}, is_ready={first_is_ready}")

            # Simulate unloading (what happens in load when switching)
            if engine.llm is not None:
                del engine.llm
                engine.llm = None
                engine.is_ready = False

            print(f"After unload: llm={engine.llm}, is_ready={engine.is_ready}")

            # Verify clean state
            assert engine.llm is None, "llm should be None after unload"
            assert engine.is_ready is False, "is_ready should be False after unload"

            # Second load - this should NOT fail
            result2 = engine.load(model_path=fake_model_path)
            assert result2 is True, "Second load should succeed (this is the bug!)"

            print("✓ test_model_unload_creates_clean_state PASSED")

        finally:
            # Cleanup
            os.unlink(fake_model_path)


def test_switch_model_preserves_config():
    """
    Test that switch_model properly updates config before loading.
    """
    from llm import LLMEngine

    engine = LLMEngine()

    # Store original config values
    original_model_file = engine._config.model_file
    original_model_id = getattr(engine._config, 'model_id', None)

    # Mock load to return True
    with patch.object(engine, 'load', return_value=True) as mock_load:
        result = engine.switch_model(
            model_path="/new/path/model.gguf",
            model_id="new_model"
        )

        # Verify config was updated BEFORE load was called
        assert engine._config.model_file == "/new/path/model.gguf"
        assert engine._config.model_id == "new_model"

        # Verify load was called with correct path
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1] if mock_load.call_args[1] else {}
        assert call_kwargs.get('model_path') == "/new/path/model.gguf"

        print("✓ test_switch_model_preserves_config PASSED")


def test_load_with_invalid_file():
    """Test that load properly handles invalid model files."""
    from llm import LLMEngine

    engine = LLMEngine()

    # Try to load a non-existent file
    result = engine.load(model_path="/nonexistent/path/model.gguf")

    assert result is False, "Load should return False for non-existent file"
    assert hasattr(engine, 'error'), "Engine should store error message"
    assert engine.llm is None, "llm should remain None on failed load"
    assert engine.is_ready is False, "is_ready should remain False on failed load"

    print("✓ test_load_with_invalid_file PASSED")


def test_load_with_corrupted_file():
    """
    Test that load properly handles corrupted GGUF files.
    This simulates the actual bug where a valid file path exists
    but llama_cpp cannot load it.
    """
    from llm import LLMEngine

    engine = LLMEngine()

    # Create a fake .gguf file that's not actually a valid GGUF
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        fake_model_path = f.name
        # Write invalid GGUF data
        f.write(b"This is not a valid GGUF file!")

    try:
        # This should fail with llama_cpp error
        # We need to actually call llama_cpp.Llama to test this
        from llama_cpp import Llama

        try:
            llm = Llama(
                model_path=fake_model_path,
                n_ctx=512,
                n_threads=1,
                n_gpu_layers=0,
                verbose=False
            )
            # If we get here, the file was somehow loaded (unexpected)
            print("Warning: Fake GGUF file was loaded (unexpected)")
        except Exception as e:
            # Expected: should get an error
            print(f"Expected error from corrupted file: {type(e).__name__}")

        # Now test that engine.load handles this gracefully
        result = engine.load(model_path=fake_model_path)
        assert result is False, "Load should return False for corrupted file"
        assert engine.llm is None, "llm should be None after failed load"

        print("✓ test_load_with_corrupted_file PASSED")

    finally:
        os.unlink(fake_model_path)


def run_all_tests():
    """Run all tests."""
    tests = [
        test_model_unload_creates_clean_state,
        test_switch_model_preserves_config,
        test_load_with_invalid_file,
        # test_load_with_corrupted_file,  # Skip for now - requires actual llama_cpp
    ]

    failed = []
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
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
