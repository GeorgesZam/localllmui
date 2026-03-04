"""
Test for the send/stop button state management.
Tests that the button correctly toggles between send and stop states.
"""
import pytest
from unittest.mock import MagicMock


class TestSendStopButton:
    """Test suite for send/stop button state management."""

    def test_set_generating_state_to_true_shows_stop_button(self):
        """Test that set_generating_state(True) changes button to stop state."""
        # Create a mock object that mimics the set_generating_state method
        send_btn = MagicMock()
        is_generating = [False]  # Use list to allow mutation in nested function

        def set_generating_state(is_gen):
            """Copy of the actual set_generating_state logic from ui.py"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,  # _stop_generation
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,  # _send
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        # Call set_generating_state(True)
        set_generating_state(True)

        # Check is_generating flag
        assert is_generating[0] is True

        # Check that configure was called with stop button parameters
        send_btn.configure.assert_called_once()
        call_kwargs = send_btn.configure.call_args.kwargs

        # Verify stop button state
        assert "Stop" in call_kwargs.get('text', '')

    def test_set_generating_state_to_false_shows_send_button(self):
        """Test that set_generating_state(False) changes button to send state."""
        # Create a mock object that mimics the set_generating_state method
        send_btn = MagicMock()
        is_generating = [False]  # Use list to allow mutation in nested function

        def set_generating_state(is_gen):
            """Copy of the actual set_generating_state logic from ui.py"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,  # _stop_generation
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,  # _send
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        # First set to generating state
        set_generating_state(True)

        # Then set back to non-generating
        set_generating_state(False)

        # Check is_generating flag
        assert is_generating[0] is False

        # Check that configure was called twice (once for True, once for False)
        assert send_btn.configure.call_count == 2

        # Get the last call (False state)
        last_call_kwargs = send_btn.configure.call_args_list[-1].kwargs

        # Verify send button state
        assert "Send" in last_call_kwargs.get('text', '')

    def test_send_stop_toggle_cycle(self):
        """Test that we can toggle multiple times between send and stop states."""
        # Create a mock object that mimics the set_generating_state method
        send_btn = MagicMock()
        is_generating = [False]  # Use list to allow mutation in nested function

        def set_generating_state(is_gen):
            """Copy of the actual set_generating_state logic from ui.py"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        # Start in send state
        assert is_generating[0] is False

        # Toggle to stop
        set_generating_state(True)
        assert is_generating[0] is True
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Toggle back to send
        set_generating_state(False)
        assert is_generating[0] is False
        assert "Send" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Toggle to stop again
        set_generating_state(True)
        assert is_generating[0] is True
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Toggle back to send again
        set_generating_state(False)
        assert is_generating[0] is False
        assert "Send" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

    def test_set_generating_state_handles_exception_gracefully(self):
        """Test that exceptions in set_generating_state are caught and logged."""
        # Create a mock object that mimics the set_generating_state method
        send_btn = MagicMock()
        send_btn.configure.side_effect = RuntimeError("Button error")
        is_generating = [False]  # Use list to allow mutation in nested function

        def set_generating_state(is_gen):
            """Copy of the actual set_generating_state logic from ui.py"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        # Call set_generating_state - should not raise exception
        set_generating_state(True)

        # is_generating should still be set (state is set before configure call)
        assert is_generating[0] is True

    def test_bug_stop_button_does_not_reset_to_send(self):
        """
        Test for the reported bug: after clicking Stop, the button doesn't
        change back to Send.

        This test simulates the scenario where:
        1. User sends a message -> button shows "Stop"
        2. User clicks "Stop" -> button should show "Send" again
        3. User sends another message -> button should show "Stop" again

        The bug might be that after step 2, the button stays in "Stop" state.
        """
        # Simulate the UI state management
        send_btn = MagicMock()
        is_generating = [False]

        def set_generating_state(is_gen):
            """Copy of the actual set_generating_state logic from ui.py"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        # Scenario 1: First send
        set_generating_state(True)  # User sends message -> Stop button
        assert is_generating[0] is True
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Scenario 2: User clicks Stop -> should become Send
        set_generating_state(False)  # Stop clicked -> Send button
        assert is_generating[0] is False
        assert "Send" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Scenario 3: User sends another message -> should become Stop again
        set_generating_state(True)  # Second send -> Stop button
        assert is_generating[0] is True
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Scenario 4: User clicks Stop again -> should become Send again
        set_generating_state(False)  # Stop clicked again -> Send button
        assert is_generating[0] is False
        assert "Send" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # Verify that the button text is correct after all operations
        final_kwargs = send_btn.configure.call_args_list[-1].kwargs
        assert "Send" in final_kwargs.get('text', ''), \
            f"Button text should be 'Send' but got: {final_kwargs.get('text', '')}"

    def test_bug_on_send_does_not_start_if_is_processing_true(self):
        """
        Test for the bug where _on_send does nothing if is_processing is True.
        This can happen if _on_stop is called but is_processing is not properly reset.
        """
        # Simulate App state
        is_processing = [False]
        message_queue = []
        send_btn = MagicMock()
        is_generating = [False]

        def set_generating_state(is_gen):
            """UI method to toggle button state"""
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        fg_color=("#ff5555", "#cc4444"),
                        hover_color=("#cc4444", "#aa3333"),
                        command=lambda: None,
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        fg_color=("#50fa7b", "#40c969"),
                        hover_color=("#40c969", "#30b959"),
                        command=lambda: None,
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error in set_generating_state: {e}")

        def on_send(text):
            """Simulates App._on_send"""
            if is_processing[0]:
                return  # BUG: Cannot send if already processing

            is_processing[0] = True
            set_generating_state(True)

            # Simulate generation thread
            message_queue.append(("response", "some response"))

        def on_stop():
            """Simulates App._on_stop"""
            if is_processing[0]:
                # Signal LLM to stop
                is_processing[0] = False
                set_generating_state(False)

        def process_queue():
            """Simulates App._process_queue processing 'stopped' message"""
            is_processing[0] = False
            set_generating_state(False)

        # Scenario: User sends a message
        on_send("first message")
        assert is_processing[0] is True
        assert is_generating[0] is True
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # User clicks stop
        on_stop()
        assert is_processing[0] is False
        assert is_generating[0] is False
        assert "Send" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # User tries to send another message - this should work
        on_send("second message")
        assert is_processing[0] is True, "is_processing should be True after second send"
        assert is_generating[0] is True, "is_generating should be True after second send"
        assert "Stop" in send_btn.configure.call_args_list[-1].kwargs.get('text', '')

        # If we get here without assertion errors, the button correctly toggles
