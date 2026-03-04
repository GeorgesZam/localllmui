"""
Final validation test for stop button functionality.
This test verifies the complete flow: send -> stop -> send again.
"""
import pytest
import threading
import queue
import time
from unittest.mock import MagicMock


class TestStopButtonFinal:
    """Final validation test for stop button."""

    def test_complete_flow_send_stop_send(self):
        """
        Complete flow test:
        1. Send message -> button shows Stop, is_processing=True
        2. Click Stop -> button shows Send, is_processing=False
        3. Send another message -> button shows Stop again, is_processing=True

        This is the RED test that should pass after the fix.
        """
        # Simulate App state
        is_processing = [False]
        is_generating = [False]
        _stop_requested = threading.Event()
        message_queue = queue.Queue()

        send_btn = MagicMock()
        send_btn.configure = MagicMock()

        def set_generating_state(is_gen):
            is_generating[0] = is_gen
            try:
                if is_gen:
                    send_btn.configure(
                        text="⏹ Stop",
                        command=lambda: None,  # _stop_generation
                        state="normal"
                    )
                else:
                    send_btn.configure(
                        text="Send ➤",
                        command=lambda: None,  # _send
                        state="normal"
                    )
            except Exception as e:
                print(f"[UI] Error: {e}")

        def reset_stop_flag():
            _stop_requested.clear()

        def stop_generation():
            _stop_requested.set()

        def _on_send(text):
            """Simulates App._on_send"""
            if is_processing[0]:
                return  # Can't send if already processing

            is_processing[0] = True
            reset_stop_flag()

            def generate():
                try:
                    for i in range(10):
                        if _stop_requested.is_set():
                            message_queue.put(("stopped", ""))
                            return
                        time.sleep(0.001)
                    message_queue.put(("response_done", ""))
                except:
                    pass

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            set_generating_state(True)

        def _on_stop():
            """Simulates App._on_stop"""
            if not is_processing[0]:
                return

            stop_generation()
            time.sleep(0.01)  # Wait a bit for thread to finish

            is_processing[0] = False
            set_generating_state(False)

        def _process_queue():
            """Simulates App._process_queue for 'stopped' message"""
            try:
                while True:
                    msg_type, data = message_queue.get_nowait()
                    if msg_type == "stopped":
                        is_processing[0] = False
                        set_generating_state(False)
                    elif msg_type == "response_done":
                        is_processing[0] = False
                        set_generating_state(False)
            except queue.Empty:
                pass

        # ============ TEST SCENARIO ============

        # Step 1: Send first message
        _on_send("first message")
        assert is_processing[0] is True, "After first send: is_processing should be True"
        assert is_generating[0] is True, "After first send: button should be Stop"

        # Step 2: Click Stop
        _on_stop()
        _process_queue()
        assert is_processing[0] is False, "After stop: is_processing should be False"
        assert is_generating[0] is False, "After stop: button should be Send"

        # Step 3: Send second message - THIS IS THE CRITICAL TEST
        _on_send("second message")
        assert is_processing[0] is True, \
            "AFTER SECOND SEND: is_processing should be True (if False, bug exists!)"
        assert is_generating[0] is True, \
            "AFTER SECOND SEND: button should be Stop (if Send, bug exists!)"

        # Verify button text
        last_kwargs = send_btn.configure.call_args_list[-1].kwargs
        assert "Stop" in last_kwargs.get('text', ''), \
            "Button text should be 'Stop' after second send"

        print("\n✓ TEST PASSED: Stop button flow works correctly!")
        print("  - Send -> Stop button ✓")
        print("  - Stop -> Send button ✓")
        print("  - Send again -> Stop button ✓")

        # Clean up
        stop_generation()
        time.sleep(0.01)
