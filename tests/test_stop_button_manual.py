"""
Manual test script for stop button functionality.
Run this to verify the fix works in the actual application.

Usage:
1. Start the application normally
2. Send a message
3. Click Stop before the response completes
4. Verify the button shows "Send" again
5. Send another message
6. Verify the button shows "Stop" again

Expected behavior:
- After clicking Stop, button should show "Send"
- You should be able to send another message
- The button should toggle correctly between Send and Stop
"""

print("""
========================================
STOP BUTTON MANUAL TEST
========================================

This script provides instructions for manually testing the stop button fix.

STEPS:
1. Run the application: python src/main.py
2. Type a message and click Send (or press Enter)
3. Before the response completes, click the "Stop" button
4. OBSERVE: The button should change to "Send ➤"
5. Type another message and click Send
6. OBSERVE: The button should change to "⏹ Stop"

If the bug still exists:
- After step 3, the button might stay as "⏹ Stop"
- In step 5, clicking Send might not work

ADDITIONAL TESTS:
- Try the cycle multiple times (Send -> Stop -> Send -> Stop)
- Try stopping immediately after sending
- Try stopping after some response has been received

The fix includes:
1. Resetting _generation_thread to None after stop
2. Added safety check for alive thread before starting new generation
3. Improved logging in set_generating_state

Check the console output for debug messages:
[UI] set_generating_state(True/False)
[App] Warning: Previous generation thread still alive...
""")
