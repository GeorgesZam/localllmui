"""
Runtime hook for tiktoken.

This hook ensures that tiktoken extensions are properly loaded
in PyInstaller bundles by forcing their import at startup.
"""

import sys

# Force import tiktoken extensions to ensure they're available
try:
    import tiktoken_ext
    import tiktoken_ext.openai_public
except ImportError:
    # These will be created dynamically by tiktoken
    # Just ensure tiktoken itself is available
    import tiktoken

print("[RuntimeHook] tiktoken extensions initialized")
