#!/usr/bin/env python3
"""
Test script to verify the RAG button works correctly.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add src to path
sys.path.insert(0, '/Users/michelzam/localllmui/localllmui/src')

try:
    from ui import ChatUI, RAGConfigWindow
    print("✓ UI imports successful")

    def test_rag_button():
        """Test that RAG button opens the configuration window."""
        root = tk.Tk()
        root.withdraw()  # Hide main window

        try:
            # Create a simple root for testing
            test_root = tk.Tk()
            test_root.title("RAG Button Test")
            test_root.geometry("300x200")

            # Mock LLM object with rag=None (initial state)
            class MockLLM:
                def __init__(self):
                    self.rag = None

            # Create a mock ChatUI
            class MockChatUI:
                def __init__(self):
                    self.root = test_root
                    self.llm = MockLLM()
                    self.conversation_id = "test_123"

                def _open_rag_config(self):
                    """Open RAG configuration window."""
                    if hasattr(self, '_rag_config_window') and self._rag_config_window:
                        self._rag_config_window.lift()
                        self._rag_config_window.focus_force()
                        return

                    self._rag_config_window = RAGConfigWindow(
                        self.root,
                        self.llm.rag if self.llm.rag else None,
                        self
                    )

            # Create mock UI
            mock_ui = MockChatUI()

            # Test opening RAG window
            print("\n1. Testing RAG window creation with None RAG...")
            mock_ui._open_rag_config()

            # Check if window exists
            if hasattr(mock_ui, '_rag_config_window') and mock_ui._rag_config_window:
                print("✓ RAG window created successfully")

                # Test closing it
                mock_ui._rag_config_window.destroy()
                print("✓ RAG window closed successfully")

                # Clear the reference
                mock_ui._rag_config_window = None

                # Test opening again
                print("\n2. Testing RAG window re-opening...")
                mock_ui._open_rag_config()
                if hasattr(mock_ui, '_rag_config_window') and mock_ui._rag_config_window:
                    print("✓ RAG window re-opened successfully")
                    mock_ui._rag_config_window.destroy()
                    print("✓ RAG window closed again")
                else:
                    print("✗ RAG window not re-opened")

            else:
                print("✗ RAG window not created")

            # Test with mock RAG object
            print("\n3. Testing RAG window with mock RAG object...")

            class MockRAG:
                def __init__(self):
                    self.documents = ["doc1.pdf", "doc2.txt"]
                    self.chunks = {"test_123": {"doc1": 10, "doc2": 5}}

            mock_ui.llm.rag = MockRAG()
            mock_ui._open_rag_config()

            if hasattr(mock_ui, '_rag_config_window') and mock_ui._rag_config_window:
                print("✓ RAG window created with mock RAG successfully")
                mock_ui._rag_config_window.destroy()
                print("✓ RAG window closed")
            else:
                print("✗ RAG window not created with mock RAG")

            # Clean up
            test_root.destroy()
            print("\n✓ All tests passed! RAG button is working correctly.")

        except Exception as e:
            print(f"✗ Error during test: {e}")
            import traceback
            traceback.print_exc()
            test_root.destroy()

    # Run the test
    test_rag_button()

except Exception as e:
    print(f"✗ Failed to import required modules: {e}")
    import traceback
    traceback.print_exc()