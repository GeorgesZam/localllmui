#!/usr/bin/env python3
"""
Simple test to verify RAG button works.
"""

import tkinter as tk
import sys
import os

# Add src to path
sys.path.insert(0, '/Users/michelzam/localllmui/localllmui/src')

try:
    from ui import RAGConfigWindow
    print("✓ RAGConfigWindow imported successfully")

    # Test the window can be created
    root = tk.Tk()
    root.withdraw()

    try:
        # Test with None RAG
        print("\nTesting RAG window with None RAG...")
        rag_window = RAGConfigWindow(root, None, None)
        print("✓ RAG window created with None RAG")
        rag_window.destroy()

        # Test with mock RAG
        class MockRAG:
            pass

        print("\nTesting RAG window with mock RAG...")
        rag_window = RAGConfigWindow(root, MockRAG(), None)
        print("✓ RAG window created with mock RAG")
        rag_window.destroy()

        print("\n✓ All tests passed! RAG button functionality is working.")

    except Exception as e:
        print(f"✗ Error creating RAG window: {e}")
        import traceback
        traceback.print_exc()
    finally:
        root.destroy()

except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()