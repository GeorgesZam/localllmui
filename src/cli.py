#!/usr/bin/env python3
"""
Simple CLI interface for Local LLM UI.
This version works with a mock/simulation mode when models aren't available.
"""

import sys
import os
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


class MockLLM:
    """Mock LLM for testing without downloaded models."""

    def __init__(self):
        self.is_ready = True

    def generate(self, prompt, max_tokens=512, stop=None, temperature=0.1, top_p=0.85):
        """Generate a mock response."""
        # Extract user message from prompt
        user_match = re.search(r'<\|im_start\|user>\n(.+?)<\|im_end\|>', prompt, re.DOTALL)
        user_message = user_match.group(1) if user_match else "your question"

        # Check if there's RAG context
        rag_match = re.search(r'=== CONTEXT ===\n(.+?)\n=== END ===', prompt, re.DOTALL)
        has_rag = rag_match is not None

        if has_rag:
            # Simulated RAG response
            response = f"""Based on the documents provided, I can help answer your question about "{user_message[:50]}...".

The documents contain relevant information that addresses your query. Here's what I found:

• The context includes details related to your question
• Information is available in the source documents
• I can provide specific answers based on this context

Is there anything specific you'd like me to elaborate on?"""
        else:
            # General mock response
            responses = [
                f"""I understand you're asking about: "{user_message[:100]}..."

As a language model, I can help you with various tasks including:
• Answering questions
• Writing and editing text
• Coding assistance
• Analysis and explanations

Note: This is a mock response running in simulation mode. To use the actual LLM, please download the required models (see setup instructions).

How else can I assist you today?""",
                f"""Thanks for your question about "{user_message[:100]}..."

I'm currently running in a lightweight mode without the full language model loaded. This allows the application to start quickly for testing purposes.

Key points:
• The app structure is working correctly
• CLI interface is functional
• Model loading can be enabled when ready

Would you like me to explain how to set up the full model, or is there something else I can help with?"""
            ]
            response = responses[hash(user_message) % len(responses)]

        # Yield the response as tokens
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            if i % 5 == 0:  # Simulate streaming
                import time
                time.sleep(0.02)


class SimpleCLI:
    """Simple CLI interface for the Local LLM."""

    def __init__(self):
        self.llm = None
        self.rag = None
        self.history = []
        self.use_mock = True
        self._check_models()

    def _check_models(self):
        """Check if models are available."""
        model_path = Path(__file__).parent / "models" / "model.gguf"
        self.use_mock = not model_path.exists()

        if self.use_mock:
            print("⚠️  Running in SIMULATION mode (models not found)")
            print(f"   Expected model at: {model_path}")
            print("   The app will work with mock responses")
            print("   Download models to enable full functionality")
            print()

    def load(self):
        """Load the LLM and RAG components."""
        print("🚀 Starting Local LLM CLI...")
        print()

        if self.use_mock:
            print("✓ Using mock LLM (fast startup)")
            self.llm = MockLLM()
        else:
            try:
                print("⏳ Loading models...")
                from llm import LLMEngine
                engine = LLMEngine.get_instance()
                if engine.load():
                    self.llm = engine
                    self.rag = engine.rag
                    print("✓ Models loaded successfully")
                else:
                    print("⚠️  Model loading failed, falling back to mock mode")
                    self.llm = MockLLM()
            except Exception as e:
                print(f"⚠️  Error loading models: {e}")
                print("   Falling back to mock mode")
                self.llm = MockLLM()

        # Try to load RAG for document search
        if not self.use_mock:
            try:
                from rag import RAG
                self.rag = RAG()
                self.rag.initialize(lambda msg: print(f"   {msg}"))
            except:
                pass

        print()
        print("=" * 60)
        print("🤖 Local LLM Assistant - Ready!")
        print("=" * 60)
        print()
        print("Commands:")
        print("  /help     - Show this help")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the application")
        print("  /files    - List available documents (if RAG enabled)")
        print()
        print("Type your message and press Enter to send.")
        print()

    def _format_response(self, text):
        """Format response for display."""
        # Clean up any extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _show_help(self):
        """Show help information."""
        print()
        print("=" * 60)
        print("HELP - Local LLM Assistant")
        print("=" * 60)
        print()
        print("USAGE:")
        print("  Just type your question and press Enter.")
        print()
        print("COMMANDS:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the application")
        print("  /files    - Show loaded documents (RAG mode)")
        print("  /stats    - Show conversation statistics")
        print()
        print("NOTES:")
        if self.use_mock:
            print("  ⚠️  Running in SIMULATION mode")
            print("  Responses are generated for testing purposes")
            print("  Download models to enable full AI functionality")
        if self.rag and hasattr(self.rag, 'documents') and self.rag.documents:
            print(f"  ✓ RAG enabled with {len(self.rag.documents)} document chunks")
        else:
            print("  ℹ️  No documents loaded (add documents to data/ folder)")
        print()

    def _show_stats(self):
        """Show conversation statistics."""
        print()
        print("=" * 60)
        print("CONVERSATION STATISTICS")
        print("=" * 60)
        print(f"  Messages exchanged: {len(self.history)}")
        print(f"  User messages: {sum(1 for h in self.history if h['role'] == 'user')}")
        print(f"  Assistant responses: {sum(1 for h in self.history if h['role'] == 'assistant')}")
        if self.rag and hasattr(self.rag, 'documents'):
            print(f"  Document chunks available: {len(self.rag.documents)}")
        print()

    def _show_files(self):
        """Show loaded documents."""
        print()
        print("=" * 60)
        print("LOADED DOCUMENTS")
        print("=" * 60)

        if self.rag and hasattr(self.rag, 'documents') and self.rag.documents:
            sources = {}
            for doc in self.rag.documents:
                source = doc.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1

            if sources:
                for source, count in sorted(sources.items()):
                    print(f"  📄 {source} ({count} chunks)")
            else:
                print("  No documents loaded")
        else:
            print("  RAG not available or no documents loaded")
            print("  Add documents to the data/ folder to enable RAG")
        print()

    def process_command(self, user_input):
        """Process special commands."""
        if user_input == '/quit':
            return False, None
        elif user_input == '/help':
            self._show_help()
            return True, None
        elif user_input == '/clear':
            self.history = []
            print("✓ Conversation history cleared")
            print()
            return True, None
        elif user_input == '/files':
            self._show_files()
            return True, None
        elif user_input == '/stats':
            self._show_stats()
            return True, None
        else:
            return True, user_input

    def run(self):
        """Run the CLI interface."""
        self.load()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Process commands
                should_continue, message = self.process_command(user_input)

                if not should_continue:
                    print("\n👋 Goodbye!")
                    break

                if message is None:
                    continue

                # Generate response
                print("\nAssistant: ", end='', flush=True)

                response = ""
                try:
                    for token in self.llm.generate(message):
                        print(token, end='', flush=True)
                        response += token
                    print()  # New line after response

                    # Check if response contains error message
                    if "exceed context window" in response or "Error:" in response:
                        print("\n⚠️  Note: The document content is too large for the model's context window.")
                        print("   Try asking a more specific question or reduce the document size.")
                        print("   You can also split large documents into smaller files.")

                except Exception as e:
                    print(f"\nError: {e}")
                    if "context" in str(e).lower() and "window" in str(e).lower():
                        print("\n💡 Tip: This error occurs when documents are too large.")
                        print("   Try asking about specific sections or use smaller documents.")

                # Add to history
                self.history.append({'role': 'user', 'content': message})
                self.history.append({'role': 'assistant', 'content': response})

                print()

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


def main():
    """Main entry point."""
    cli = SimpleCLI()
    cli.run()


if __name__ == "__main__":
    main()
