#!/usr/bin/env python3
"""
Local Chat - Entry Point with conversations support.
"""

import sys
import os

if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    if hasattr(sys, '_MEIPASS'):
        import multiprocessing
        multiprocessing.freeze_support()

import customtkinter as ctk
import threading
from llm import LLMEngine
from ui import ChatUI
from conversations import ConversationManager


class App:
    """Main application with conversation management."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.engine = LLMEngine()
        self.conv_manager = ConversationManager()
        self.generating = False
        
        self.ui = ChatUI(
            self.root,
            on_send=self.send,
            on_clear=self.clear,
            on_load_files=self.load_files,
            on_new_chat=self.new_chat,
            on_select_chat=self.select_chat,
            on_delete_chat=self.delete_chat
        )
        
        self.ui.add_message("System", "🚀 Starting...", "system")
        self._load_model()
    
    def _load_model(self):
        """Load model in background."""
        def task():
            success = self.engine.load(
                on_progress=lambda m: self.root.after(0, lambda: self.ui.add_message("Debug", m, "system"))
            )
            if success:
                self.root.after(0, self._on_ready)
            else:
                self.root.after(0, lambda: self.ui.set_status("❌ Failed", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _on_ready(self):
        """Called when model is ready."""
        self.ui.set_status("✅ Ready!")
        
        # Create first conversation if none exists
        if not self.conv_manager.get_all():
            self.conv_manager.create_conversation("Welcome Chat")
        
        # Load current conversation
        self._load_current_conversation()
        self._update_sidebar()
        
        self.ui.add_message("System", "✨ Ready! Start chatting or load documents.", "system")
    
    def _update_sidebar(self):
        """Update sidebar with conversations."""
        convs = self.conv_manager.get_all()
        current = self.conv_manager.get_current()
        current_id = current.id if current else None
        self.ui.update_sidebar(convs, current_id)
    
    def _load_current_conversation(self):
        """Load current conversation into UI and RAG."""
        conv = self.conv_manager.get_current()
        
        if not conv:
            self.ui.clear_chat()
            self.ui.update_doc_count(0)
            return
        
        # Load messages into UI
        self.ui.load_messages(conv.messages)
        
        # Load documents into RAG
        self.engine.rag.set_conversation(
            conv.id,
            conv.document_ids,
            on_progress=lambda m: print(f"[App] {m}")
        )
        
        self.ui.update_doc_count(len(conv.document_ids))
    
    def new_chat(self):
        """Create new conversation."""
        self.conv_manager.create_conversation()
        self._load_current_conversation()
        self._update_sidebar()
        self.ui.add_message("System", "💬 New conversation started!", "system")
    
    def select_chat(self, conv_id: str):
        """Select a conversation."""
        self.conv_manager.set_current(conv_id)
        self._load_current_conversation()
        self._update_sidebar()
    
    def delete_chat(self, conv_id: str):
        """Delete a conversation."""
        self.conv_manager.delete_conversation(conv_id)
        self._load_current_conversation()
        self._update_sidebar()
    
    def send(self, message: str):
        """Send a message."""
        if self.generating or not self.engine.is_ready:
            return
        
        # Ensure we have a conversation
        if not self.conv_manager.get_current():
            self.conv_manager.create_conversation()
            self._update_sidebar()
        
        # Add to UI and conversation
        self.ui.add_message("You", message, "user")
        self.conv_manager.add_message("user", message)
        
        self.generating = True
        self.ui.set_enabled(False)
        self.ui.set_status("🤔 Thinking...")
        
        threading.Thread(target=self._generate, args=(message,), daemon=True).start()
    
    def _generate(self, message: str):
        """Generate response."""
        self.root.after(0, lambda: self.ui.add_message("Assistant", "", "bot"))
        
        conv = self.conv_manager.get_current()
        history = conv.messages[:-1] if conv else []  # Exclude the message we just added
        
        full_response = ""
        try:
            for token in self.engine.generate(message, history):
                full_response += token
                self.root.after(0, lambda t=token: self.ui.stream(t))
        except Exception as e:
            self.root.after(0, lambda: self.ui.add_message("Error", str(e), "error"))
        finally:
            # Save assistant response
            clean_response = full_response.split("📚 Sources:")[0].strip()
            if clean_response:
                self.conv_manager.add_message("assistant", clean_response)
            
            self.root.after(0, self._done)
    
    def _done(self):
        """Generation complete."""
        self.generating = False
        self.ui.set_enabled(True)
        self.ui.set_status("✅ Ready!")
        self.ui.focus_input()
        self._update_sidebar()  # Update title if it changed
    
    def clear(self):
        """Clear current conversation messages."""
        self.conv_manager.clear_history()
        self.ui.clear_chat()
        self.ui.add_message("System", "💬 Messages cleared.", "system")
    
    def load_files(self, files: tuple):
        """Load documents into current conversation."""
        conv = self.conv_manager.get_current()
        if not conv:
            self.conv_manager.create_conversation()
            conv = self.conv_manager.get_current()
            self._update_sidebar()
        
        self.ui.add_message("System", f"📥 Loading {len(files)} file(s)...", "system")
        self.ui.set_status("⏳ Loading...")
        
        def task():
            added = self.engine.rag.add_documents(
                conv.id,
                files,
                on_progress=lambda m: self.root.after(0, lambda msg=m: self.ui.add_message("Debug", msg, "system"))
            )
            
            if added:
                # Update conversation with new documents
                for filename in added:
                    self.conv_manager.add_document(filename)
                
                # Reload RAG with updated documents
                conv = self.conv_manager.get_current()
                self.engine.rag.set_conversation(conv.id, conv.document_ids)
                
                self.root.after(0, lambda: self.ui.update_doc_count(len(conv.document_ids)))
                self.root.after(0, lambda: self.ui.add_message("System", f"✅ Loaded {len(added)} files!", "system"))
                self.root.after(0, lambda: self.ui.set_status("✅ Ready!"))
                self.root.after(0, self._update_sidebar)
            else:
                self.root.after(0, lambda: self.ui.add_message("Error", "❌ Failed to load", "error"))
                self.root.after(0, lambda: self.ui.set_status("❌ Failed", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def run(self):
        """Run the app."""
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
