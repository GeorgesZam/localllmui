import os
import sys
import threading
import queue
from pathlib import Path
from typing import Optional

# CRITICAL FIX: Add src directory to path for PyInstaller standalone builds
# This ensures imports work when bundled in a single .exe
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        src_path = os.path.join(sys._MEIPASS, 'src')
    else:
        src_path = os.path.join(os.path.dirname(sys.executable), 'src')
else:
    # Running from source
    src_path = os.path.join(os.path.dirname(__file__))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

import customtkinter as ctk
from tkinter import filedialog, messagebox

from llm import LLMEngine
from conversations import ConversationManager
from ui import ChatUI, ModelCatalogWindow, RAGConfigWindow
from skills_manager import SkillsManager, SkillExecutor
from model_manager import ModelManager
import config


def get_resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return str(base_path / relative_path)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"🤖 {config.APP_NAME}")
        self.geometry(config.WINDOW_SIZE)
        self.minsize(900, 600)

        self.llm = LLMEngine()
        self.conv_manager = ConversationManager()
        self.message_queue = queue.Queue()
        self.is_processing = False
        self.last_question = None
        self._generation_thread = None
        self._is_switching_model = False

        # Model management
        self.model_manager = ModelManager(config.models_dir)
        self.model_manager.scan_for_models()
        self._model_catalog_window = None

        # Skills system
        self.skills_manager = SkillsManager()
        self.skills_executor = SkillExecutor(self.skills_manager)
        self.llm.set_skills_executor(self.skills_executor)

        self._create_ui()
        self._initialize_models()

        # Attach UI to LLM engine events
        self.llm.attach(self.ui)

    def _create_ui(self):
        self.ui = ChatUI(
            self,
            on_send=self._on_send,
            on_stop=self._on_stop,
            on_clear=self._on_clear,
            on_load_files=self._on_load_files,
            on_new_chat=self._on_new_chat,
            on_select_chat=self._on_select_chat,
            on_delete_chat=self._on_delete_chat,
            skills_manager=self.skills_manager,
            on_skill_toggle=self._on_skill_toggle,
            on_open_model_catalog=self._open_model_catalog,
            model_manager=self.model_manager
        )

    def _on_skill_toggle(self, skill_id: str, enabled: bool):
        """Handle skill toggle event - update LLM with new skills."""
        if self.llm.is_ready:
            self.llm.update_skills_content(self.skills_manager)

    def _initialize_models(self):
        def init():
            try:
                self.llm.load(on_progress=lambda msg: self.message_queue.put(("status", msg)))
                self.message_queue.put(("initialized", None))
            except Exception as e:
                self.message_queue.put(("error", str(e)))

        threading.Thread(target=init, daemon=True).start()
        self._process_queue()

    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "status":
                    self.ui.set_status(data, is_error=False)

                elif msg_type == "initialized":
                    self.llm.update_skills_content(self.skills_manager)
                    # Don't set "Ready" status if we're in the middle of switching models
                    if not self._is_switching_model:
                        self.ui.set_status("Ready", is_error=False)
                    self.ui.set_enabled(True)
                    self.ui.focus_input()
                    self._update_sidebar()
                    # Set initial model display
                    model_id = self.llm.get_current_model_id()
                    self.ui.set_model(model_id)

                elif msg_type == "error":
                    self.ui.set_status(f"Error: {data}", is_error=True)
                    messagebox.showerror("Error", data)

                elif msg_type == "response":
                    self.ui.stream(data)

                elif msg_type == "response_done":
                    conv = self.conv_manager.get_current()
                    if conv:
                        self.conv_manager.add_message("assistant", data)
                        # Ajouter à l'historique de la conversation
                        self.ui.response_handler.add_to_conversation(self.last_question, data)
                    # Apply markdown formatting to code blocks
                    self.ui.apply_markdown_to_last_message()
                    self.is_processing = False
                    self.ui.set_generating_state(False)
                    self.ui.set_enabled(True)

                elif msg_type == "stopped":
                    # Handle partial response from stopped generation
                    if data.strip():
                        conv = self.conv_manager.get_current()
                        if conv:
                            self.conv_manager.add_message("assistant", data)
                    self.is_processing = False
                    self.ui.set_generating_state(False)
                    self.ui.set_enabled(True)
                    self.ui.set_status("Generation stopped", is_error=False)

                elif msg_type == "model_switched":
                    self._is_switching_model = False
                    self.ui.set_status("Ready", is_error=False)
                    self.ui.set_model(data)  # data contains model_id
                    self.ui.set_enabled(True)
                    self.ui.focus_input()
                    self._update_sidebar()

                elif msg_type == "switch_failed":
                    self._is_switching_model = False
                    self.ui.set_status(f"Failed to switch model: {data}", is_error=True)
                    self.ui.set_enabled(True)
                    self.ui.focus_input()

        except queue.Empty:
            pass

        self.after(50, self._process_queue)

    def _update_sidebar(self):
        conversations = self.conv_manager.get_all()
        current_id = self.conv_manager.current_id
        self.ui.update_sidebar(conversations, current_id)

    def _on_stop(self):
        """Handle stop button click."""
        if self.is_processing:
            # Signal LLM to stop
            self.llm.stop_generation()

            # Wait for thread to finish (with timeout)
            if self._generation_thread:
                self._generation_thread.join(timeout=2)
                self._generation_thread = None  # Clear thread reference

            # Reset state
            self.is_processing = False
            self.ui.set_generating_state(False)
            self.ui.set_enabled(True)
            self.ui.set_status("Generation stopped", is_error=False)

    def _on_send(self, text: str):
        if self.is_processing:
            return

        # Safety check: if a previous thread still exists, wait for it
        if self._generation_thread and self._generation_thread.is_alive():
            print("[App] Warning: Previous generation thread still alive, waiting...")
            self._generation_thread.join(timeout=1)
            self._generation_thread = None

        # Store the question for later use in response processing
        self.last_question = text

        if not self.llm.is_ready:
            # Check if there's a download in progress
            download_progress = self.model_manager.get_download_progress()
            if download_progress and download_progress.status == "downloading":
                self.ui.set_status(f"Downloading model... {download_progress.percentage:.1f}%", is_error=False)
                messagebox.showinfo("Model Downloading", f"Please wait for the model to finish downloading.\n\nProgress: {download_progress.percentage:.1f}%")
            else:
                self.ui.set_status("Model not ready", is_error=True)
                messagebox.showwarning("Model Not Ready", "The model is not ready yet. Please wait for initialization to complete.")
            return

        conv = self.conv_manager.get_current()
        if not conv:
            conv = self.conv_manager.create_conversation()

        # Store the last question for response handling
        self.last_question = text

        self.conv_manager.add_message("user", text)
        self.ui.add_message("You", text)

        self.is_processing = True
        self.ui.set_enabled(False)

        # Reset stop flag for new generation
        self.llm.reset_stop_flag()

        # Get current conversation's document IDs for RAG filtering
        allowed_docs = conv.document_ids if conv else []

        def generate():
            try:
                response = ""
                for chunk in self.llm.generate(text, allowed_document_sources=allowed_docs):
                    if self.llm._stop_requested.is_set():
                        break
                    response += chunk
                    self.message_queue.put(("response", chunk))
                if self.llm._stop_requested.is_set():
                    self.message_queue.put(("stopped", response))
                else:
                    self.message_queue.put(("response_done", response))
            except Exception as e:
                self.message_queue.put(("error", str(e)))
                self.message_queue.put(("response_done", ""))

        self._generation_thread = threading.Thread(target=generate, daemon=True)
        self._generation_thread.start()

        # Tell UI to show stop button
        self.ui.set_generating_state(True)

    def _on_clear(self):
        if messagebox.askyesno("Clear Chat", "Clear current chat history?"):
            self.conv_manager.clear_history()
            self.ui.clear_chat()

    def _on_load_files(self, files):
        conv = self.conv_manager.get_current()
        if not conv:
            conv = self.conv_manager.create_conversation()

        docs_folder = self.conv_manager.get_conversation_docs_folder()
        added = 0

        for file_path in files:
            filename = Path(file_path).name
            dest = os.path.join(docs_folder, filename)

            try:
                import shutil
                shutil.copy2(file_path, dest)
                self.conv_manager.add_document(filename)
                added += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

        if added > 0:
            self.llm.rag.add_documents(list(files))
            # Add mapping from conversation to documents
            for doc_id in conv.document_ids[-added:]:  # Get the newly added documents
                self.llm.rag.add_document_to_conversation(conv.id, doc_id)
            self.llm.rag.add_conversation_mapping(conv.id, conv.document_ids)
            self.ui.update_doc_count(len(conv.document_ids))
            self.ui.update_doc_info()
            self._update_sidebar()

    def _on_new_chat(self):
        conv = self.conv_manager.create_conversation()
        self.ui.clear_chat()
        # Update RAG conversation mapping
        if hasattr(self.llm, 'rag') and self.llm.rag:
            self.llm.rag.add_conversation_mapping(conv.id, conv.document_ids)
        self.ui.update_doc_info()
        # Set conversation for response handler
        self.ui.set_conversation_for_response_handler(conv.id)
        self._update_sidebar()

    def _on_select_chat(self, conv_id: str):
        conv = self.conv_manager.set_current(conv_id)
        if conv:
            self.ui.clear_chat()
            self.ui.load_messages(conv.messages)
            # Update RAG conversation mapping
            if hasattr(self.llm, 'rag') and self.llm.rag:
                self.llm.rag.add_conversation_mapping(conv_id, conv.document_ids)
            self.ui.update_doc_count(len(conv.document_ids))
            self.ui.update_doc_info()
            # Set conversation for response handler
            self.ui.set_conversation_for_response_handler(conv_id)
            self._update_sidebar()

    def _on_delete_chat(self, conv_id: str):
        self.conv_manager.delete_conversation(conv_id)
        conv = self.conv_manager.get_current()
        self.ui.clear_chat()
        if conv:
            self.ui.load_messages(conv.messages)
        self._update_sidebar()

    def _open_model_catalog(self):
        """Open the model catalog window."""
        ModelCatalogWindow(
            self,
            self.model_manager,
            on_model_select=self._on_model_selected,
            on_model_download=self._on_model_download
        )

    def _on_model_selected(self, model_id: str):
        """Handle model selection from catalog."""
        if messagebox.askyesno(
            "Switch Model",
            "Switch to this model? The current model will be unloaded."
        ):
            model_path = self.model_manager.get_model_path(model_id)
            if model_path:
                # Set as active in manager
                self.model_manager.set_active_model(model_id)

                # Disable UI during switch
                self._is_switching_model = True
                self.ui.set_enabled(False)
                self.ui.set_status("Switching model...", is_error=False)

                def switch():
                    try:
                        success = self.llm.switch_model(
                            model_path,
                            model_id,
                            on_progress=lambda msg: self.message_queue.put(("status", msg))
                        )

                        if success:
                            # Update skills for new model
                            self.llm.update_skills_content(self.skills_manager)
                            self.message_queue.put(("model_switched", model_id))
                        else:
                            # Get the specific error from LLM engine
                            error_msg = self.llm.error if hasattr(self.llm, 'error') else "Failed to switch model"
                            self.message_queue.put(("error", f"Failed to switch model: {error_msg}"))
                            # Ensure UI is re-enabled on failure
                            self.message_queue.put(("switch_failed", error_msg))

                    except Exception as e:
                        self.message_queue.put(("error", str(e)))
                        # Ensure UI is re-enabled on exception
                        self.message_queue.put(("switch_failed", str(e)))

                threading.Thread(target=switch, daemon=True).start()

    def _on_model_download(self, model_id: str):
        """Handle model download request."""
        pass  # Handled by the catalog window directly


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
