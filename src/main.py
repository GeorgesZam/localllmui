import os
import sys
import threading
import queue
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

from llm import LLMEngine
from conversations import ConversationManager
from ui import ChatUI
from skills_manager import SkillsManager, SkillExecutor
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

        # Skills system
        self.skills_manager = SkillsManager()
        self.skills_executor = SkillExecutor(self.skills_manager)
        self.llm.set_skills_executor(self.skills_executor)

        self._create_ui()
        self._initialize_models()

    def _create_ui(self):
        self.ui = ChatUI(
            self,
            on_send=self._on_send,
            on_clear=self._on_clear,
            on_load_files=self._on_load_files,
            on_new_chat=self._on_new_chat,
            on_select_chat=self._on_select_chat,
            on_delete_chat=self._on_delete_chat,
            skills_manager=self.skills_manager,
            on_skill_toggle=self._on_skill_toggle
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
                    self.ui.set_status("Ready", is_error=False)
                    self.ui.set_enabled(True)
                    self.ui.focus_input()
                    self._update_sidebar()

                elif msg_type == "error":
                    self.ui.set_status(f"Error: {data}", is_error=True)
                    messagebox.showerror("Error", data)

                elif msg_type == "response":
                    self.ui.stream(data)

                elif msg_type == "response_done":
                    conv = self.conv_manager.get_current()
                    if conv:
                        self.conv_manager.add_message("assistant", data)
                    self.is_processing = False
                    self.ui.set_enabled(True)

        except queue.Empty:
            pass

        self.after(50, self._process_queue)

    def _update_sidebar(self):
        conversations = self.conv_manager.get_all()
        current_id = self.conv_manager.current_id
        self.ui.update_sidebar(conversations, current_id)

    def _on_send(self, text: str):
        if self.is_processing or not self.llm.is_ready:
            return

        conv = self.conv_manager.get_current()
        if not conv:
            conv = self.conv_manager.create_conversation()

        self.conv_manager.add_message("user", text)
        self.ui.add_message("You", text)

        self.is_processing = True
        self.ui.set_enabled(False)

        def generate():
            try:
                response = ""
                for chunk in self.llm.generate(text):
                    response += chunk
                    self.message_queue.put(("response", chunk))
                self.message_queue.put(("response_done", response))
            except Exception as e:
                self.message_queue.put(("error", str(e)))
                self.message_queue.put(("response_done", ""))

        threading.Thread(target=generate, daemon=True).start()

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
            self.ui.update_doc_count(len(conv.document_ids))
            self._update_sidebar()

    def _on_new_chat(self):
        self.conv_manager.create_conversation()
        self.ui.clear_chat()
        self._update_sidebar()

    def _on_select_chat(self, conv_id: str):
        conv = self.conv_manager.set_current(conv_id)
        if conv:
            self.ui.clear_chat()
            self.ui.load_messages(conv.messages)
            self.ui.update_doc_count(len(conv.document_ids))
            self._update_sidebar()

    def _on_delete_chat(self, conv_id: str):
        self.conv_manager.delete_conversation(conv_id)
        conv = self.conv_manager.get_current()
        self.ui.clear_chat()
        if conv:
            self.ui.load_messages(conv.messages)
        self._update_sidebar()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
