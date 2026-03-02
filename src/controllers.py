"""
MVC Controllers for Local RAG Assistant.

Implements the Controller layer of the MVC pattern, separating
business logic from UI components.
"""

import threading
import queue
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from patterns import Observable, Event, StateEvent, Command, CommandResult, CommandInvoker
from config import ConfigManager
from llm import LLMEngine, GenerationStats
from conversations import ConversationManager
from observers import ProgressObserver, UIStateObserver, DocumentCountObserver


@dataclass
class AppCommand(Command):
    """
    Base command class for application commands.

    Implements Command pattern for undoable actions.
    """

    app_controller: 'AppController'
    command_name: str
    args: tuple
    kwargs: dict

    def __init__(self, app_controller: 'AppController', command_name: str,
                 *args, **kwargs):
        super().__init__()
        self.app_controller = app_controller
        self.command_name = command_name
        self.args = args
        self.kwargs = kwargs

    def execute(self) -> CommandResult:
        """Execute the command."""
        try:
            method = getattr(self.app_controller, f"_cmd_{self.command_name}")
            result = method(*self.args, **self.kwargs)
            return CommandResult.success_result(
                message=f"{self.command_name} completed",
                data=result
            )
        except Exception as e:
            return CommandResult.failure_result(
                message=f"{self.command_name} failed",
                error=str(e)
            )


class AppController(Observable):
    """
    Main application controller.

    Coordinates between the model (LLM, RAG, conversations) and view (UI).
    Implements the Controller layer of the MVC pattern.

    Features:
    - Command pattern for undoable actions
    - Thread-safe operation
    - Observable state changes
    - Progress reporting
    """

    def __init__(self):
        """Initialize application controller."""
        super().__init__()

        # Core components
        self._config = ConfigManager.get_instance()
        self._llm_engine: Optional[LLMEngine] = None
        self._conversation_manager = ConversationManager()

        # Command system
        self._command_invoker = CommandInvoker()

        # Threading
        self._queue = queue.Queue()
        self._is_processing = False

        # State
        self._current_state = "initializing"

        # Observers
        self._progress_observer: Optional[ProgressObserver] = None
        self._ui_observer: Optional[UIStateObserver] = None
        self._doc_count_observer: Optional[DocumentCountObserver] = None

    def initialize(self, progress_callback: Optional[Callable[[str], None]] = None,
                   status_callback: Optional[Callable[[str, bool], None]] = None,
                   state_callback: Optional[Callable[[str, Any], None]] = None,
                   doc_count_callback: Optional[Callable[[int], None]] = None) -> bool:
        """
        Initialize the application.

        Args:
            progress_callback: Callback for progress updates
            status_callback: Callback for status updates (text, is_error)
            state_callback: Callback for state changes
            doc_count_callback: Callback for document count changes

        Returns:
            True if initialization successful
        """
        # Create and attach observers
        if progress_callback or status_callback:
            self._progress_observer = ProgressObserver(progress_callback, status_callback)
            self.attach(self._progress_observer)

        if state_callback:
            self._ui_observer = UIStateObserver(state_callback)
            self.attach(self._ui_observer)

        if doc_count_callback:
            self._doc_count_observer = DocumentCountObserver(doc_count_callback)
            self.attach(self._doc_count_observer)

        # Initialize LLM engine
        self._llm_engine = LLMEngine.get_instance()

        # Attach engine observers
        if self._progress_observer:
            self._llm_engine.attach(self._progress_observer)
        if self._ui_observer:
            self._llm_engine.attach(self._ui_observer)

        # Load model in background
        def load_model():
            success = self._llm_engine.load(progress_callback)
            if not success:
                self.notify(StateEvent.create(StateEvent.ERROR, {
                    'error': self._llm_engine.error
                }))

        threading.Thread(target=load_model, daemon=True).start()

        return True

    def is_ready(self) -> bool:
        """Check if application is ready."""
        return self._llm_engine and self._llm_engine.is_ready

    def get_state(self) -> str:
        """Get current application state."""
        return self._current_state

    # === Conversation Management ===

    def create_conversation(self, title: str = "") -> bool:
        """Create a new conversation."""
        conv = self._conversation_manager.create_conversation(title)
        self.notify(Event('conversation_created', {'id': conv.id}))
        return True

    def get_current_conversation_id(self) -> Optional[str]:
        """Get current conversation ID."""
        return self._conversation_manager.current_id

    def set_current_conversation(self, conv_id: str) -> bool:
        """Set current conversation."""
        conv = self._conversation_manager.set_current(conv_id)
        if conv:
            self.notify(Event('conversation_changed', {'id': conv.id}))
            return True
        return False

    def get_all_conversations(self) -> List:
        """Get all conversations."""
        return self._conversation_manager.get_all()

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        success = self._conversation_manager.delete_conversation(conv_id)
        if success:
            self.notify(Event('conversation_deleted', {'id': conv_id}))
        return success

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        """Rename a conversation."""
        return self._conversation_manager.rename_conversation(conv_id, new_title)

    def clear_history(self) -> None:
        """Clear current conversation history."""
        self._conversation_manager.clear_history()
        if self._llm_engine:
            self._llm_engine.clear_history()
        self.notify(Event('history_cleared', {}))

    # === Message Handling ===

    def send_message(self, message: str,
                     response_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Send a message and generate response.

        Args:
            message: User message
            response_callback: Callback for each response token
        """
        if not self.is_ready():
            return

        # Add user message to conversation
        self._conversation_manager.add_message("user", message)

        def generate():
            """Generate response in background thread."""
            try:
                for token in self._llm_engine.generate(message):
                    if response_callback:
                        response_callback(token)

                    # Also add to queue for main thread processing
                    self._queue.put(("token", token))

                # Notify complete
                self._queue.put(("complete", None))

            except Exception as e:
                self._queue.put(("error", str(e)))

        threading.Thread(target=generate, daemon=True).start()

    def process_queue(self) -> None:
        """Process queued events from background threads."""
        try:
            while True:
                event_type, data = self._queue.get_nowait()

                if event_type == "token":
                    # Add assistant message to conversation
                    if data:
                        self._conversation_manager.add_message("assistant", data)
                elif event_type == "complete":
                    # Generation complete
                    stats = self._llm_engine.get_stats() if self._llm_engine else GenerationStats()
                    self.notify(Event('message_complete', stats.__dict__))
                elif event_type == "error":
                    self.notify(StateEvent.create(StateEvent.ERROR, {'error': data}))

        except queue.Empty:
            pass

    # === Document Management ===

    def add_documents(self, file_paths: List[str],
                     progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Add documents to the RAG system."""
        if not self._llm_engine or not self._llm_engine.rag:
            return False

        def add_docs():
            success = self._llm_engine.rag.add_documents(file_paths, progress_callback)
            if success:
                doc_count = len(self._llm_engine.rag.documents)
                self.notify(Event('documents_added', {'count': len(file_paths), 'total': doc_count}))
            else:
                self.notify(StateEvent.create(StateEvent.ERROR, {'error': 'Failed to add documents'}))

        threading.Thread(target=add_docs, daemon=True).start()
        return True

    def get_document_count(self) -> int:
        """Get current document count."""
        if self._llm_engine and self._llm_engine.rag:
            return len(self._llm_engine.rag.documents)
        return 0

    # === Command System ===

    def execute_command(self, command_name: str, *args, **kwargs) -> CommandResult:
        """Execute a command via the command system."""
        command = AppCommand(self, command_name, *args, **kwargs)
        return self._command_invoker.execute(command)

    def undo_last_command(self) -> Optional[CommandResult]:
        """Undo the last command."""
        return self._command_invoker.undo()

    def redo_last_command(self) -> Optional[CommandResult]:
        """Redo the last undone command."""
        return self._command_invoker.redo()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._command_invoker.can_undo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._command_invoker.can_redo()

    # === Internal Commands ===

    def _cmd_load_model(self) -> bool:
        """Command: Load the LLM model."""
        if self._llm_engine:
            return self._llm_engine.load()
        return False

    def _cmd_clear_cache(self) -> bool:
        """Command: Clear the RAG cache."""
        if self._llm_engine and self._llm_engine.rag:
            self._llm_engine.rag.clear_cache()
            return True
        return False

    def _cmd_export_conversation(self, conv_id: str, filepath: str) -> bool:
        """Command: Export conversation to file."""
        import json
        try:
            conv = self._conversation_manager.conversations.get(conv_id)
            if conv:
                with open(filepath, 'w') as f:
                    json.dump(conv.to_dict(), f, indent=2)
                return True
        except Exception as e:
            print(f"[Controller] Export error: {e}")
        return False

    def _cmd_import_conversation(self, filepath: str) -> Optional[str]:
        """Command: Import conversation from file."""
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            conv = ConversationManager.Conversation.from_dict(data)
            self._conversation_manager.conversations[conv.id] = conv
            return conv.id
        except Exception as e:
            print(f"[Controller] Import error: {e}")
        return None

    # === Shutdown ===

    def shutdown(self) -> None:
        """Shutdown the application controller."""
        if self._llm_engine:
            self._llm_engine.detach_all()

        self.detach_all()

        self.notify(Event('shutdown', {}))


# Global controller instance
_app_controller: Optional[AppController] = None


def get_app_controller() -> AppController:
    """Get the global application controller instance."""
    global _app_controller
    if _app_controller is None:
        _app_controller = AppController()
    return _app_controller
