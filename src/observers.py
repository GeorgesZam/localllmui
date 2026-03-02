"""
UI Observers for Progress and Event Reporting.

Implements Observer pattern for updating UI components in response
to model loading, document processing, and generation events.
"""

from typing import Optional, Callable, Any
from datetime import datetime

from patterns import Observer, Observable, Event, ProgressEvent, StateEvent


class ProgressObserver(Observer):
    """
    Observer for progress updates during long-running operations.

    Suitable for model loading, document processing, etc.
    """

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None,
                 status_callback: Optional[Callable[[str, bool], None]] = None):
        """
        Initialize progress observer.

        Args:
            progress_callback: Called with progress message
            status_callback: Called with status text and error flag
        """
        self._progress_callback = progress_callback
        self._status_callback = status_callback
        self._last_message = ""
        self._message_count = 0

    def on_notify(self, event: Event) -> None:
        """Handle progress events."""
        if event.name == ProgressEvent.START:
            self._handle_start(event)
        elif event.name == ProgressEvent.UPDATE:
            self._handle_update(event)
        elif event.name == ProgressEvent.COMPLETE:
            self._handle_complete(event)
        elif event.name == ProgressEvent.ERROR:
            self._handle_error(event)

    def _handle_start(self, event: Event) -> None:
        """Handle progress start."""
        message = event.data.get('message', 'Starting...')
        self._last_message = message
        self._message_count = 0
        self._notify(message)

    def _handle_update(self, event: Event) -> None:
        """Handle progress update."""
        percent = event.data.get('percent', 0)
        message = event.data.get('message', '')

        if message:
            self._last_message = message
        else:
            message = self._last_message

        full_message = f"{message} ({percent:.0f}%)" if percent > 0 else message
        self._notify(full_message)

    def _handle_complete(self, event: Event) -> None:
        """Handle progress complete."""
        message = event.data.get('message', 'Complete!')
        self._notify(message)
        if self._status_callback:
            self._status_callback(message, False)

    def _handle_error(self, event: Event) -> None:
        """Handle progress error."""
        message = event.data.get('message', 'Error occurred')
        self._notify(message)
        if self._status_callback:
            self._status_callback(message, True)

    def _notify(self, message: str) -> None:
        """Send notification to callback."""
        self._message_count += 1
        if self._progress_callback:
            self._progress_callback(message)


class UIStateObserver(Observer):
    """
    Observer for UI state changes.

    Updates UI elements based on application state changes.
    """

    def __init__(self, ui_callback: Optional[Callable[[str, Any], None]] = None):
        """
        Initialize UI state observer.

        Args:
            ui_callback: Called with state name and data
        """
        self._ui_callback = ui_callback
        self._current_state = "initializing"

    def on_notify(self, event: Event) -> None:
        """Handle state change events."""
        if event.name == StateEvent.CHANGED:
            self._handle_state_change(event)

    def _handle_state_change(self, event: Event) -> None:
        """Handle state change."""
        new_state = event.data.get('state', 'unknown')
        self._current_state = new_state

        if self._ui_callback:
            self._ui_callback(new_state, event.data.get('data'))

    def get_current_state(self) -> str:
        """Get current state."""
        return self._current_state


class DocumentCountObserver(Observer):
    """
    Observer for document count changes.

    Updates document count displays when documents are added/removed.
    """

    def __init__(self, count_callback: Optional[Callable[[int], None]] = None):
        """
        Initialize document count observer.

        Args:
            count_callback: Called with new document count
        """
        self._count_callback = count_callback
        self._current_count = 0

    def on_notify(self, event: Event) -> None:
        """Handle document events."""
        if event.name == 'documents_added':
            self._handle_added(event)
        elif event.name == 'documents_removed':
            self._handle_removed(event)
        elif event.name == 'documents_cleared':
            self._handle_cleared(event)

    def _handle_added(self, event: Event) -> None:
        """Handle documents added."""
        added = event.data.get('count', 0)
        self._current_count += added
        self._notify()

    def _handle_removed(self, event: Event) -> None:
        """Handle documents removed."""
        removed = event.data.get('count', 0)
        self._current_count = max(0, self._current_count - removed)
        self._notify()

    def _handle_cleared(self, event: Event) -> None:
        """Handle documents cleared."""
        self._current_count = 0
        self._notify()

    def _notify(self) -> None:
        """Notify callback of count change."""
        if self._count_callback:
            self._count_callback(self._current_count)

    def get_count(self) -> int:
        """Get current document count."""
        return self._current_count


class GenerationObserver(Observer):
    """
    Observer for text generation events.

    Tracks generation statistics and provides real-time updates.
    """

    def __init__(self, token_callback: Optional[Callable[[str], None]] = None,
                 complete_callback: Optional[Callable[[dict], None]] = None):
        """
        Initialize generation observer.

        Args:
            token_callback: Called with each generated token
            complete_callback: Called when generation completes with stats
        """
        self._token_callback = token_callback
        self._complete_callback = complete_callback
        self._tokens_generated = 0
        self._start_time: Optional[datetime] = None
        self._current_text = ""

    def on_notify(self, event: Event) -> None:
        """Handle generation events."""
        if event.name == 'generation_start':
            self._handle_start(event)
        elif event.name == 'generation_token':
            self._handle_token(event)
        elif event.name == 'generation_complete':
            self._handle_complete(event)

    def _handle_start(self, event: Event) -> None:
        """Handle generation start."""
        self._tokens_generated = 0
        self._current_text = ""
        self._start_time = datetime.now()

    def _handle_token(self, event: Event) -> None:
        """Handle token generation."""
        token = event.data.get('token', '')
        self._tokens_generated += 1
        self._current_text += token

        if self._token_callback:
            self._token_callback(token)

    def _handle_complete(self, event: Event) -> None:
        """Handle generation complete."""
        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        tps = self._tokens_generated / elapsed if elapsed > 0 else 0

        stats = {
            'tokens': self._tokens_generated,
            'time': elapsed,
            'tokens_per_second': tps,
            'text': self._current_text,
            'has_code': event.data.get('has_code', False),
            'rag_results': event.data.get('rag_results', 0)
        }

        if self._complete_callback:
            self._complete_callback(stats)

    def get_stats(self) -> dict:
        """Get current generation statistics."""
        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        return {
            'tokens': self._tokens_generated,
            'time': elapsed,
            'tokens_per_second': self._tokens_generated / elapsed if elapsed > 0 else 0
        }


class CompositeObserver(Observer):
    """
    Composite observer that delegates to multiple observers.

    Useful for combining multiple observer behaviors.
    """

    def __init__(self):
        """Initialize composite observer."""
        self._observers: list[Observer] = []

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to the composite."""
        self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer from the composite."""
        if observer in self._observers:
            self._observers.remove(observer)

    def on_notify(self, event: Event) -> None:
        """Notify all child observers."""
        for observer in self._observers:
            try:
                observer.on_notify(event)
            except Exception as e:
                print(f"[CompositeObserver] Error in observer: {e}")

    def get_priority(self) -> int:
        """Composite has high priority to ensure all observers are notified."""
        return 100


class LoggingObserver(Observer):
    """
    Observer that logs all events.

    Useful for debugging and audit trails.
    """

    def __init__(self, log_file: Optional[str] = None,
                 include_timestamp: bool = True):
        """
        Initialize logging observer.

        Args:
            log_file: Optional file to write logs to
            include_timestamp: Whether to include timestamps
        """
        self._log_file = log_file
        self._include_timestamp = include_timestamp
        self._events: list[Event] = []

    def on_notify(self, event: Event) -> None:
        """Handle and log event."""
        self._events.append(event)

        timestamp = ""
        if self._include_timestamp:
            timestamp = f"[{event.timestamp.isoformat()}] "

        log_message = f"{timestamp}{event.name}: {event.data}"
        print(f"[Log] {log_message}")

        if self._log_file:
            try:
                with open(self._log_file, 'a') as f:
                    f.write(log_message + '\n')
            except Exception as e:
                print(f"[LoggingObserver] Error writing to file: {e}")

    def get_events(self) -> list[Event]:
        """Get all logged events."""
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear event history."""
        self._events.clear()

    def get_events_by_name(self, name: str) -> list[Event]:
        """Get all events with a specific name."""
        return [e for e in self._events if e.name == name]


# Convenience functions for creating observers
def create_progress_observer(progress_callback: Callable[[str], None],
                            status_callback: Callable[[str, bool], None]) -> ProgressObserver:
    """Create a progress observer with callbacks."""
    return ProgressObserver(progress_callback, status_callback)


def create_ui_observer(ui_callback: Callable[[str, Any], None]) -> UIStateObserver:
    """Create a UI state observer with callback."""
    return UIStateObserver(ui_callback)


def create_document_count_observer(count_callback: Callable[[int], None]) -> DocumentCountObserver:
    """Create a document count observer with callback."""
    return DocumentCountObserver(count_callback)


def create_generation_observer(token_callback: Callable[[str], None],
                               complete_callback: Callable[[dict], None]) -> GenerationObserver:
    """Create a generation observer with callbacks."""
    return GenerationObserver(token_callback, complete_callback)
