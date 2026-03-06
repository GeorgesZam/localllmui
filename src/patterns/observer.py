"""
Observer Pattern Implementation.

Defines a one-to-many dependency between objects so that when one object
changes state, all its dependents are notified and updated automatically.

Usage:
    class ProgressObserver(Observer):
        def on_notify(self, event):
            print(f"Progress: {event.data['percent']}%")

    observable = Observable()
    observable.attach(ProgressObserver())
    observable.notify(Event('progress', {'percent': 50}))
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """
    Represents an event in the observer pattern.

    Attributes:
        name: Event name/type
        data: Event payload data
        source: The object that emitted the event
        timestamp: When the event was created
    """

    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    propagation_stopped: bool = False

    def stop_propagation(self) -> None:
        """Stop event propagation to remaining observers."""
        self.propagation_stopped = True

    def __str__(self) -> str:
        return f"Event({self.name}, data={self.data})"


class Observer:
    """
    Base observer interface.

    Observers can be notified of events from Observable objects.
    """

    def on_notify(self, event: Event) -> None:
        """
        Handle notification of an event.

        Args:
            event: The event being notified
        """
        raise NotImplementedError("Subclasses must implement on_notify()")

    def get_priority(self) -> int:
        """Return the priority of this observer (higher = notified first)."""
        return 1


class Observable:
    """
    Base class for objects that can be observed.

    Maintains a list of observers and notifies them of events.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._observers: List[Observer] = []
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = 100

    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to this observable.

        Args:
            observer: The observer to attach
        """
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
                # Sort by priority (higher first)
                self._observers.sort(key=lambda o: o.get_priority(), reverse=True)

    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from this observable.

        Args:
            observer: The observer to detach
        """
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def notify(self, event: Event) -> None:
        """
        Notify all attached observers of an event.

        Args:
            event: The event to notify observers about
        """
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Notify observers
            for observer in self._observers:
                if event.propagation_stopped:
                    break
                try:
                    observer.on_notify(event)
                except Exception as e:
                    print(f"[Observable] Error in observer: {e}")

    def detach_all(self) -> None:
        """Detach all observers."""
        with self._lock:
            self._observers.clear()

    def observer_count(self) -> int:
        """Return the number of attached observers."""
        with self._lock:
            return len(self._observers)


class ProgressEvent:
    """Helper class for progress events."""

    START = "progress.start"
    UPDATE = "progress.update"
    COMPLETE = "progress.complete"
    ERROR = "progress.error"

    @staticmethod
    def create(stage: str, percent: float, message: str = "") -> Event:
        """Create a progress event."""
        return Event(
            name=stage,
            data={
                "percent": percent,
                "message": message,
            },
        )


class StateEvent:
    """Helper class for state change events."""

    CHANGED = "state.changed"
    LOADING = "state.loading"
    READY = "state.ready"
    ERROR = "state.error"

    @staticmethod
    def create(state: str, data: Dict[str, Any] = None) -> Event:
        """Create a state event."""
        return Event(name=StateEvent.CHANGED, data={"state": state, "data": data or {}})
