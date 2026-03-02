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
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Represents an event in the observer pattern.

    Attributes:
        name: Event name/type
        data: Event payload data
        source: The object that emitted the event
        timestamp: When the event was created
        priority: Event priority for ordering
    """
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
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

    def get_priority(self) -> EventPriority:
        """Return the priority of this observer (higher = notified first)."""
        return EventPriority.NORMAL


class CallableObserver(Observer):
    """
    Observer that wraps a callable function.

    Usage:
        def handler(event):
            print(event.data)

        observer = CallableObserver('my_event', handler)
    """

    def __init__(self, event_name: str, callback: Callable[[Event], None],
                 priority: EventPriority = EventPriority.NORMAL):
        self.event_name = event_name
        self.callback = callback
        self._priority = priority

    def on_notify(self, event: Event) -> None:
        if event.name == self.event_name:
            self.callback(event)

    def get_priority(self) -> EventPriority:
        return self._priority


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
                self._observers.sort(key=lambda o: o.get_priority().value, reverse=True)

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

    def get_event_history(self) -> List[Event]:
        """Get the history of events (thread-safe copy)."""
        with self._lock:
            return self._event_history.copy()


class EventBus:
    """
    Global event bus for application-wide event distribution.

    Implements the Observer pattern with named event channels.

    Usage:
        bus = EventBus()
        bus.subscribe('progress', lambda e: print(e.data))
        bus.publish('progress', {'percent': 50})
    """

    _instance: Optional['EventBus'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._initialized = True

    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event."""
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event."""
        with self._lock:
            if event_name in self._subscribers:
                if callback in self._subscribers[event_name]:
                    self._subscribers[event_name].remove(callback)

    def publish(self, event_name: str, data: Dict[str, Any] = None,
                source: Any = None) -> None:
        """Publish an event to all subscribers."""
        event = Event(name=event_name, data=data or {}, source=source)

        with self._lock:
            subscribers = self._subscribers.get(event_name, []).copy()

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Error in subscriber: {e}")

    def clear(self) -> None:
        """Clear all subscribers."""
        with self._lock:
            self._subscribers.clear()

    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Get the global EventBus instance."""
        return cls()


class ProgressEvent:
    """Helper class for progress events."""

    START = 'progress.start'
    UPDATE = 'progress.update'
    COMPLETE = 'progress.complete'
    ERROR = 'progress.error'

    @staticmethod
    def create(stage: str, percent: float, message: str = "") -> Event:
        """Create a progress event."""
        return Event(
            name=stage,
            data={
                'percent': percent,
                'message': message,
            }
        )


class StateEvent:
    """Helper class for state change events."""

    CHANGED = 'state.changed'
    LOADING = 'state.loading'
    READY = 'state.ready'
    ERROR = 'state.error'

    @staticmethod
    def create(state: str, data: Dict[str, Any] = None) -> Event:
        """Create a state event."""
        return Event(
            name=StateEvent.CHANGED,
            data={'state': state, 'data': data or {}}
        )
