"""
Design patterns and utility classes for Local RAG Application.
"""

class SingletonMeta(type):
    """
    Metaclass that implements the Singleton pattern.

    Ensures that only one instance of a class can exist.
    Thread-safe implementation for use in multi-threaded environments.
    """

    _instances = {}
    _lock = __import__('threading').Lock()

    def __call__(cls, *args, **kwargs):
        # Double-check locking pattern for thread safety
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class Observable:
    """
    Base class for objects that can be observed.

    Provides a simple observer pattern implementation for progress
    reporting and state changes.
    """

    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        """Add an observer."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer):
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event):
        """Notify all observers of an event."""
        for observer in self._observers:
            if hasattr(observer, 'on_event'):
                observer.on_event(event)


class StateEvent:
    """Event object for state changes."""

    READY = 'ready'
    LOADING = 'loading'
    ERROR = 'error'
    GENERATING = 'generating'

    @staticmethod
    def create(event_type: str, data: dict = None):
        """Create a state event."""
        return {
            'type': event_type,
            'data': data or {}
        }
