"""
Singleton Pattern Implementation.

Ensures a class has only one instance and provides a global point of access to it.

Usage:
    class MySingleton(metaclass=SingletonMeta):
        pass
"""

import threading
from typing import Any, Dict, Type


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.

    Uses double-checked locking for thread safety.
    """

    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking pattern
        if cls not in cls._instances:
            with cls._lock:
                # Check again in case another thread created it
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def reset(mcs, cls: Type) -> None:
        """Reset the singleton instance (useful for testing)."""
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]

    @classmethod
    def is_initialized(mcs, cls: Type) -> bool:
        """Check if a singleton instance exists."""
        return cls in mcs._instances
