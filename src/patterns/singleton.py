"""
Singleton Pattern Implementation.

Ensures a class has only one instance and provides a global point of access to it.

Usage:
    @Singleton
    class MySingleton:
        pass

    # Or using metaclass
    class MySingleton(metaclass=SingletonMeta):
        pass
"""

import threading
from typing import Dict, Any, Type, Optional


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


def Singleton(cls: Type) -> Type:
    """
    Class decorator for implementing Singleton pattern.

    Usage:
        @Singleton
        class MyClass:
            pass

        obj1 = MyClass()
        obj2 = MyClass()
        assert obj1 is obj2  # True
    """
    original_new = cls.__new__

    def __new__(singleton_cls, *args, **kwargs):
        if not hasattr(singleton_cls, '_instance'):
            singleton_cls._instance = original_new(singleton_cls, *args, **kwargs)
        return singleton_cls._instance

    cls.__new__ = __new__

    # Add reset method for testing
    def reset_instance(singleton_cls) -> None:
        if hasattr(singleton_cls, '_instance'):
            delattr(singleton_cls, '_instance')

    cls.reset = classmethod(reset_instance)

    # Add is_initialized method
    def is_initialized(singleton_cls) -> bool:
        return hasattr(singleton_cls, '_instance')

    cls.is_initialized = classmethod(is_initialized)

    return cls


class LazySingleton:
    """
    Base class for lazy-loaded singletons.

    The instance is created only when first accessed.

    Usage:
        class MySingleton(LazySingleton):
            def __init__(self):
                self.value = 42

        instance = MySingleton.get_instance()
    """

    _instance: Optional['LazySingleton'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'LazySingleton':
        """Get the singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton instance exists."""
        return cls._instance is not None


class ThreadSafeSingleton:
    """
    Thread-safe singleton using instance locking.

    Provides instance-level locking for thread-safe operations.
    """

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            with cls._lock:
                if not hasattr(cls, '_instance'):
                    cls._instance = super().__new__(cls)
                    cls._instance._lock = threading.RLock()
        return cls._instance

    _lock = threading.Lock()

    def with_lock(self, func):
        """Execute a function with the instance lock held."""
        def wrapper(*args, **kwargs):
            with self._lock:
                return func(*args, **kwargs)
        return wrapper
