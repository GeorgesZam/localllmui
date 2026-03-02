"""
Factory Pattern Implementation.

Defines an interface for creating objects but lets subclasses decide
which classes to instantiate.

Usage:
    factory = Factory()
    factory.register('pdf', PDFParser)
    parser = factory.create('pdf', file_path='doc.pdf')
"""

from abc import ABC, abstractmethod
from typing import Type, Dict, Any, TypeVar, Optional, Callable
import threading


T = TypeVar('T')


class FactoryItem:
    """
    Represents an item that can be created by a factory.

    Wraps a class with optional builder function and metadata.
    """

    def __init__(self,
                 item_class: Type[T],
                 builder: Optional[Callable[..., T]] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a factory item.

        Args:
            item_class: The class to instantiate
            builder: Optional builder function
            metadata: Optional metadata about the item
        """
        self.item_class = item_class
        self.builder = builder
        self.metadata = metadata or {}

    def create(self, *args, **kwargs) -> T:
        """Create an instance of the item."""
        if self.builder:
            return self.builder(*args, **kwargs)
        return self.item_class(*args, **kwargs)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)


class Factory:
    """
    Generic factory for creating objects.

    Supports registration, creation, and lifecycle management.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._items: Dict[str, FactoryItem] = {}
        self._lock = threading.RLock()
        self._default_item: Optional[str] = None
        self._creation_history: list = []
        self._max_history = 100

    def register(self, name: str, item_class: Type[T],
                 builder: Optional[Callable[..., T]] = None,
                 metadata: Dict[str, Any] = None,
                 set_as_default: bool = False) -> None:
        """
        Register a class with the factory.

        Args:
            name: Name to register the class under
            item_class: The class to register
            builder: Optional builder function
            metadata: Optional metadata
            set_as_default: Whether to set as default item
        """
        with self._lock:
            self._items[name] = FactoryItem(item_class, builder, metadata)
            if set_as_default or self._default_item is None:
                self._default_item = name

    def unregister(self, name: str) -> None:
        """
        Unregister a class from the factory.

        Args:
            name: Name of the class to unregister
        """
        with self._lock:
            if name in self._items:
                del self._items[name]
                if self._default_item == name:
                    self._default_item = next(iter(self._items), None)

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a registered class.

        Args:
            name: Name of the class to create
            *args: Arguments to pass to the class constructor
            **kwargs: Keyword arguments to pass to the class constructor

        Returns:
            Instance of the requested class

        Raises:
            KeyError: If the name is not registered
        """
        with self._lock:
            if name not in self._items:
                raise KeyError(f"Item '{name}' not registered in factory")

            # Record in history
            self._creation_history.append({
                'name': name,
                'args': str(args)[:100],
            })
            if len(self._creation_history) > self._max_history:
                self._creation_history.pop(0)

            return self._items[name].create(*args, **kwargs)

    def create_or_default(self, name: Optional[str], *args, **kwargs) -> Any:
        """
        Create an instance, using default if name is None.

        Args:
            name: Name of the class to create, or None for default
            *args: Arguments to pass to the class constructor
            **kwargs: Keyword arguments to pass to the class constructor

        Returns:
            Instance of the requested class

        Raises:
            KeyError: If no default is set and name is None
        """
        if name is None:
            name = self._default_item
        if name is None:
            raise KeyError("No default item set and no name provided")
        return self.create(name, *args, **kwargs)

    def set_default(self, name: str) -> None:
        """
        Set the default item name.

        Args:
            name: Name of the item to set as default
        """
        with self._lock:
            if name not in self._items:
                raise KeyError(f"Item '{name}' not registered")
            self._default_item = name

    def get_default(self) -> Optional[str]:
        """Get the default item name."""
        return self._default_item

    def is_registered(self, name: str) -> bool:
        """Check if an item is registered."""
        with self._lock:
            return name in self._items

    def get_registered_names(self) -> list:
        """Get list of all registered item names."""
        with self._lock:
            return list(self._items.keys())

    def get_metadata(self, name: str, key: str, default: Any = None) -> Any:
        """Get metadata for a registered item."""
        with self._lock:
            if name in self._items:
                return self._items[name].get_metadata(key, default)
        return default

    def clear(self) -> None:
        """Clear all registered items."""
        with self._lock:
            self._items.clear()
            self._default_item = None

    def get_creation_history(self) -> list:
        """Get history of object creation."""
        with self._lock:
            return self._creation_history.copy()

    def clear_history(self) -> None:
        """Clear the creation history."""
        with self._lock:
            self._creation_history.clear()


class AbstractFactory(ABC):
    """
    Abstract factory interface for creating families of related objects.

    Use this when you have multiple related objects that need to be
    created together.
    """

    @abstractmethod
    def create_product_a(self) -> Any:
        """Create a product of type A."""
        pass

    @abstractmethod
    def create_product_b(self) -> Any:
        """Create a product of type B."""
        pass


class Builder:
    """
    Builder pattern implementation for complex object construction.

    Separates construction from representation.
    """

    def __init__(self):
        self._parts: Dict[str, Any] = {}

    def reset(self) -> 'Builder':
        """Reset the builder state."""
        self._parts.clear()
        return self

    def set(self, key: str, value: Any) -> 'Builder':
        """Set a part of the object being built."""
        self._parts[key] = value
        return self

    def build(self, product_class: Type[T]) -> T:
        """
        Build the final product.

        Args:
            product_class: The class to instantiate

        Returns:
            Instance of the product class
        """
        return product_class(**self._parts)

    def get_parts(self) -> Dict[str, Any]:
        """Get all parts set so far."""
        return self._parts.copy()


class SingletonFactory(Factory):
    """
    Factory that creates and manages singleton instances.

    Each name gets exactly one instance that is reused.
    """

    def __init__(self):
        super().__init__()
        self._instances: Dict[str, Any] = {}

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Get or create a singleton instance.

        Args:
            name: Name of the item
            *args: Arguments (ignored if instance exists)
            **kwargs: Keyword arguments (ignored if instance exists)

        Returns:
            Singleton instance
        """
        with self._lock:
            if name not in self._instances:
                self._instances[name] = super().create(name, *args, **kwargs)
            return self._instances[name]

    def reset_instance(self, name: str) -> None:
        """Reset a specific instance (useful for testing)."""
        with self._lock:
            if name in self._instances:
                del self._instances[name]

    def reset_all(self) -> None:
        """Reset all instances."""
        with self._lock:
            self._instances.clear()


class LazyFactory(Factory):
    """
    Factory that supports lazy initialization.

    Items are registered with a factory function that is only
    called when the item is first created.
    """

    def register_lazy(self, name: str, factory_func: Callable[[], T],
                      metadata: Dict[str, Any] = None) -> None:
        """
        Register a lazy factory function.

        Args:
            name: Name to register under
            factory_func: Function that creates the item
            metadata: Optional metadata
        """
        self.register(name, object, builder=lambda: factory_func(), metadata=metadata)

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance using the registered factory function.

        Args and kwargs are passed to the factory function if it accepts them.
        """
        with self._lock:
            if name not in self._items:
                raise KeyError(f"Item '{name}' not registered in factory")
            return self._items[name].create(*args, **kwargs)
