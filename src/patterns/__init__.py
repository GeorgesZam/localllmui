"""
Design Patterns Module for Local RAG Assistant.

This module provides base implementations of common design patterns:
- Singleton: Ensure only one instance of a class exists
- Observer: Event notification system
"""

from .observer import Event, Observable, Observer, ProgressEvent, StateEvent
from .singleton import SingletonMeta

__all__ = [
    "SingletonMeta",
    "Observer",
    "Observable",
    "Event",
    "StateEvent",
    "ProgressEvent",
]
