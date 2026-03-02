"""
Design Patterns Module for Local RAG Assistant.

This module provides base implementations of common design patterns:
- Singleton: Ensure only one instance of a class exists
- Observer: Event notification system
- Strategy: Interchangeable algorithms
- Factory: Centralized object creation
- Command: Encapsulated actions
"""

from .singleton import Singleton, SingletonMeta
from .observer import Observer, Observable, Event
from .strategy import Strategy, StrategyContext
from .factory import Factory, FactoryItem
from .command import Command, CommandInvoker, CommandResult

__all__ = [
    'Singleton',
    'SingletonMeta',
    'Observer',
    'Observable',
    'Event',
    'Strategy',
    'StrategyContext',
    'Factory',
    'FactoryItem',
    'Command',
    'CommandInvoker',
    'CommandResult',
]
