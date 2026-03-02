"""
Strategy Pattern Implementation.

Defines a family of algorithms, encapsulates each one, and makes them interchangeable.

Usage:
    class QuickSort(Strategy):
        def execute(self, data):
            return quick_sort(data)

    context = StrategyContext(QuickSort())
    result = context.execute([3, 1, 2])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import threading


class Strategy(ABC):
    """
    Base class for strategy implementations.

    A strategy defines an algorithm that can be used interchangeably.
    """

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the strategy algorithm.

        Args:
            *args: Strategy-specific arguments
            **kwargs: Strategy-specific keyword arguments

        Returns:
            Strategy-specific result
        """
        pass

    def get_name(self) -> str:
        """Return the name of this strategy."""
        return self.__class__.__name__

    def get_description(self) -> str:
        """Return a description of this strategy."""
        return self.__doc__ or "No description available"

    def validate_input(self, *args, **kwargs) -> bool:
        """
        Validate input before execution.

        Returns:
            True if input is valid, False otherwise
        """
        return True


class StrategyContext:
    """
    Context class that uses a strategy to execute algorithms.

    The context can switch strategies at runtime and delegates
    algorithm execution to the current strategy.
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        self._strategy = strategy
        self._lock = threading.RLock()
        self._execution_history: list = []
        self._max_history = 100

    def set_strategy(self, strategy: Strategy) -> None:
        """
        Set the strategy to use.

        Args:
            strategy: The strategy to set
        """
        with self._lock:
            self._strategy = strategy

    def get_strategy(self) -> Optional[Strategy]:
        """Get the current strategy."""
        return self._strategy

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the current strategy.

        Args:
            *args: Arguments to pass to the strategy
            **kwargs: Keyword arguments to pass to the strategy

        Returns:
            Result from the strategy execution

        Raises:
            RuntimeError: If no strategy is set
        """
        with self._lock:
            if self._strategy is None:
                raise RuntimeError("No strategy set")

            if not self._strategy.validate_input(*args, **kwargs):
                raise ValueError("Invalid input for strategy")

            result = self._strategy.execute(*args, **kwargs)

            # Record in history
            self._execution_history.append({
                'strategy': self._strategy.get_name(),
                'args': str(args)[:100],  # Truncate for storage
                'result': str(result)[:100]
            })
            if len(self._execution_history) > self._max_history:
                self._execution_history.pop(0)

            return result

    def get_execution_history(self) -> list:
        """Get the history of strategy executions."""
        with self._lock:
            return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear the execution history."""
        with self._lock:
            self._execution_history.clear()


class CompositeStrategy(Strategy):
    """
    Combines multiple strategies and executes them in sequence.

    Results can be combined using a result handler function.
    """

    def __init__(self, strategies: list[Strategy], combine_results=None):
        """
        Initialize with a list of strategies.

        Args:
            strategies: List of strategies to execute
            combine_results: Function to combine results (optional)
        """
        self._strategies = strategies
        self._combine_results = combine_results or self._default_combine

    def execute(self, *args, **kwargs) -> Any:
        """Execute all strategies in sequence."""
        results = []
        for strategy in self._strategies:
            result = strategy.execute(*args, **kwargs)
            results.append(result)
        return self._combine_results(results)

    def _default_combine(self, results: list) -> list:
        """Default combination method returns list of results."""
        return results

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to the composite."""
        self._strategies.append(strategy)

    def remove_strategy(self, strategy: Strategy) -> None:
        """Remove a strategy from the composite."""
        if strategy in self._strategies:
            self._strategies.remove(strategy)


class AdaptiveStrategy(Strategy):
    """
    Strategy that adapts its behavior based on input characteristics.

    Uses different internal strategies based on conditions.
    """

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._selector: Optional[callable] = None

    def register_strategy(self, name: str, strategy: Strategy,
                          condition: callable) -> None:
        """
        Register a strategy with a selection condition.

        Args:
            name: Strategy name
            strategy: The strategy instance
            condition: Function that returns True if this strategy should be used
        """
        self._strategies[name] = {
            'strategy': strategy,
            'condition': condition
        }

    def set_selector(self, selector: callable) -> None:
        """
        Set a custom selector function.

        Args:
            selector: Function that takes input and returns strategy name
        """
        self._selector = selector

    def execute(self, *args, **kwargs) -> Any:
        """Execute the best strategy based on input."""
        # Use custom selector if available
        if self._selector:
            strategy_name = self._selector(*args, **kwargs)
            if strategy_name in self._strategies:
                return self._strategies[strategy_name]['strategy'].execute(*args, **kwargs)

        # Otherwise, check conditions
        for name, config in self._strategies.items():
            if config['condition'](*args, **kwargs):
                return config['strategy'].execute(*args, **kwargs)

        raise RuntimeError("No suitable strategy found for the given input")


class CachedStrategy(Strategy):
    """
    Decorator strategy that caches results.

    Useful for expensive operations with repeated inputs.
    """

    def __init__(self, strategy: Strategy, max_size: int = 100):
        """
        Initialize with a strategy to wrap.

        Args:
            strategy: The strategy to wrap
            max_size: Maximum cache size
        """
        self._strategy = strategy
        self._cache: Dict = {}
        self._max_size = max_size
        self._lock = threading.RLock()

    def execute(self, *args, **kwargs) -> Any:
        """Execute with caching."""
        # Create cache key
        cache_key = (args, tuple(sorted(kwargs.items())))

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Execute strategy
        result = self._strategy.execute(*args, **kwargs)

        with self._lock:
            # Add to cache
            self._cache[cache_key] = result
            # Trim cache if needed
            if len(self._cache) > self._max_size:
                # Remove oldest entry (first)
                self._cache.pop(next(iter(self._cache)))

        return result

    def clear_cache(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def cache_size(self) -> int:
        """Return the current cache size."""
        return len(self._cache)


class RetryStrategy(Strategy):
    """
    Decorator strategy that retries execution on failure.

    Useful for flaky operations or network requests.
    """

    def __init__(self, strategy: Strategy, max_retries: int = 3,
                 backoff_factor: float = 1.0):
        """
        Initialize with a strategy to wrap.

        Args:
            strategy: The strategy to wrap
            max_retries: Maximum number of retries
            backoff_factor: Multiplier for delay between retries
        """
        self._strategy = strategy
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    def execute(self, *args, **kwargs) -> Any:
        """Execute with retry logic."""
        import time

        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._strategy.execute(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._backoff_factor * (2 ** attempt)
                    time.sleep(delay)

        raise last_exception
