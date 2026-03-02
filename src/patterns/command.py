"""
Command Pattern Implementation.

Encapsulates a request as an object, thereby allowing for parameterization
of clients with different requests, queuing of requests, and logging of operations.

Usage:
    class SaveCommand(Command):
        def execute(self):
            # save logic
            return CommandResult.success()

        def undo(self):
            # undo logic
            return CommandResult.success()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime


class CommandStatus(Enum):
    """Status of a command execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    UNDONE = "undone"


@dataclass
class CommandResult:
    """
    Result of a command execution.

    Attributes:
        success: Whether the command succeeded
        message: Human-readable message
        data: Optional result data
        error: Optional error information
        status: Current status
        timestamp: When the result was created
    """
    success: bool
    message: str = ""
    data: Any = None
    error: Optional[str] = None
    status: CommandStatus = CommandStatus.SUCCESS
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def success_result(cls, message: str = "Success", data: Any = None) -> 'CommandResult':
        """Create a successful result."""
        return cls(success=True, message=message, data=data, status=CommandStatus.SUCCESS)

    @classmethod
    def failure_result(cls, message: str, error: Optional[str] = None) -> 'CommandResult':
        """Create a failed result."""
        return cls(success=False, message=message, error=error, status=CommandStatus.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'error': self.error,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat()
        }


class Command(ABC):
    """
    Base class for commands.

    Encapsulates an action that can be executed and optionally undone.
    """

    def __init__(self):
        self._executed = False
        self._result: Optional[CommandResult] = None

    @abstractmethod
    def execute(self) -> CommandResult:
        """
        Execute the command.

        Returns:
            CommandResult with execution outcome
        """
        pass

    def undo(self) -> CommandResult:
        """
        Undo the command.

        Returns:
            CommandResult with undo outcome

        Raises:
            NotImplementedError: If command cannot be undone
        """
        raise NotImplementedError("Command cannot be undone")

    def can_undo(self) -> bool:
        """Check if command can be undone."""
        return True

    def is_executed(self) -> bool:
        """Check if command has been executed."""
        return self._executed

    def get_result(self) -> Optional[CommandResult]:
        """Get the result of the last execution."""
        return self._result

    def validate(self) -> bool:
        """
        Validate command before execution.

        Returns:
            True if command is valid
        """
        return True


class CommandInvoker:
    """
    Invoker that executes commands.

    Supports command history, undo/redo, and command queuing.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._history: List[Command] = []
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._lock = threading.RLock()
        self._max_history = 100
        self._before_hooks: List[Callable[[Command], None]] = []
        self._after_hooks: List[Callable[[Command, CommandResult], None]] = []

    def execute(self, command: Command) -> CommandResult:
        """
        Execute a command.

        Args:
            command: The command to execute

        Returns:
            CommandResult from execution
        """
        with self._lock:
            # Before hooks
            for hook in self._before_hooks:
                try:
                    hook(command)
                except Exception as e:
                    print(f"[CommandInvoker] Error in before hook: {e}")

            # Validate
            if not command.validate():
                result = CommandResult.failure_result("Command validation failed")
                return result

            # Execute
            command._executed = True
            result = command.execute()
            command._result = result

            # Add to history
            self._history.append(command)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Add to undo stack if command can be undone
            if command.can_undo():
                self._undo_stack.append(command)
                self._redo_stack.clear()  # Clear redo stack on new command

            # After hooks
            for hook in self._after_hooks:
                try:
                    hook(command, result)
                except Exception as e:
                    print(f"[CommandInvoker] Error in after hook: {e}")

            return result

    def undo(self) -> Optional[CommandResult]:
        """
        Undo the last command.

        Returns:
            CommandResult from undo, or None if nothing to undo
        """
        with self._lock:
            if not self._undo_stack:
                return None

            command = self._undo_stack.pop()
            if command.can_undo():
                result = command.undo()
                self._redo_stack.append(command)
                return result

            return None

    def redo(self) -> Optional[CommandResult]:
        """
        Redo the last undone command.

        Returns:
            CommandResult from redo, or None if nothing to redo
        """
        with self._lock:
            if not self._redo_stack:
                return None

            command = self._redo_stack.pop()
            result = command.execute()
            self._undo_stack.append(command)
            return result

    def can_undo(self) -> bool:
        """Check if there are commands to undo."""
        with self._lock:
            return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if there are commands to redo."""
        with self._lock:
            return len(self._redo_stack) > 0

    def clear_history(self) -> None:
        """Clear all command history."""
        with self._lock:
            self._history.clear()
            self._undo_stack.clear()
            self._redo_stack.clear()

    def get_history(self) -> List[Command]:
        """Get the command history."""
        with self._lock:
            return self._history.copy()

    def add_before_hook(self, hook: Callable[[Command], None]) -> None:
        """Add a hook to run before command execution."""
        with self._lock:
            self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[Command, CommandResult], None]) -> None:
        """Add a hook to run after command execution."""
        with self._lock:
            self._after_hooks.append(hook)

    def remove_before_hooks(self) -> None:
        """Remove all before hooks."""
        with self._lock:
            self._before_hooks.clear()

    def remove_after_hooks(self) -> None:
        """Remove all after hooks."""
        with self._lock:
            self._after_hooks.clear()


class MacroCommand(Command):
    """
    Command that executes multiple commands in sequence.

    All commands are executed, and failures are tracked.
    """

    def __init__(self, commands: List[Command], stop_on_failure: bool = False):
        """
        Initialize with a list of commands.

        Args:
            commands: List of commands to execute
            stop_on_failure: Whether to stop on first failure
        """
        super().__init__()
        self._commands = commands
        self._stop_on_failure = stop_on_failure
        self._results: List[CommandResult] = []

    def execute(self) -> CommandResult:
        """Execute all commands in sequence."""
        self._results.clear()
        all_success = True
        messages = []

        for i, command in enumerate(self._commands):
            result = command.execute()
            self._results.append(result)

            if not result.success:
                all_success = False
                messages.append(f"Command {i}: {result.message}")
                if self._stop_on_failure:
                    break
            else:
                messages.append(f"Command {i}: {result.message}")

        return CommandResult(
            success=all_success,
            message="; ".join(messages),
            data=self._results,
            status=CommandStatus.SUCCESS if all_success else CommandStatus.FAILED
        )

    def undo(self) -> CommandResult:
        """Undo all commands in reverse order."""
        # Undo in reverse order
        undone = []
        for command in reversed(self._commands):
            if command.can_undo():
                result = command.undo()
                undone.append(result)

        all_success = all(r.success for r in undone)
        return CommandResult(
            success=all_success,
            message=f"Undone {len(undone)} commands",
            status=CommandStatus.SUCCESS if all_success else CommandStatus.FAILED
        )

    def can_undo(self) -> bool:
        """Check if all commands can be undone."""
        return all(cmd.can_undo() for cmd in self._commands)


class FunctionCommand(Command):
    """
    Command that wraps a callable function.

    Useful for ad-hoc commands without creating a class.
    """

    def __init__(self, func: Callable[..., Any],
                 undo_func: Optional[Callable[..., Any]] = None,
                 *args, **kwargs):
        """
        Initialize with a function.

        Args:
            func: Function to execute
            undo_func: Optional undo function
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        super().__init__()
        self._func = func
        self._undo_func = undo_func
        self._args = args
        self._kwargs = kwargs
        self._undo_args = []
        self._undo_kwargs = {}

    def execute(self) -> CommandResult:
        """Execute the wrapped function."""
        try:
            result = self._func(*self._args, **self._kwargs)
            return CommandResult.success_result(
                message="Function executed successfully",
                data=result
            )
        except Exception as e:
            return CommandResult.failure_result(
                message="Function execution failed",
                error=str(e)
            )

    def undo(self) -> CommandResult:
        """Undo using the undo function if provided."""
        if self._undo_func is None:
            raise NotImplementedError("No undo function provided")

        try:
            result = self._undo_func(*self._undo_args, **self._undo_kwargs)
            return CommandResult.success_result(
                message="Undo successful",
                data=result
            )
        except Exception as e:
            return CommandResult.failure_result(
                message="Undo failed",
                error=str(e)
            )

    def can_undo(self) -> bool:
        """Check if undo function is available."""
        return self._undo_func is not None

    def set_undo_args(self, *args, **kwargs) -> None:
        """Set arguments for the undo function."""
        self._undo_args = args
        self._undo_kwargs = kwargs


class AsyncCommand(Command):
    """
    Base class for asynchronous commands.

    Override execute_async instead of execute.
    """

    @abstractmethod
    async def execute_async(self) -> CommandResult:
        """Execute the command asynchronously."""
        pass

    def execute(self) -> CommandResult:
        """
        Synchronous wrapper - raises error by default.

        Subclasses should override if they want to support both.
        """
        raise NotImplementedError("Use execute_async for async commands")

    async def undo_async(self) -> CommandResult:
        """Undo the command asynchronously."""
        raise NotImplementedError("Async undo not implemented")
