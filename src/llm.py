"""
LLM Engine with Singleton and Observer Patterns.

This module provides thread-safe singleton access to the LLM engine
with observable progress reporting during model loading and generation.
"""

import os
import re
import threading
from typing import Iterator, Optional, Callable
from dataclasses import dataclass

from patterns import SingletonMeta, Observable, Event, StateEvent
from config import ConfigManager
from utils import get_resource_path
from rag import RAG


@dataclass
class GenerationStats:
    """Statistics for a generation run."""
    tokens_generated: int = 0
    time_elapsed: float = 0.0
    tokens_per_second: float = 0.0
    has_code_execution: bool = False
    rag_results_used: int = 0


class LLMEngine(Observable, metaclass=SingletonMeta):
    """
    Singleton LLM engine with observable progress reporting.

    Features:
    - Thread-safe singleton pattern
    - Observable loading and generation progress
    - Code execution integration
    - RAG integration
    - Generation statistics tracking

    Usage:
        engine = LLMEngine.get_instance()
        engine.attach(my_observer)
        engine.load()
        for token in engine.generate("Hello"):
            print(token, end='')
    """

    def __init__(self):
        # Only initialize once (Singleton pattern)
        if hasattr(self, '_initialized'):
            return

        # Initialize Observable base class
        super().__init__()

        # Configuration
        self._config = ConfigManager.get_instance()

        # Core components
        self.llm = None
        self.rag: Optional[RAG] = None

        # State
        self.is_ready = False
        self.error: Optional[str] = None
        self._max_history = 3
        self.history: list = []

        # Code execution
        self.code_executor = None
        self.code_execution_enabled = self._config.CODE_EXECUTION_ENABLED

        # Skills integration
        self.skills_executor = None
        self._enabled_skills_content = ""

        # Statistics
        self._current_stats = GenerationStats()

        # Stop flag for generation cancellation
        self._stop_requested = threading.Event()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> 'LLMEngine':
        """Get the singleton LLM engine instance."""
        return cls()

    def stop_generation(self):
        """Request to stop the current generation."""
        self._stop_requested.set()

    def reset_stop_flag(self):
        """Reset stop flag for new generation."""
        self._stop_requested.clear()

    def load(self, on_progress: Optional[Callable[[str], None]] = None, model_path: Optional[str] = None) -> bool:
        """
        Load the LLM model and initialize RAG.

        Args:
            on_progress: Optional callback for progress updates
            model_path: Optional path to model file (overrides config)

        Returns:
            True if loading successful
        """
        def log(msg: str) -> None:
            """Log message and notify observers."""
            print(f"[LLM] {msg}")
            if on_progress:
                on_progress(msg)
            self.notify(StateEvent.create(
                StateEvent.LOADING,
                {'message': msg}
            ))

        try:
            # Initialize RAG if not already initialized
            if self.rag is None:
                # Notify loading started
                self.notify(StateEvent.create(
                    StateEvent.LOADING,
                    {'stage': 'rag_init'}
                ))

                # Initialize RAG
                self.rag = RAG()
                self.rag.initialize(log)

                # Notify RAG loaded
                self.notify(StateEvent.create(
                    StateEvent.LOADING,
                    {'stage': 'rag_complete', 'documents': len(self.rag.documents)}
                ))
            else:
                log("RAG already initialized, skipping...")

            # Import llama_cpp
            log("Importing llama_cpp...")
            from llama_cpp import Llama

            # Use provided path or config
            if model_path:
                # Convert to absolute path if relative
                if not os.path.isabs(model_path):
                    actual_model_path = os.path.abspath(model_path)
                else:
                    actual_model_path = model_path
            else:
                actual_model_path = get_resource_path(self._config.MODEL_FILE)

            log(f"Model: {actual_model_path}")

            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(f"Model not found: {actual_model_path}")

            # Unload previous model if exists
            if self.llm is not None:
                log("Unloading previous model...")
                try:
                    # Properly cleanup llama_cpp model
                    # The model needs to be explicitly closed to free mmap and GPU resources
                    if hasattr(self.llm, 'close'):
                        self.llm.close()
                    # Also try __exit__ for context manager cleanup
                    if hasattr(self.llm, '__exit__'):
                        self.llm.__exit__(None, None, None)
                except Exception as cleanup_error:
                    log(f"Warning during cleanup: {cleanup_error}")
                finally:
                    # Always delete and set to None
                    try:
                        del self.llm
                    except:
                        pass  # Ignore errors during cleanup
                    self.llm = None
                    self.is_ready = False

                # Force garbage collection to free memory immediately
                import gc
                gc.collect()
                log("Previous model unloaded")

            # Load model
            log("Loading model...")
            self.notify(StateEvent.create(
                StateEvent.LOADING,
                {'stage': 'model_loading'}
            ))

            self.llm = Llama(
                model_path=actual_model_path,
                n_ctx=self._config.CONTEXT_SIZE,
                n_threads=self._config.THREADS,
                n_gpu_layers=self._config.GPU_LAYERS,
                n_batch=512,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )

            # Clear history when switching models
            self.history = []

            # Reset stop flag - ensure clean state for new model
            self._stop_requested.clear()

            self.is_ready = True
            log("Ready!")

            # Notify ready state
            self.notify(StateEvent.create(
                StateEvent.READY,
                {'model': actual_model_path}
            ))

            return True

        except Exception as e:
            self.error = str(e)
            error_str = str(e)

            # Provide more helpful error messages for common issues
            if "corrupted" in error_str.lower() or "incomplete" in error_str.lower() or "not within the file bounds" in error_str.lower():
                error_str = f"Model file is corrupted or incomplete. Please re-download the model.\nOriginal error: {error_str}"
            elif "failed to load model from file" in error_str.lower():
                error_str = f"Failed to load model. The file may be corrupted or incompatible.\nOriginal error: {error_str}"

            log(f"Error: {error_str}")

            # Notify error state
            self.notify(StateEvent.create(
                StateEvent.ERROR,
                {'error': error_str}
            ))

            import traceback
            traceback.print_exc()
            return False

    def set_skills_executor(self, skills_executor):
        """Set the skills executor for prompt enhancement."""
        self.skills_executor = skills_executor

    def update_skills_content(self, skills_manager):
        """Update the skills content from the skills manager."""
        if self.skills_executor and skills_manager:
            base_prompt = self._config.SYSTEM_PROMPT
            enhanced_prompt, skill_names = self.skills_executor.apply_skills_to_prompt(
                "", base_prompt
            )
            self._enabled_skills_content = enhanced_prompt.replace(base_prompt, "").strip()
        else:
            self._enabled_skills_content = ""

    def _build_prompt(self, message: str, rag_context: str = "",
                      enable_code: bool = False) -> str:
        """Build prompt for the LLM."""
        # Build base system prompt
        base_system = self._config.SYSTEM_PROMPT

        # Add skills content if available
        if self._enabled_skills_content:
            system = f"{base_system}\n\n{self._enabled_skills_content}"
        else:
            system = base_system

        # Modify based on mode
        if enable_code:
            system = self._config.CODE_EXECUTION_SYSTEM_PROMPT
            if self._enabled_skills_content:
                system = f"{self._config.CODE_EXECUTION_SYSTEM_PROMPT}\n\n{self._enabled_skills_content}"
        elif rag_context:
            system = f"""{system}

=== CONTEXT ===
{rag_context}
=== END ===

Answer based on context. If not found, say so."""

        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"

        for h in self.history[-self._max_history:]:
            prompt += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def generate(self, message: str, allowed_document_sources: list = None) -> Iterator[str]:
        """
        Generate response for the given message.

        Args:
            message: User message
            allowed_document_sources: Optional list of document filenames to filter RAG results (for conversation isolation)

        Yields:
            Generated tokens
        """
        if not self.is_ready:
            yield "Error: Model not ready"
            return

        # Reset stats
        self._current_stats = GenerationStats()

        # Check if code execution needed
        if (self._config.CODE_EXECUTION_AUTO_DETECT and
                self.code_execution_enabled):
            try:
                from code_executor import CodeDetector
                needs_code, _ = CodeDetector.detect_code_request(message)

                if needs_code:
                    self._current_stats.has_code_execution = True
                    yield from self.execute_code_generation(message)
                    return
            except ImportError:
                pass  # code_executor not available, use normal path

        # Normal RAG generation
        rag_context = ""
        sources = []

        if self._config.RAG_ENABLED and self.rag and self.rag.documents:
            rag_context, sources = self.rag.search(message, allowed_sources=allowed_document_sources)
            self._current_stats.rag_results_used = len(sources)

        prompt = self._build_prompt(message, rag_context)

        full_response = ""

        try:
            for chunk in self.llm(
                prompt,
                max_tokens=self._config.MAX_TOKENS,
                stop=self._config.STOP_TOKENS,
                temperature=self._config.TEMPERATURE,
                top_p=self._config.TOP_P,
                repeat_penalty=self._config.REPEAT_PENALTY,
                stream=True
            ):
                # Check if stop was requested
                if self._stop_requested.is_set():
                    break

                token = chunk["choices"][0]["text"]
                full_response += token
                self._current_stats.tokens_generated += 1
                yield token
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            yield f"\n[Error: {e}]"

        if self._config.RAG_SHOW_SOURCES and sources:
            sources_text = self.rag.format_sources_for_display()
            yield sources_text
            full_response += sources_text

        if full_response.strip():
            clean = full_response.split("📚 Sources:")[0].strip()
            self.history.append({"user": message, "assistant": clean})

            if len(self.history) > self._max_history * 2:
                self.history = self.history[-self._max_history:]

        # Auto-detect code in response if not already in code execution mode
        if not self._current_stats.has_code_execution and self.code_execution_enabled:
            try:
                from code_executor import CodeDetector
                has_code, language, code_content = CodeDetector.detect_code_in_response(full_response)
                if has_code and language in ['python', 'py', 'javascript', 'js', '']:
                    # Notify that code was detected and can be executed
                    self.notify(Event('code_detected', {
                        'language': language,
                        'code': code_content,
                        'full_response': full_response
                    }))
            except ImportError:
                pass

        # Notify generation complete
        self.notify(Event('generation_complete', {
            'tokens': self._current_stats.tokens_generated,
            'has_code': self._current_stats.has_code_execution,
            'rag_results': self._current_stats.rag_results_used
        }))

    def execute_code_generation(self, message: str) -> Iterator[str]:
        """
        Generate and execute code for user requests.

        Args:
            message: User message requesting code execution

        Yields:
            Generated tokens and execution results
        """
        try:
            from code_executor import (
                EnhancedSandboxedCodeExecutor,
                CodeDetector,
                ResourceLimits
            )
        except ImportError:
            yield "Error: Code execution module not available."
            return

        # Build code-enabled prompt
        prompt = self._build_prompt(message, enable_code=True)

        # Generate code from LLM
        full_response = ""

        try:
            for chunk in self.llm(
                prompt,
                max_tokens=self._config.MAX_TOKENS,
                stop=self._config.STOP_TOKENS,
                temperature=self._config.TEMPERATURE,
                top_p=self._config.TOP_P,
                repeat_penalty=self._config.REPEAT_PENALTY,
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                full_response += token
                yield token
        except Exception as e:
            print(f"[LLM] Code generation error: {e}")
            yield f"\n[Error: {e}]"
            return

        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(full_response)

        if not code_blocks:
            yield "\n\nNo executable code found in response."
            return

        # Create executor with resource limits
        resource_limits = ResourceLimits(
            max_cpu_time=self._config.CODE_EXECUTION_TIMEOUT,
            max_memory_mb=self._config.CODE_EXECUTION_MAX_MEMORY_MB,
            allow_network=False
        )

        executor = EnhancedSandboxedCodeExecutor(resource_limits)
        self.code_executor = executor  # Store for UI access

        # Execute each code block
        for i, code in enumerate(code_blocks, 1):
            yield f"\n\n⚡ Executing code block {i}/{len(code_blocks)}...\n"

            result = executor.execute(code)

            # Format and yield result
            yield self._format_execution_result(result)

        # Notify about files ready for download
        if executor.get_downloadable_files():
            self.notify(Event('code_execution_files_ready', {
                'files': [f.__dict__ for f in executor.get_downloadable_files()],
                'executor': executor
            }))

        # Store in history
        if full_response.strip():
            clean = full_response.strip()
            self.history.append({"user": message, "assistant": clean})
            if len(self.history) > self._max_history * 2:
                self.history = self.history[-self._max_history:]

    def _extract_code_blocks(self, text: str) -> list:
        """Extract Python code blocks from text."""
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _format_execution_result(self, result) -> str:
        """Format execution result for display."""
        lines = []

        if result.success:
            lines.append("\n✅ Execution successful!\n")

            if result.stdout:
                lines.append(f"📊 Output:\n{result.stdout}")

            if result.files_created:
                lines.append("📁 Files created:")
                for filepath in result.files_created:
                    filename = os.path.basename(filepath)
                    lines.append(f"  • {filename}")

            lines.append(f"\n⏱️ Execution time: {result.execution_time:.2f}s")
        else:
            lines.append("\n❌ Execution failed!")
            if result.error:
                lines.append(f"🚫 Error: {result.error}")
            if result.stderr:
                lines.append(f"📋 Details:\n{result.stderr}")

        return "\n".join(lines)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        print("[LLM] History cleared")
        self.notify(Event('history_cleared', {}))

    def get_stats(self) -> GenerationStats:
        """Get statistics from the last generation."""
        return self._current_stats

    def reset(self) -> None:
        """Reset the engine state (useful for testing)."""
        self.history = []
        self.error = None
        self._current_stats = GenerationStats()
        self.notify(Event('engine_reset', {}))

    def switch_model(self, model_path: str, model_id: str, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """
        Switch to a different model.

        Args:
            model_path: Path to the new model file
            model_id: ID of the new model
            on_progress: Optional callback for progress updates

        Returns:
            True if successful
        """
        # Update config
        self._config.model_file = model_path
        self._config.model_id = model_id

        # Reload with new model
        return self.load(on_progress=on_progress, model_path=model_path)

    def get_current_model_id(self) -> str:
        """Get the current model ID."""
        return getattr(self._config, 'model_id', 'unknown')


# Global convenience instance
def get_llm_engine() -> LLMEngine:
    """Get the global LLM engine instance."""
    return LLMEngine.get_instance()
