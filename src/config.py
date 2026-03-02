"""
Configuration Management with Singleton Pattern.

This module provides centralized configuration management using the
Singleton pattern to ensure consistent access to settings across
the application.
"""

import os
import multiprocessing
from typing import Any, Dict, Optional
from pathlib import Path

from patterns import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):
    """
    Singleton configuration manager.

    Provides centralized access to all application settings
    with thread-safe initialization and runtime updates.
    """

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        # Application
        self.app_name = "Local Chat"
        self.window_size = "1100x700"

        # Model Configuration
        self.model_file = "models/model.gguf"
        self.embedding_model_folder = "models/embedding_model"

        # CPU Configuration
        self._cpu_count = multiprocessing.cpu_count()
        self.context_size = 2048
        self.max_tokens = 384
        self.threads = max(4, self._cpu_count - 2)
        self.gpu_layers = -1

        # Prompts
        self.system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have this information in the provided documents."
Be concise and specific. Quote relevant parts when possible.
Answer in the same language as the user."""

        self.stop_tokens = ["<|im_end|>", "<end_of_turn>"]

        # RAG Configuration
        self.rag_enabled = True
        self.rag_folder = "data"
        self.rag_chunk_size = 384
        self.rag_chunk_overlap = 50
        self.rag_top_k = 3
        self.rag_min_score = 0.3
        self.rag_show_sources = True

        # Generation Parameters
        self.temperature = 0.1
        self.top_p = 0.85
        self.repeat_penalty = 1.15

        # Performance
        self.batch_size = 512
        self.lazy_load_embedding = True
        self.index_cache_enabled = True

        # Code Execution
        self.code_execution_enabled = True
        self.code_execution_timeout = 30
        self.code_execution_max_memory_mb = 512
        self.code_execution_auto_detect = True
        self.code_execution_prompt_save = True

        self.code_execution_system_prompt = """You are a helpful assistant with code execution capabilities.

You can WRITE and EXECUTE Python code to solve problems.

When you need to create documents, analyze data, or generate files:
1. Write clear, well-commented Python code
2. Wrap code in triple backticks with 'python' tag
3. Use available libraries (see below)
4. Print progress messages so user knows what's happening

Available libraries:
- python-docx: Create Word documents (.docx)
- python-pptx: Create PowerPoint presentations (.pptx)
- openpyxl: Create Excel spreadsheets (.xlsx)
- reportlab: Create PDF documents (.pdf)
- pandas: Data analysis, CSV/Excel processing
- matplotlib: Charts, graphs, visualizations
- json, csv: Standard file formats

Keep code simple, safe, and focused on the task.
Handle errors gracefully.

Answer in the same language as the user."""

        self._initialized = True

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get the singleton instance."""
        return cls()

    # Convenience properties for backward compatibility
    @property
    def APP_NAME(self) -> str:
        return self.app_name

    @property
    def WINDOW_SIZE(self) -> str:
        return self.window_size

    @property
    def MODEL_FILE(self) -> str:
        return self.model_file

    @property
    def EMBEDDING_MODEL_FOLDER(self) -> str:
        return self.embedding_model_folder

    @property
    def CONTEXT_SIZE(self) -> int:
        return self.context_size

    @property
    def MAX_TOKENS(self) -> int:
        return self.max_tokens

    @property
    def THREADS(self) -> int:
        return self.threads

    @property
    def GPU_LAYERS(self) -> int:
        return self.gpu_layers

    @property
    def SYSTEM_PROMPT(self) -> str:
        return self.system_prompt

    @property
    def STOP_TOKENS(self) -> list:
        return self.stop_tokens

    @property
    def RAG_ENABLED(self) -> bool:
        return self.rag_enabled

    @property
    def RAG_FOLDER(self) -> str:
        return self.rag_folder

    @property
    def RAG_CHUNK_SIZE(self) -> int:
        return self.rag_chunk_size

    @property
    def RAG_CHUNK_OVERLAP(self) -> int:
        return self.rag_chunk_overlap

    @property
    def RAG_TOP_K(self) -> int:
        return self.rag_top_k

    @property
    def RAG_MIN_SCORE(self) -> float:
        return self.rag_min_score

    @property
    def RAG_SHOW_SOURCES(self) -> bool:
        return self.rag_show_sources

    @property
    def TEMPERATURE(self) -> float:
        return self.temperature

    @property
    def TOP_P(self) -> float:
        return self.top_p

    @property
    def REPEAT_PENALTY(self) -> float:
        return self.repeat_penalty

    @property
    def BATCH_SIZE(self) -> int:
        return self.batch_size

    @property
    def LAZY_LOAD_EMBEDDING(self) -> bool:
        return self.lazy_load_embedding

    @property
    def INDEX_CACHE_ENABLED(self) -> bool:
        return self.index_cache_enabled

    @property
    def CODE_EXECUTION_ENABLED(self) -> bool:
        return self.code_execution_enabled

    @property
    def CODE_EXECUTION_TIMEOUT(self) -> int:
        return self.code_execution_timeout

    @property
    def CODE_EXECUTION_MAX_MEMORY_MB(self) -> int:
        return self.code_execution_max_memory_mb

    @property
    def CODE_EXECUTION_AUTO_DETECT(self) -> bool:
        return self.code_execution_auto_detect

    @property
    def CODE_EXECUTION_PROMPT_SAVE(self) -> bool:
        return self.code_execution_prompt_save

    @property
    def CODE_EXECUTION_SYSTEM_PROMPT(self) -> str:
        return self.code_execution_system_prompt

    def update(self, settings: Dict[str, Any]) -> None:
        """
        Update configuration values.

        Args:
            settings: Dictionary of setting names to values
        """
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    def load_from_file(self, filepath: str) -> bool:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.update(data)
            return True
        except Exception as e:
            print(f"[Config] Error loading from file: {e}")
            return False

    def save_to_file(self, filepath: str) -> bool:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save config

        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"[Config] Error saving to file: {e}")
            return False


# Global convenience instance
config = ConfigManager.get_instance()


# Backward compatibility: expose module-level attributes
def __getattr__(name: str) -> Any:
    """Provide backward compatibility for module-level attribute access."""
    return getattr(config, name)


# CPU_COUNT as a constant
_CPU_COUNT = multiprocessing.cpu_count()
