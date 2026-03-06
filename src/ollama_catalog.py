"""
Ollama Model Catalog - Available models for Ollama backend.

This module defines the catalog of available Ollama models
with their metadata for pull/download operations.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelSize(Enum):
    """Model size categories."""
    TINY = "Tiny (< 1B)"
    SMALL = "Small (1B-3B)"
    MEDIUM = "Medium (3B-7B)"
    LARGE = "Large (7B-15B)"
    XLARGE = "X-Large (> 15B)"


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    id: str
    ollama_name: str  # Name for `ollama pull`
    name: str
    description: str
    size_category: ModelSize
    parameter_count: float
    file_size_mb: int
    context_size: int
    requires_ram_mb: int
    tags: List[str]
    family: str = ""

    def get_display_name(self) -> str:
        """Get formatted display name."""
        return f"{self.name} ({self.parameter_count}B)"

    def get_size_display(self) -> str:
        """Get formatted file size."""
        if self.file_size_mb >= 1000:
            return f"{self.file_size_mb / 1024:.1f} GB"
        return f"{self.file_size_mb} MB"

    def get_ram_display(self) -> str:
        """Get formatted RAM requirement."""
        if self.requires_ram_mb >= 1000:
            return f"{self.requires_ram_mb / 1024:.1f} GB"
        return f"{self.requires_ram_mb} MB"


# Ollama Model Catalog
OLLAMA_MODEL_CATALOG: List[OllamaModelInfo] = [
    # Qwen 2.5 - 0.5B (Default - fastest)
    OllamaModelInfo(
        id="qwen2.5-0.5b",
        ollama_name="qwen2.5:0.5b",
        name="Qwen 2.5",
        description="Compact yet capable model. Fastest option with good quality.",
        size_category=ModelSize.TINY,
        parameter_count=0.5,
        file_size_mb=400,
        context_size=32768,
        requires_ram_mb=1500,
        tags=["fast", "efficient", "multilingual", "default"],
        family="qwen"
    ),

    # Qwen 2.5 - 1.5B
    OllamaModelInfo(
        id="qwen2.5-1.5b",
        ollama_name="qwen2.5:1.5b",
        name="Qwen 2.5",
        description="Enhanced capabilities with better understanding and generation quality.",
        size_category=ModelSize.SMALL,
        parameter_count=1.5,
        file_size_mb=900,
        context_size=32768,
        requires_ram_mb=2500,
        tags=["balanced", "multilingual", "quality"],
        family="qwen"
    ),

    # Qwen 2.5 - 3B
    OllamaModelInfo(
        id="qwen2.5-3b",
        ollama_name="qwen2.5:3b",
        name="Qwen 2.5",
        description="Strong performance on complex tasks with larger context understanding.",
        size_category=ModelSize.SMALL,
        parameter_count=3,
        file_size_mb=1900,
        context_size=32768,
        requires_ram_mb=4000,
        tags=["quality", "complex-tasks", "reasoning"],
        family="qwen"
    ),

    # Qwen 2.5 - 7B
    OllamaModelInfo(
        id="qwen2.5-7b",
        ollama_name="qwen2.5:7b",
        name="Qwen 2.5",
        description="High-end model with excellent reasoning and knowledge across many domains.",
        size_category=ModelSize.MEDIUM,
        parameter_count=7,
        file_size_mb=4300,
        context_size=32768,
        requires_ram_mb=8000,
        tags=["high-quality", "knowledge", "reasoning"],
        family="qwen"
    ),

    # Llama 3.2 - 1B
    OllamaModelInfo(
        id="llama3.2-1b",
        ollama_name="llama3.2:1b",
        name="Llama 3.2",
        description="Meta's compact model with strong general capabilities and English proficiency.",
        size_category=ModelSize.SMALL,
        parameter_count=1,
        file_size_mb=700,
        context_size=131072,
        requires_ram_mb=2000,
        tags=["meta", "english", "general-purpose"],
        family="llama"
    ),

    # Llama 3.2 - 3B
    OllamaModelInfo(
        id="llama3.2-3b",
        ollama_name="llama3.2:3b",
        name="Llama 3.2",
        description="Enhanced Meta model with better performance on instruction following.",
        size_category=ModelSize.SMALL,
        parameter_count=3,
        file_size_mb=2000,
        context_size=131072,
        requires_ram_mb=4000,
        tags=["meta", "quality", "instruction-following"],
        family="llama"
    ),

    # Phi 3.5 - Mini
    OllamaModelInfo(
        id="phi3.5-mini",
        ollama_name="phi3.5:3.8b",
        name="Phi 3.5 Mini",
        description="Microsoft's compact model with strong reasoning and coding capabilities.",
        size_category=ModelSize.SMALL,
        parameter_count=3.8,
        file_size_mb=2400,
        context_size=128000,
        requires_ram_mb=5000,
        tags=["microsoft", "coding", "reasoning", "compact"],
        family="phi"
    ),

    # Phi 3 - Mini (older but stable)
    OllamaModelInfo(
        id="phi3-mini",
        ollama_name="phi3:mini",
        name="Phi 3 Mini",
        description="Microsoft's efficient model optimized for edge computing.",
        size_category=ModelSize.SMALL,
        parameter_count=3.8,
        file_size_mb=2300,
        context_size=12800,
        requires_ram_mb=4500,
        tags=["microsoft", "efficient", "edge"],
        family="phi"
    ),

    # Gemma 2 - 2B
    OllamaModelInfo(
        id="gemma2-2b",
        ollama_name="gemma2:2b",
        name="Gemma 2",
        description="Google's efficient model with good performance across various tasks.",
        size_category=ModelSize.SMALL,
        parameter_count=2,
        file_size_mb=1400,
        context_size=8192,
        requires_ram_mb=3000,
        tags=["google", "efficient", "general-purpose"],
        family="gemma"
    ),

    # DeepSeek Coder - 1.3B (for coding)
    OllamaModelInfo(
        id="deepseek-coder-1.3b",
        ollama_name="deepseek-coder:1.3b",
        name="DeepSeek Coder",
        description="Specialized model for code generation and programming tasks.",
        size_category=ModelSize.SMALL,
        parameter_count=1.3,
        file_size_mb=800,
        context_size=16384,
        requires_ram_mb=2000,
        tags=["coding", "programming", "specialized"],
        family="deepseek"
    ),

    # Mistral - 7B
    OllamaModelInfo(
        id="mistral-7b",
        ollama_name="mistral:7b",
        name="Mistral 7B",
        description="High-quality general purpose model with excellent instruction following.",
        size_category=ModelSize.MEDIUM,
        parameter_count=7,
        file_size_mb=4100,
        context_size=32768,
        requires_ram_mb=7500,
        tags=["quality", "general-purpose", "instruction"],
        family="mistral"
    ),
]


def get_ollama_model_by_id(model_id: str) -> Optional[OllamaModelInfo]:
    """Get Ollama model info by ID."""
    for model in OLLAMA_MODEL_CATALOG:
        if model.id == model_id:
            return model
    return None


def get_ollama_models_by_size(size: ModelSize) -> List[OllamaModelInfo]:
    """Get all models in a size category."""
    return [m for m in OLLAMA_MODEL_CATALOG if m.size_category == size]


def get_ollama_models_by_family(family: str) -> List[OllamaModelInfo]:
    """Get all models in a family."""
    return [m for m in OLLAMA_MODEL_CATALOG if m.family == family]


def get_ollama_models_by_tag(tag: str) -> List[OllamaModelInfo]:
    """Get all models with a specific tag."""
    return [m for m in OLLAMA_MODEL_CATALOG if tag in m.tags]


def get_recommended_ollama_models() -> List[OllamaModelInfo]:
    """Get recommended Ollama models for general use."""
    recommended = []

    # Smallest fast model (default)
    if model := get_ollama_model_by_id("qwen2.5-0.5b"):
        recommended.append(model)

    # Balanced model
    if model := get_ollama_model_by_id("qwen2.5-1.5b"):
        recommended.append(model)

    # Quality model
    if model := get_ollama_model_by_id("phi3.5-mini"):
        recommended.append(model)

    return recommended


def get_default_ollama_model() -> OllamaModelInfo:
    """Get the default Ollama model."""
    return get_ollama_model_by_id("qwen2.5-0.5b")


def get_installed_ollama_models() -> List[str]:
    """Get list of models currently installed in Ollama."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
    except Exception as e:
        print(f"[OllamaCatalog] Error getting installed models: {e}")
    return []


def is_ollama_model_installed(ollama_name: str) -> bool:
    """Check if an Ollama model is installed."""
    installed = get_installed_ollama_models()
    return ollama_name in installed
