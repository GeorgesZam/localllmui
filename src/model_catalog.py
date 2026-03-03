"""
Model Catalog - Available LLM models for download.

This module defines the catalog of available models with their metadata,
download URLs, and compatibility information.
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


class ModelFormat(Enum):
    """Model file formats."""
    GGUF = "GGUF"
    GGML = "GGML"


@dataclass
class ModelInfo:
    """Information about a model in the catalog."""
    id: str
    name: str
    description: str
    size_category: ModelSize
    parameter_count: int
    file_size_mb: int
    download_url: str
    filename: str
    format: ModelFormat
    quantization: str
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


# Model Catalog
MODEL_CATALOG: List[ModelInfo] = [
    # Qwen 3.5 - Latest (Smallest)
    ModelInfo(
        id="qwen3.5-0.8b",
        name="Qwen 3.5",
        description="Latest Qwen model with improved reasoning and multilingual support. Fast and efficient.",
        size_category=ModelSize.TINY,
        parameter_count=0.8,
        file_size_mb=480,
        download_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
        filename="qwen3.5-0.8b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=32768,
        requires_ram_mb=2048,
        tags=["latest", "fast", "multilingual", "reasoning"],
        family="qwen3.5"
    ),

    # Qwen 2.5 - Tiny (New default)
    ModelInfo(
        id="qwen2.5-0.5b",
        name="Qwen 2.5",
        description="Compact yet capable model. Good balance of speed and quality for everyday tasks.",
        size_category=ModelSize.TINY,
        parameter_count=0.5,
        file_size_mb=380,
        download_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
        filename="qwen2.5-0.5b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=32768,
        requires_ram_mb=1500,
        tags=["fast", "efficient", "multilingual"],
        family="qwen2.5"
    ),

    # Qwen 2.5 - 1.5B
    ModelInfo(
        id="qwen2.5-1.5b",
        name="Qwen 2.5",
        description="Enhanced capabilities with better understanding and generation quality.",
        size_category=ModelSize.SMALL,
        parameter_count=1.5,
        file_size_mb=920,
        download_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf",
        filename="qwen2.5-1.5b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=32768,
        requires_ram_mb=2500,
        tags=["balanced", "multilingual", "quality"],
        family="qwen2.5"
    ),

    # Qwen 2.5 - 3B
    ModelInfo(
        id="qwen2.5-3b",
        name="Qwen 2.5",
        description="Strong performance on complex tasks with larger context understanding.",
        size_category=ModelSize.SMALL,
        parameter_count=3,
        file_size_mb=1900,
        download_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf",
        filename="qwen2.5-3b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=32768,
        requires_ram_mb=4000,
        tags=["quality", "complex-tasks", "reasoning"],
        family="qwen2.5"
    ),

    # Qwen 2.5 - 7B
    ModelInfo(
        id="qwen2.5-7b",
        name="Qwen 2.5",
        description="High-end model with excellent reasoning and knowledge across many domains.",
        size_category=ModelSize.MEDIUM,
        parameter_count=7,
        file_size_mb=4300,
        download_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf",
        filename="qwen2.5-7b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=32768,
        requires_ram_mb=8000,
        tags=["high-quality", "knowledge", "reasoning"],
        family="qwen2.5"
    ),

    # Llama 3.2 - 1B
    ModelInfo(
        id="llama3.2-1b",
        name="Llama 3.2",
        description="Meta's compact model with strong general capabilities and English proficiency.",
        size_category=ModelSize.SMALL,
        parameter_count=1,
        file_size_mb=680,
        download_url="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1b-instruct-q4_0.gguf",
        filename="llama-3.2-1b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=131072,
        requires_ram_mb=2000,
        tags=["meta", "english", "general-purpose"],
        family="llama"
    ),

    # Llama 3.2 - 3B
    ModelInfo(
        id="llama3.2-3b",
        name="Llama 3.2",
        description="Enhanced Meta model with better performance on instruction following.",
        size_category=ModelSize.SMALL,
        parameter_count=3,
        file_size_mb=2000,
        download_url="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF/resolve/main/llama-3.2-3b-instruct-q4_0.gguf",
        filename="llama-3.2-3b-instruct-q4_0.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_0",
        context_size=131072,
        requires_ram_mb=4000,
        tags=["meta", "quality", "instruction-following"],
        family="llama"
    ),

    # Phi-3.5 Mini
    ModelInfo(
        id="phi3.5-mini",
        name="Phi 3.5 Mini",
        description="Microsoft's compact model with strong reasoning and coding capabilities.",
        size_category=ModelSize.SMALL,
        parameter_count=3.8,
        file_size_mb=2400,
        download_url="https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-q4.gguf",
        filename="phi-3.5-mini-instruct-q4.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4",
        context_size=128000,
        requires_ram_mb=5000,
        tags=["microsoft", "coding", "reasoning", "compact"],
        family="phi"
    ),

    # Gemma 2 - 2B
    ModelInfo(
        id="gemma2-2b",
        name="Gemma 2",
        description="Google's efficient model with good performance across various tasks.",
        size_category=ModelSize.SMALL,
        parameter_count=2,
        file_size_mb=1400,
        download_url="https://huggingface.co/gemma2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        format=ModelFormat.GGUF,
        quantization="Q4_K_M",
        context_size=8192,
        requires_ram_mb=3000,
        tags=["google", "efficient", "general-purpose"],
        family="gemma"
    ),
]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    for model in MODEL_CATALOG:
        if model.id == model_id:
            return model
    return None


def get_models_by_size(size: ModelSize) -> List[ModelInfo]:
    """Get all models in a size category."""
    return [m for m in MODEL_CATALOG if m.size_category == size]


def get_models_by_family(family: str) -> List[ModelInfo]:
    """Get all models in a family."""
    return [m for m in MODEL_CATALOG if m.family == family]


def get_models_by_tag(tag: str) -> List[ModelInfo]:
    """Get all models with a specific tag."""
    return [m for m in MODEL_CATALOG if tag in m.tags]


def get_recommended_models() -> List[ModelInfo]:
    """Get recommended models for general use."""
    recommended = []

    # Smallest fast model
    if model := get_model_by_id("qwen2.5-0.5b"):
        recommended.append(model)

    # Balanced model
    if model := get_model_by_id("qwen2.5-1.5b"):
        recommended.append(model)

    # Quality model
    if model := get_model_by_id("qwen2.5-3b"):
        recommended.append(model)

    return recommended


def get_default_model() -> ModelInfo:
    """Get the default model (Qwen 2.5 smallest as requested)."""
    return get_model_by_id("qwen2.5-0.5b")


def get_latest_model() -> ModelInfo:
    """Get the latest model (Qwen 3.5)."""
    return get_model_by_id("qwen3.5-0.8b")
