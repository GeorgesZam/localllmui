"""
LLM Engine - handles model loading and text generation.
"""

import os
from typing import Iterator, Optional, Callable

import config
from utils import get_resource_path
from rag import RAG


class LLMEngine:
    def __init__(self):
        self.llm = None
        self.history = []
        self.rag = RAG()
        self.is_ready = False
        self.error = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        def log(msg):
            print(f"[LLM] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            self.rag.initialize(log)
            
            log("Importing llama_cpp...")
            from llama_cpp import Llama
            
            model_path = get_resource_path(config.MODEL_FILE)
            log(f"Model: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            log("Loading model...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=config.CONTEXT_SIZE,
                n_threads=config.THREADS,
                n_gpu_layers=-1,
                verbose=True
            )
            
            self.is_ready = True
            log("Ready!")
            return True
            
        except Exception as e:
            self.error = str(e)
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_prompt(self, message: str, rag_context: str = "") -> str:
        if rag_context:
            system = f"""{config.SYSTEM_PROMPT}

=== CONTEXT DOCUMENTS ===
{rag_context}
=== END CONTEXT ===

Answer based on the context above. If not found, say so."""
        else:
            system = config.SYSTEM_PROMPT
        
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        for h in self.history[-3:]:
            prompt += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def generate(self, message: str) -> Iterator[str]:
        if not self.is_ready:
            yield "Error: Model not ready"
            return
        
        rag_context = ""
        sources = []
        
        if config.RAG_ENABLED and self.rag.documents:
            print(f"[LLM] Searching {len(self.rag.documents)} documents...")
            rag_context, sources = self.rag.search(message)
            print(f"[LLM] Found {len(sources)} sources")
        
        prompt = self._build_prompt(message, rag_context)
        print(f"[LLM] Prompt: {len(prompt)} chars")
        
        full_response = ""
        
        try:
            for chunk in self.llm(
                prompt,
                max_tokens=config.MAX_TOKENS,
                stop=config.STOP_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repeat_penalty=config.REPEAT_PENALTY,
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                full_response += token
                yield token
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            yield f"\n[Error: {e}]"
        
        if config.RAG_SHOW_SOURCES and sources:
            # Only show sources with meaningful relevance (0.60+ for embeddings, 2+ for keyword)
            display_threshold = 0.60 if self.rag.embeddings is not None else 2.0
            filtered_sources = [s for s in sources if s['score'] >= display_threshold]

            if filtered_sources:
                # Temporarily replace last_sources for display
                original_sources = self.rag.last_sources
                self.rag.last_sources = filtered_sources
                sources_text = self.rag.format_sources_for_display()
                self.rag.last_sources = original_sources

                yield sources_text
                full_response += sources_text
        
        if full_response.strip():
            clean = full_response.split("📚 Sources:")[0].strip()
            self.history.append({"user": message, "assistant": clean})
    
    def clear_history(self):
        self.history = []
        print("[LLM] History cleared")
