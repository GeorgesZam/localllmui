"""
LLM Engine - handles model loading and text generation.
"""

import os
from typing import Iterator, Optional, Callable

import config
from utils import get_resource_path
from rag import RAG


class LLMEngine:
    """LLM Engine with RAG integration."""
    
    def __init__(self):
        self.llm = None
        self.rag = RAG()
        self.is_ready = False
        self.error = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Load the LLM model and initialize RAG."""
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
    
    def _build_prompt(self, message: str, rag_context: str = "", history: list = None) -> str:
        """Build prompt with system message, history, and RAG context."""
        history = history or []
        
        if rag_context:
            system = f"""{config.SYSTEM_PROMPT}

=== CONTEXT DOCUMENTS ===
{rag_context}
=== END CONTEXT ===

Answer based on the context above. If not found, say so."""
        else:
            system = config.SYSTEM_PROMPT
        
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        # Add history (last 3 exchanges)
        for msg in history[-6:]:  # 6 messages = 3 exchanges
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def generate(self, message: str, history: list = None) -> Iterator[str]:
        """Generate response with streaming."""
        if not self.is_ready:
            yield "Error: Model not ready"
            return
        
        # Get RAG context
        rag_context = ""
        sources = []
        
        if config.RAG_ENABLED and self.rag.documents:
            print(f"[LLM] Searching {len(self.rag.documents)} documents...")
            rag_context, sources = self.rag.search(message)
            print(f"[LLM] Found {len(sources)} sources")
        
        prompt = self._build_prompt(message, rag_context, history)
        print(f"[LLM] Prompt: {len(prompt)} chars")
        
        full_response = ""
        
        try:
            for chunk in self.llm(
                prompt,
                max_tokens=config.MAX_TOKENS,
                stop=config.STOP_TOKENS,
                temperature=getattr(config, 'TEMPERATURE', 0.2),
                top_p=getattr(config, 'TOP_P', 0.9),
                repeat_penalty=getattr(config, 'REPEAT_PENALTY', 1.1),
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                full_response += token
                yield token
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            yield f"\n[Error: {e}]"
        
        if getattr(config, 'RAG_SHOW_SOURCES', True) and sources:
            sources_text = self.rag.format_sources_for_display()
            yield sources_text
