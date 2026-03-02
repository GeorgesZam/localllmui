import os
import re
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
        self._max_history = 3
        self.code_executor = None
        self.code_execution_enabled = config.CODE_EXECUTION_ENABLED

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
                n_gpu_layers=getattr(config, 'GPU_LAYERS', -1),
                n_batch=512,
                use_mmap=True,
                use_mlock=False,
                verbose=False
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

    def _build_prompt(self, message: str, rag_context: str = "", enable_code: bool = False) -> str:
        if enable_code:
            system = config.CODE_EXECUTION_SYSTEM_PROMPT
        elif rag_context:
            system = f"""{config.SYSTEM_PROMPT}

=== CONTEXT ===
{rag_context}
=== END ===

Answer based on context. If not found, say so."""
        else:
            system = config.SYSTEM_PROMPT

        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"

        for h in self.history[-self._max_history:]:
            prompt += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def generate(self, message: str) -> Iterator[str]:
        if not self.is_ready:
            yield "Error: Model not ready"
            return

        # Check if code execution needed
        if config.CODE_EXECUTION_AUTO_DETECT and self.code_execution_enabled:
            try:
                from code_executor import CodeDetector
                needs_code, _ = CodeDetector.detect_code_request(message)

                if needs_code:
                    yield from self.execute_code_generation(message)
                    return
            except ImportError:
                pass  # code_executor not available, use normal path

        rag_context = ""
        sources = []

        if config.RAG_ENABLED and self.rag.documents:
            rag_context, sources = self.rag.search(message)

        prompt = self._build_prompt(message, rag_context)

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
            sources_text = self.rag.format_sources_for_display()
            yield sources_text
            full_response += sources_text

        if full_response.strip():
            clean = full_response.split("📚 Sources:")[0].strip()
            self.history.append({"user": message, "assistant": clean})

            if len(self.history) > self._max_history * 2:
                self.history = self.history[-self._max_history:]

    def execute_code_generation(self, message: str) -> Iterator[str]:
        """Generate and execute code for user requests."""
        try:
            from code_executor import SandboxedCodeExecutor, CodeDetector
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
            print(f"[LLM] Code generation error: {e}")
            yield f"\n[Error: {e}]"
            return

        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(full_response)

        if not code_blocks:
            yield "\n\nNo executable code found in response."
            return

        # Execute each code block
        executor = SandboxedCodeExecutor(
            timeout=config.CODE_EXECUTION_TIMEOUT,
            max_memory_mb=config.CODE_EXECUTION_MAX_MEMORY_MB
        )

        for i, code in enumerate(code_blocks, 1):
            yield f"\n\n⚡ Executing code block {i}/{len(code_blocks)}...\n"

            result = executor.execute(code)

            # Format and yield result
            yield self._format_execution_result(result)

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

    def clear_history(self):
        self.history = []
        print("[LLM] History cleared")
