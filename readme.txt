
# Local LLM UI - Architecture Overview

## Current Architecture

```mermaid
graph TB
    subgraph "UI Layer"
        UI[ModernRAGApp<br/>CustomTkinter Interface]
        Chat[Chat Frame]
        Input[Input Area]
        Sidebar[Document Sidebar]
    end

    subgraph "Processing Layer"
        DP[DocumentProcessor<br/>PDF/DOCX/TXT/CSV/XLSX]
        RAG[RAGEngine<br/>RAG Orchestration]
        VS[VectorStore<br/>BGE-small-en Embeddings]
        TS[TextSplitter<br/>LangChain Chunking]
    end

    subgraph "AI Layer"
        LLM[LLM Engine<br/>Qwen2.5:0.5b GGUF]
        EM[Embedding Model<br/>BGE-small-en-v1.5]
    end

    subgraph "Data Layer"
        Docs[(Document Files)]
        Models[(Model Files)]
    end

    UI --> DP
    UI --> RAG
    UI --> LLM

    DP --> Docs
    RAG --> VS
    RAG --> TS
    RAG --> EM
    RAG --> LLM

    VS --> EM
    LLM --> Models

    style UI fill:#E6A23C
    style LLM fill:#67C23A
    style RAG fill:#409EFF
    style VS fill:#909399
```

## Proposed Code Sandbox Integration

```mermaid
graph TB
    subgraph "UI Layer"
        UI[ModernRAGApp<br/>CustomTkinter Interface]
        Chat[Chat Frame]
        Input[Input Area]
        Sidebar[Document Sidebar]
    end

    subgraph "Processing Layer"
        DP[DocumentProcessor<br/>PDF/DOCX/TXT/CSV/XLSX]
        RAG[RAGEngine<br/>RAG Orchestration]
        VS[VectorStore<br/>BGE-small-en Embeddings]
        TS[TextSplitter<br/>LangChain Chunking]
    end

    subgraph "AI Layer"
        LLM[LLM Engine<br/>Qwen2.5:0.5b GGUF]
        EM[Embedding Model<br/>BGE-small-en-v1.5]
    end

    subgraph "Code Execution Sandbox - NEW"
        CS[CodeSandbox<br/>Python Execution]
        PD[Pyodide/Docker<br/>Isolation Layer]
        VM[Virtual Machine<br/>Resource Limits]
        FR[File Restrictions<br/>No Network/FS Access]
        TO[Timeout Manager<br/>Max Execution Time]
        MO[Memory Monitor<br/>RAM Limits]
    end

    subgraph "Data Layer"
        Docs[(Document Files)]
        Models[(Model Files)]
        Code[(Generated Code)]
        Results[(Execution Results)]
    end

    UI --> DP
    UI --> RAG
    UI --> LLM
    UI --> CS

    DP --> Docs
    RAG --> VS
    RAG --> TS
    RAG --> EM
    RAG --> LLM

    VS --> EM
    LLM --> Models

    %% Sandbox integration
    LLM -.->|Code Generation| CS
    CS --> PD
    PD --> VM
    VM --> FR
    VM --> TO
    VM --> MO

    CS --> Code
    CS --> Results
    Results -.->|Output| UI

    style UI fill:#E6A23C
    style LLM fill:#67C23A
    style RAG fill:#409EFF
    style VS fill:#909399
    style CS fill:#F56C6C
    style PD fill:#E6A23C
    style VM fill:#67C23A
```

## Sandbox Architecture Details

### Integration Points

1. **LLM → Sandbox**: When LLM generates code blocks
   - Detect code blocks in response (```python)
   - Extract code for execution
   - Pass to sandbox with context

2. **Sandbox → UI**: Display execution results
   - Standard output capture
   - Error handling and formatting
   - Result visualization (charts, dataframes)

3. **User → Sandbox**: Direct code execution requests
   - `/run` command in chat
   - Code execution toggle in UI
   - Manual code submission

### Sandbox Implementation Options

```mermaid
graph LR
    subgraph "Option 1: Pyodide (Browser-based)"
        PY1[Pyodide<br/>Python in WASM]
        BR1[Browser Environment]
    end

    subgraph "Option 2: Docker Container"
        DK[Docker Container<br/>python:slim]
        VM2[Isolated Process]
    end

    subgraph "Option 3: Restricted Subprocess"
        SP[subprocess.Popen<br/>with restrictions]
        LM[Resource Limits<br/>rlimit]
    end

    subgraph "Option 4: PyPy Sandbox"
        PP[pypy-sandbox<br/>Restricted Python]
        ST[Secure Translations]
    end

    Input[Code Input] --> PY1
    Input --> DK
    Input --> SP
    Input --> PP

    PY1 --> Out1[Output]
    DK --> Out2[Output]
    SP --> Out3[Output]
    PP --> Out4[Output]

    style PY1 fill:#409EFF
    style DK fill:#67C23A
    style SP fill:#E6A23C
    style PP fill:#F56C6C
```

### Recommended Architecture: Restricted Subprocess

```python
# Proposed structure: src/sandbox.py

class CodeSandbox:
    """
    Secure code execution sandbox for Python code
    Integrates with LLM to execute generated code
    """

    def __init__(self):
        self.timeout = 30  # seconds
        self.memory_limit = 256  # MB
        self.allowed_modules = [
            'math', 'random', 'datetime', 'json',
            're', 'collections', 'itertools', 'statistics'
        ]
        self.blocked_modules = [
            'os', 'sys', 'subprocess', 'socket',
            'urllib', 'requests', 'pickle', 'shutil'
        ]

    def execute(self, code: str, context: dict = None) -> SandboxResult:
        """
        Execute code in isolated environment

        Args:
            code: Python code to execute
            context: Optional variables to inject

        Returns:
            SandboxResult with output, errors, execution time
        """
        pass

    def execute_with_timeout(self, code: str) -> SandboxResult:
        """Execute with timeout protection"""
        pass

    def sanitize_code(self, code: str) -> str:
        """Remove dangerous operations"""
        pass
```

### Security Layers

```mermaid
graph TB
    subgraph "Layer 1: Code Analysis"
        CA[AST Parser<br/>Detect dangerous patterns]
        BL[Blocked Operations<br/>import os, __import__, eval]
    end

    subgraph "Layer 2: Process Isolation"
        PI[subprocess.Popen<br/>separate process]
        NS[Namespace restrictions<br/>limited builtins]
    end

    subgraph "Layer 3: Resource Limits"
        RL[rlimit<br/>CPU, Memory, File size]
        TO[Timeout<br/>signal.SIGALRM]
    end

    subgraph "Layer 4: Network/Filesystem"
        NF[Network disabled<br/>no socket access]
        FS[Filesystem sandbox<br/>temp directory only]
    end

    Code[User/LLM Code] --> CA
    CA --> BL
    BL --> PI
    PI --> NS
    NS --> RL
    RL --> TO
    TO --> NF
    NF --> FS
    FS --> Output[Safe Execution]

    style CA fill:#E6A23C
    style PI fill:#67C23A
    style RL fill:#409EFF
    style NF fill:#F56C6C
```

### Usage Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant LLM
    participant Sandbox
    participant Process

    User->>UI: Ask question requiring code
    UI->>LLM: Send query
    LLM->>LLM: Generate response with code
    LLM->>Sandbox: Extract code block
    Sandbox->>Sandbox: Validate & sanitize
    Sandbox->>Process: Execute in isolated env
    Process->>Sandbox: Return output/errors
    Sandbox->>UI: Format results
    UI->>User: Display code + results
```

### Configuration Options

```python
# Add to config.py

# === SANDBOX ===
SANDBOX_ENABLED = True
SANDBOX_TIMEOUT = 30  # seconds
SANDBOX_MEMORY_LIMIT = 256  # MB
SANDBOX_MAX_OUTPUT = 10000  # characters
SANDBOX_ALLOW_NETWORK = False
SANDBOX_ALLOW_FILESYSTEM = False
SANDBOX_TEMP_DIR = "/tmp/localllm_sandbox"
SANDBOX_ALLOWED_MODULES = [
    'math', 'random', 'datetime', 'json',
    're', 'collections', 'statistics'
]
```

### UI Enhancements

```mermaid
graph TB
    subgraph "Chat Interface Enhancements"
        CB[Code Block Detection]
        RE[Run Button]
        EXP[Expand/Collapse]
        OUT[Output Panel]
        ERR[Error Display]
    end

    subgraph "Settings Panel"
        SE[Sandbox Toggle]
        TO[Timeout Slider]
        ME[Memory Limit]
        AM[Allowed Modules]
    end

    Chat[Chat Message] --> CB
    CB --> RE
    RE --> OUT
    RE --> ERR

    Settings[Settings] --> SE
    SE --> TO
    SE --> ME
    SE --> AM

    style CB fill:#409EFF
    style RE fill:#67C23A
    style OUT fill:#E6A23C
    style ERR fill:#F56C6C
```

## Implementation Priority

1. **Phase 1**: Basic sandbox with subprocess isolation
2. **Phase 2**: Code validation and sanitization
3. **Phase 3**: Resource limits (timeout, memory)
4. **Phase 4**: UI integration (run button, output display)
5. **Phase 5**: Advanced features (visualization, file handling)

## File Structure

```
localllmui/
├── src/
│   ├── main.py          # UI and main app
│   ├── llm.py           # LLM engine
│   ├── sandbox.py       # NEW: Code execution sandbox
│   ├── rag.py           # RAG implementation
│   ├── config.py        # Configuration
│   └── utils.py         # Utilities
├── skills/
│   ├── skill_code.md    # NEW: Code execution skill
│   ├── skill_docx.md
│   ├── skill_pdf.md
│   ├── skill_ocr.md
│   ├── skill_rag.md
│   └── skill_summary.md
└── readme.txt           # This file
```

---

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python src/main.py`
4. Load documents and ask questions!

## Tech Stack

- **LLM**: Qwen2.5:0.5b (local, quantized)
- **Embeddings**: BGE-small-en-v1.5
- **UI**: CustomTkinter (modern dark theme)
- **RAG**: Custom implementation with LangChain text splitters
- **Sandbox**: Python subprocess with resource limits
