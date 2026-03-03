# LocalLLM UI

Chat UI local avec RAG et exécution de code.

## Architecture

```mermaid
graph TB
    subgraph ["Application"]
        App[App]
        UI[UI]
        LLM[LLM Engine]
        RAG[RAG System]
        Skills[Skills Manager]
        Conv[Conversation Manager]
        Code[Code Executor]
    end

    subgraph ["Data"]
        Models[Models]
        Docs[Documents]
    end

    UI -->|"User Message"| LLM
    LLM -->|"Search Context"| RAG
    LLM -->|"Get Current"| Conv
    LLM -->|"Apply Skills"| Skills
    LLM -->|"Execute Code"| Code
    RAG -->|"Index"| Docs
    LLM -->|"Load"| Models

    style App fill:#4a9eff
    style UI fill:#50fa7b
    style LLM fill:#ff79c6
    style RAG fill:#bd93f9
    style Skills fill:#9b59b6
    style Conv fill:#f1fa8c
    style Code fill:#ff5555
```

## Flux de message

```mermaid
sequenceDiagram
    actor U as User
    participant UI as ChatUI
    participant App as Main App
    participant Conv as ConversationManager
    participant LLM as LLM Engine
    participant RAG as RAG System

    U->>UI: Send message
    UI->>App: _on_send(text)
    App->>Conv: get_current()
    Conv-->>App: conversation
    App->>Conv: add_message("user", text)

    App->>LLM: generate(text, allowed_docs)
    LLM->>RAG: search(query, allowed_sources)

    alt Documents found
        RAG-->>LLM: context + sources
    else No documents
        RAG-->>LLM: empty
    end

    LLM->>LLM: build prompt
    LLM->>LLM: stream response
    LLM-->>App: response chunks
    App-->>UI: display chunks
    App->>Conv: add_message("assistant", response)
    UI-->>U: Show response
```

## Isolement des conversations

```mermaid
graph LR
    subgraph ["Conversation 1"]
        Conv1[Conversation 1]
        Docs1[doc1.txt, doc2.txt]
    end

    subgraph ["Conversation 2"]
        Conv2[Conversation 2]
        Docs2[doc3.txt]
    end

    subgraph ["RAG Global Index"]
        Index[All Documents Indexed]
    end

    subgraph ["Search Filtering"]
        Filter1[allowed_sources=<br/>["doc1.txt", "doc2.txt"]]
        Filter2[allowed_sources=<br/>["doc3.txt"]]
        Filter3[allowed_sources=[]]
    end

    Docs1 --> Index
    Docs2 --> Index

    Conv1 --> Filter1
    Conv2 --> Filter2
    New[New Conversation] --> Filter3

    Filter1 -->|"Results only from<br/>doc1.txt, doc2.txt"| Index
    Filter2 -->|"Results only from<br/>doc3.txt"| Index
    Filter3 -->|"No results"| Index

    style Conv1 fill:#50fa7b
    style Conv2 fill:#9b59b6
    style New fill:#6272a4
    style Index fill:#f1fa8c
```

## Relations entre classes

```mermaid
classDiagram
    class App {
        +llm: LLMEngine
        +conv_manager: ConversationManager
        +skills_manager: SkillsManager
    }

    class ChatUI {
        +on_send: Callable
        +on_load_files: Callable
        +on_new_chat: Callable
    }

    class LLMEngine {
        +llm: Llama
        +rag: RAG
        +history: list
        +generate(message, allowed_docs)
    }

    class RAG {
        +documents: list
        +embedding_model: EmbeddingModel
        +search(query, allowed_sources)
    }

    class ConversationManager {
        +conversations: Dict
        +current_id: str
        +create_conversation()
        +add_document(filename)
    }

    class Conversation {
        +id: str
        +title: str
        +messages: list
        +document_ids: list
    }

    App *-- LLMEngine
    App *-- ConversationManager
    App *-- SkillsManager
    App *-- ChatUI
    LLMEngine *-- RAG
    ConversationManager *-- Conversation
```

## Modules

### Core

| Module | Role |
|--------|------|
| `main` | Entry |
| `ui` | View |
| `llm` | AI |
| `rag` | Docs |

### Data

| Module | Role |
|--------|------|
| `conversations` | Chat |
| `model_manager` | Models |
| `model_catalog` | List |

### Features

| Module | Role |
|--------|------|
| `skills_manager` | Skills |
| `code_executor` | Code |
| `ocr` | Images |

## Patterns

```mermaid
graph LR
    Singleton[Singleton Pattern]
    Observer[Observer Pattern]
    Strategy[Strategy Pattern]
    Factory[Factory Pattern]

    LLM[LLM Engine]
    Events[Loading Events]
    Parsers[Document Parsers]
    Skills[Skills Factory]

    Singleton -->|"One instance"| LLM
    Observer -->|"Notify progress"| Events
    Strategy -->|"Swap algorithms"| Parsers
    Factory -->|"Create skills"| Skills

    style Singleton fill:#ff79c6
    style Observer fill:#50fa7b
    style Strategy fill:#bd93f9
    style Factory fill:#9b59b6
```

## Install

```bash
pip install -r requirements.txt
python src/main.py
```

## Utilisation

1. **Chat** : Tapez texte
2. **Docs** : Cliquez Load
3. **Models** : Cliquez Models
4. **Skills** : Cliquez Skills
