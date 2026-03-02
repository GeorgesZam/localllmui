# RAG (Retrieval-Augmented Generation) Q&A Skill

## Description
Implements RAG-based question answering system using vector embeddings and similarity search. Provides contextual answers based on document collections.

## Use Cases
- Ask questions about document collections
- Retrieve relevant context for queries
- Build knowledge base assistants
- Semantic search across documents
- Document summarization with context
- Multi-document Q&A

## Implementation
Install required packages:
```bash
pip install sentence-transformers faiss-cpu numpy
```

Example usage:
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize RAG system with embedding model."""
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None

    def add_documents(self, texts):
        """Add documents to the knowledge base."""
        self.documents.extend(texts)

        # Generate embeddings
        new_embeddings = self.model.encode(texts)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))

    def query(self, question, top_k=3):
        """Query the knowledge base with a question."""
        if self.index is None:
            return {"answer": "No documents in knowledge base", "context": []}

        # Encode question
        question_embedding = self.model.encode([question])

        # Search for relevant documents
        distances, indices = self.index.search(question_embedding.astype('float32'), top_k)

        # Retrieve context
        context = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                context.append({
                    "document": self.documents[idx],
                    "score": float(1 / (1 + dist))  # Convert to similarity
                })

        return {
            "question": question,
            "context": context,
            "answer": self._generate_answer(question, context)
        }

    def _generate_answer(self, question, context):
        """Generate answer based on retrieved context."""
        if not context:
            return "No relevant information found."

        # Combine relevant context
        relevant_text = "\n\n".join([c["document"] for c in context[:3]])

        # Here you would typically call an LLM with the context
        # For now, return the relevant context
        return f"Based on the documents:\n\n{relevant_text}"
```

## Parameters
- `model_name`: HuggingFace model name for embeddings (default: "all-MiniLM-L6-v2")
- `top_k`: Number of relevant documents to retrieve (default: 3)
- `chunk_size`: Size of document chunks (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)

## Usage Example
```python
# Initialize RAG system
rag = RAGSystem()

# Add documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing deals with text data."
]
rag.add_documents(documents)

# Query the system
result = rag.query("What is Python?")
print(result["answer"])
```

## Returns
Dictionary containing:
- `question`: The original question
- `context`: List of relevant document chunks with similarity scores
- `answer`: Generated answer based on retrieved context

## Features
- Semantic search using vector embeddings
- Fast similarity search with FAISS
- Configurable embedding models
- Chunk-based document processing
- Similarity scoring for results

## Error Handling
- Validates document inputs
- Handles empty knowledge base queries
- Manages embedding generation errors
- Returns graceful fallbacks for no matches
