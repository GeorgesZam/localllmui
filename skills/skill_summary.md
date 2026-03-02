# Document Summarization Skill

## Description
Generates concise summaries of long documents using LLM capabilities. Supports extractive and abstractive summarization approaches.

## Use Cases
- Summarize long articles or reports
- Create executive summaries
- Condense meeting notes
- Summarize document collections
- Generate bullet-point summaries
- Multi-document synthesis

## Implementation
Example usage:
```python
class DocumentSummarizer:
    def __init__(self, llm_client):
        """Initialize summarizer with LLM client."""
        self.llm_client = llm_client
        self.max_chunk_size = 4000
        self.overlap = 200

    def chunk_text(self, text, max_size=None):
        """Split text into manageable chunks."""
        if max_size is None:
            max_size = self.max_chunk_size

        chunks = []
        current_chunk = ""

        sentences = text.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def summarize_single_document(self, text, summary_type="bullet"):
        """Summarize a single document."""
        chunks = self.chunk_text(text)

        if len(chunks) == 1:
            return self._generate_summary(chunks[0], summary_type)

        # Multi-stage summarization for long documents
        chunk_summaries = []
        for chunk in chunks:
            summary = self._generate_summary(chunk, "concise")
            chunk_summaries.append(summary)

        # Combine chunk summaries
        combined = " ".join(chunk_summaries)
        return self._generate_summary(combined, summary_type)

    def summarize_multiple_documents(self, documents, summary_type="executive"):
        """Summarize multiple documents into a single summary."""
        individual_summaries = []

        for doc in documents:
            summary = self.summarize_single_document(doc, "concise")
            individual_summaries.append(summary)

        # Create synthesis prompt
        prompt = f"""Synthesize the following document summaries into a coherent {summary_type} summary:

Documents:
{chr(10).join([f'{i+1}. {s}' for i, s in enumerate(individual_summaries)])}

Provide a unified summary that captures key points from all documents."""

        return self.llm_client.generate(prompt)

    def _generate_summary(self, text, summary_type):
        """Generate summary using LLM."""
        prompts = {
            "bullet": f"Create a bullet-point summary of the following text:\n\n{text}\n\nSummary:",
            "executive": f"Create an executive summary (2-3 paragraphs) of the following:\n\n{text}\n\nSummary:",
            "concise": f"Summarize this text in 2-3 sentences:\n\n{text}\n\nSummary:",
            "detailed": f"Provide a detailed summary of the following, maintaining key information:\n\n{text}\n\nSummary:"
        }

        prompt = prompts.get(summary_type, prompts["concise"])
        return self.llm_client.generate(prompt)

    def extract_key_points(self, text, num_points=5):
        """Extract key points from document."""
        prompt = f"""Extract the {num_points} most important points from the following text. Present each point on a separate line.

Text:
{text}

Key Points:"""

        return self.llm_client.generate(prompt)
```

## Parameters
- `summary_type`: Type of summary to generate
  - `bullet`: Bullet-point format
  - `executive`: Executive summary (2-3 paragraphs)
  - `concise`: 2-3 sentence summary
  - `detailed`: Comprehensive summary
- `max_length`: Maximum length of summary in characters (optional)
- `num_points`: Number of key points to extract (default: 5)
- `chunk_size`: Size of text chunks for processing (default: 4000)

## Returns
For single document:
- `summary`: Generated summary text
- `type`: Summary type used
- `original_length`: Length of original text
- `summary_length`: Length of summary

For multiple documents:
- `summary`: Unified summary
- `individual_summaries`: List of individual document summaries
- `document_count`: Number of documents processed

## Usage Example
```python
summarizer = DocumentSummarizer(llm_client)

# Single document
result = summarizer.summarize_single_document(long_text, "executive")

# Multiple documents
docs = [doc1_text, doc2_text, doc3_text]
result = summarizer.summarize_multiple_documents(docs, "bullet")

# Extract key points
key_points = summarizer.extract_key_points(document_text, num_points=7)
```

## Features
- Handles documents of any length through chunking
- Multiple summary formats
- Multi-document synthesis
- Key point extraction
- Configurable summary length and style

## Error Handling
- Validates input text is not empty
- Handles LLM API errors gracefully
- Returns partial summaries if some chunks fail
- Manages memory for very large documents
