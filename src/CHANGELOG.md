# Changelog - Local LLM UI

## [Fixed] Context Window Overflow Issue

### Problem
When using RAG with large documents, the app would show:
```
[Error: Requested tokens (2231) exceed context window of 2048]
```

### Solution
Updated `src/rag.py` with intelligent context truncation:
- Limited RAG context to ~1200 characters (leaves room for prompt/response)
- Reduced chunk size from 384 to 256 tokens
- Reduced top_k results from 3 to 2
- Added smart truncation that preserves document structure

### Changes Made

**src/rag.py:**
- Added `max_context_chars` parameter to `search()` method
- Implemented per-chunk size calculation based on available context
- Added automatic truncation with "..." suffix for long content

**src/config.py:**
- `rag_chunk_size`: 384 → 256
- `rag_chunk_overlap`: 50 → 30
- `rag_top_k`: 3 → 2

**src/cli.py:**
- Added helpful error messages for context overflow
- Added tips for users when this error occurs

### New Tool: Document Splitter

Created `src/split_document.py` to help users split large documents:

```bash
# Split a large document into smaller chunks
python src/split_document.py large_document.txt

# Custom output directory and chunk size
python src/split_document.py large_document.txt split_docs/ 1500
```

### Usage Recommendations

1. **For small documents (< 5 pages)**: Use directly
   ```bash
   cp document.txt src/data/
   ```

2. **For medium documents (5-20 pages)**: Let RAG handle it
   - The system will automatically chunk and limit context

3. **For large documents (> 20 pages)**: Pre-split manually
   ```bash
   python src/split_document.py large_book.txt
   mv large_book_split/* src/data/
   ```

4. **Ask specific questions** instead of general ones
   - ✅ "What is the main conclusion of the pipeline section?"
   - ❌ "Tell me everything about this document"

### Testing

The fix has been tested with:
- Documents up to 10,000 characters
- Multiple chunks returned
- Context length limits properly enforced

Run the tests to verify:
```bash
pytest tests/unit/test_rag.py -v
```

### Performance Impact

- **Before**: Context overflow errors on any document with RAG
- **After**: Smooth operation with automatic context management
- **Response Quality**: Maintained or improved due to focused context
