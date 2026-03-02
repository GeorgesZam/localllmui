# DOCX Document Processing Skill

## Description
Extracts, processes, and analyzes content from Microsoft Word (.docx) documents. Reads text, tables, and structured content from Word documents for further processing or RAG indexing.

## Use Cases
- Extract text content from .docx files
- Parse and preserve document structure (headings, paragraphs, lists)
- Extract tables and their data
- Prepare document content for RAG indexing
- Analyze document metadata (author, created date, modified date)

## Implementation
Install required package:
```bash
pip install python-docx
```

Example usage:
```python
from docx import Document

def extract_from_docx(file_path):
    """Extract all text and structured content from a DOCX file."""
    doc = Document(file_path)

    content = {
        "paragraphs": [],
        "tables": [],
        "metadata": {}
    }

    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            content["paragraphs"].append({
                "text": para.text,
                "style": para.style.name if para.style else "Normal"
            })

    # Extract tables
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        content["tables"].append(table_data)

    # Extract metadata
    core_props = doc.core_properties
    content["metadata"] = {
        "author": core_props.author,
        "created": core_props.created,
        "modified": core_props.modified,
        "title": core_props.title,
        "comments": core_props.comments
    }

    return content
```

## Parameters
- `file_path`: Path to the .docx file to process
- `extract_tables`: Boolean to include/exclude table extraction (default: True)
- `preserve_formatting`: Boolean to preserve style information (default: False)

## Returns
Dictionary containing:
- `paragraphs`: List of paragraph objects with text and style info
- `tables`: List of tables as 2D arrays
- `metadata`: Document metadata
- `full_text`: Concatenated text content (optional)

## Error Handling
- Raises `ValueError` for invalid file paths
- Raises `docx.opc.exceptions.PackageNotFoundError` for non-docx files
- Handles corrupted documents gracefully with try/except blocks
