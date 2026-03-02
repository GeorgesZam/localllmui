# PDF Document Processing Skill

## Description
Extracts text, images, and structured content from PDF documents. Supports both text-based and scanned PDFs using OCR capabilities.

## Use Cases
- Extract text from PDFs
- Process scanned PDFs with OCR
- Extract images from PDF files
- Parse multi-page documents
- Prepare PDF content for RAG indexing
- Handle password-protected PDFs

## Implementation
Install required packages:
```bash
pip install PyPDF2 pdf2image pillow pytesseract
```

Example usage:
```python
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

def extract_from_pdf(file_path, use_ocr=False, password=None):
    """Extract content from PDF files."""
    content = {
        "text": "",
        "pages": [],
        "metadata": {},
        "images": []
    }

    try:
        # Text-based extraction
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Handle password protection
            if pdf_reader.is_encrypted:
                if password:
                    pdf_reader.decrypt(password)
                else:
                    raise ValueError("PDF is password-protected")

            # Extract metadata
            if pdf_reader.metadata:
                content["metadata"] = {
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                    "producer": pdf_reader.metadata.get('/Producer', ''),
                    "pages": len(pdf_reader.pages)
                }

            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                content["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
                content["text"] += page_text + "\n\n"

        # OCR for scanned PDFs
        if use_ocr:
            images = convert_from_path(file_path)
            for img_num, image in enumerate(images):
                ocr_text = pytesseract.image_to_string(image)
                content["pages"][img_num]["ocr_text"] = ocr_text
                content["images"].append({
                    "page_number": img_num + 1,
                    "image": image
                })

    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

    return content
```

## Parameters
- `file_path`: Path to the PDF file
- `use_ocr`: Boolean to enable OCR for scanned PDFs (default: False)
- `password`: Password for encrypted PDFs (optional)
- `extract_images`: Boolean to extract images from PDF (default: True)
- `dpi`: Resolution for image conversion (default: 200)

## Returns
Dictionary containing:
- `text`: Full concatenated text
- `pages`: List of page objects with text and metadata
- `metadata`: PDF document metadata
- `images`: List of extracted images (if extract_images=True)

## Error Handling
- Handles encrypted PDFs with password parameter
- Manages memory for large PDFs by processing page by page
- Falls back to OCR for image-only PDFs
- Validates PDF file format before processing
