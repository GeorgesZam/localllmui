"""
OCR processor for scanned documents and images.
"""

import os
import sys
from typing import Optional, Callable

# OCR dependencies
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class OCRProcessor:
    """Handles OCR for scanned documents and images."""
    
    def __init__(self):
        self.available = HAS_OCR
        self.pdf_support = HAS_PDF2IMAGE
        self._configure_tesseract()
    
    def _configure_tesseract(self):
        """Configure Tesseract path for PyInstaller on Windows."""
        if not HAS_OCR:
            return
        
        if sys.platform == 'win32':
            # Try bundled path first
            if hasattr(sys, '_MEIPASS'):
                tesseract_path = os.path.join(sys._MEIPASS, 'tesseract', 'tesseract.exe')
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    return
            
            # Try standard Windows installation paths
            standard_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for path in standard_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    return
    
    def get_status(self) -> dict:
        """Returns OCR availability status."""
        return {
            "ocr_available": self.available,
            "pdf_ocr_available": self.available and self.pdf_support,
        }
    
    def ocr_image(self, image_path: str, lang: str = 'fra+eng') -> str:
        """Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            lang: Tesseract language codes (default: French + English)
        
        Returns:
            Extracted text or empty string on failure
        """
        if not self.available:
            print("[OCR] OCR not available")
            return ""
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            print(f"[OCR] Error processing image {image_path}: {e}")
            return ""
    
    def ocr_image_from_bytes(self, image_bytes: bytes, lang: str = 'fra+eng') -> str:
        """Extract text from image bytes using OCR.
        
        Args:
            image_bytes: Raw image bytes
            lang: Tesseract language codes
        
        Returns:
            Extracted text or empty string on failure
        """
        if not self.available:
            return ""
        
        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            print(f"[OCR] Error processing image bytes: {e}")
            return ""
    
    def ocr_pdf(self, pdf_path: str, lang: str = 'fra+eng', 
                dpi: int = 300, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Extract text from a scanned PDF using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            lang: Tesseract language codes
            dpi: Resolution for PDF to image conversion
            on_progress: Optional callback for progress updates
        
        Returns:
            Extracted text or empty string on failure
        """
        if not self.available:
            print("[OCR] OCR not available")
            return ""
        
        if not self.pdf_support:
            print("[OCR] pdf2image not available")
            return ""
        
        def log(msg):
            print(f"[OCR] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            log(f"Converting PDF to images (dpi={dpi})...")
            images = convert_from_path(pdf_path, dpi=dpi)
            
            all_text = []
            total_pages = len(images)
            
            for i, image in enumerate(images):
                log(f"OCR page {i + 1}/{total_pages}")
                
                # Convert to RGB if needed
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                text = pytesseract.image_to_string(image, lang=lang)
                if text.strip():
                    all_text.append(f"=== Page {i + 1} ===\n{text.strip()}")
            
            log(f"OCR complete: {len(all_text)} pages with text")
            return "\n\n".join(all_text)
        except Exception as e:
            log(f"Error processing PDF: {e}")
            return ""
    
    def ocr_pptx_images(self, pptx_path: str, lang: str = 'fra+eng') -> str:
        """Extract text from images embedded in PowerPoint.
        
        Args:
            pptx_path: Path to the PPTX file
            lang: Tesseract language codes
        
        Returns:
            Extracted text from all images or empty string
        """
        if not self.available:
            return ""
        
        try:
            from pptx import Presentation
            
            prs = Presentation(pptx_path)
            ocr_texts = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                for shape in slide.shapes:
                    if hasattr(shape, "image"):
                        try:
                            image_bytes = shape.image.blob
                            text = self.ocr_image_from_bytes(image_bytes, lang)
                            if text:
                                ocr_texts.append(f"[Slide {slide_num} - Image]\n{text}")
                        except Exception as e:
                            print(f"[OCR] Error with image on slide {slide_num}: {e}")
            
            return "\n\n".join(ocr_texts)
        except ImportError:
            print("[OCR] python-pptx not available")
            return ""
        except Exception as e:
            print(f"[OCR] Error processing PPTX images: {e}")
            return ""
    
    def ocr_docx_images(self, docx_path: str, lang: str = 'fra+eng') -> str:
        """Extract text from images embedded in Word documents.
        
        Args:
            docx_path: Path to the DOCX file
            lang: Tesseract language codes
        
        Returns:
            Extracted text from all images or empty string
        """
        if not self.available:
            return ""
        
        try:
            from docx import Document
            
            doc = Document(docx_path)
            ocr_texts = []
            image_count = 0
            
            # Access images through document parts
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_bytes = rel.target_part.blob
                        text = self.ocr_image_from_bytes(image_bytes, lang)
                        if text:
                            image_count += 1
                            ocr_texts.append(f"[Image {image_count}]\n{text}")
                    except Exception as e:
                        print(f"[OCR] Error with image in DOCX: {e}")
            
            return "\n\n".join(ocr_texts)
        except ImportError:
            print("[OCR] python-docx not available")
            return ""
        except Exception as e:
            print(f"[OCR] Error processing DOCX images: {e}")
            return ""
