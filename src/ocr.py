"""
OCR processor for scanned documents and images.
"""

import os
import sys
import glob
import shutil
from typing import Optional, Callable

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
        self.poppler_path = None
        self._configure_tesseract()
        self._configure_poppler()
    
    def _configure_tesseract(self):
        if not HAS_OCR:
            print("[OCR] pytesseract not installed")
            return
        
        if sys.platform == 'win32':
            if hasattr(sys, '_MEIPASS'):
                tesseract_path = os.path.join(sys._MEIPASS, 'tesseract', 'tesseract.exe')
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    print(f"[OCR] Using bundled Tesseract: {tesseract_path}")
                    return
            
            standard_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"[OCR] Using Tesseract: {path}")
                    break
            else:
                print("[OCR] Warning: Tesseract not found in standard paths")
        
        try:
            version = pytesseract.get_tesseract_version()
            print(f"[OCR] Tesseract version: {version}")
        except Exception as e:
            print(f"[OCR] Tesseract not accessible: {e}")
            self.available = False
    
    def _configure_poppler(self):
        if not HAS_PDF2IMAGE:
            print("[OCR] pdf2image not installed")
            return
        
        if sys.platform == 'win32':
            poppler_search_paths = [
                r"C:\Program Files\poppler-*\Library\bin",
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files (x86)\poppler-*\Library\bin",
                r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin",
                r"C:\ProgramData\chocolatey\bin",
            ]
            
            for pattern in poppler_search_paths:
                if '*' in pattern:
                    matches = glob.glob(pattern)
                    for match in matches:
                        if os.path.isdir(match):
                            self.poppler_path = match
                            print(f"[OCR] Using Poppler: {self.poppler_path}")
                            return
                elif os.path.isdir(pattern):
                    self.poppler_path = pattern
                    print(f"[OCR] Using Poppler: {self.poppler_path}")
                    return
            
            if shutil.which('pdftoppm'):
                print("[OCR] Poppler found in PATH")
            else:
                print("[OCR] Warning: Poppler not found")
        else:
            if shutil.which('pdftoppm'):
                print("[OCR] Poppler available in PATH")
            else:
                print("[OCR] Warning: Poppler not found in PATH")
    
    def get_status(self) -> dict:
        status = {
            "ocr_available": self.available,
            "pdf_ocr_available": self.available and self.pdf_support,
        }
        
        if self.available:
            try:
                langs = pytesseract.get_languages()
                status["available_languages"] = langs
                print(f"[OCR] Available languages: {langs}")
            except Exception:
                status["available_languages"] = []
        
        return status
    
    def _get_best_language(self, preferred: str = 'eng') -> str:
        if not self.available:
            return 'eng'
        
        try:
            available_langs = pytesseract.get_languages()
            available_langs = [l for l in available_langs if l != 'osd']
            
            preferred_list = preferred.split('+')
            valid_langs = [lang for lang in preferred_list if lang in available_langs]
            
            if valid_langs:
                return '+'.join(valid_langs)
            if 'eng' in available_langs:
                return 'eng'
            if available_langs:
                return available_langs[0]
            return 'eng'
        except Exception:
            return 'eng'
    
    def ocr_image(self, image_path: str, lang: str = 'eng') -> str:
        if not self.available:
            print("[OCR] OCR not available")
            return ""
        
        try:
            print(f"[OCR] Processing image: {image_path}")
            image = Image.open(image_path)
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            use_lang = self._get_best_language(lang)
            
            try:
                text = pytesseract.image_to_string(image, lang=use_lang)
            except pytesseract.TesseractError as e:
                print(f"[OCR] Language error: {e}")
                try:
                    text = pytesseract.image_to_string(image)
                except:
                    return ""
            
            result = text.strip()
            print(f"[OCR] Extracted {len(result)} characters")
            return result
        except Exception as e:
            print(f"[OCR] Error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def ocr_image_from_bytes(self, image_bytes: bytes, lang: str = 'eng') -> str:
        if not self.available:
            return ""
        
        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            use_lang = self._get_best_language(lang)
            
            try:
                text = pytesseract.image_to_string(image, lang=use_lang)
            except:
                try:
                    text = pytesseract.image_to_string(image)
                except:
                    return ""
            
            return text.strip()
        except Exception as e:
            print(f"[OCR] Error: {e}")
            return ""
    
    def ocr_pdf(self, pdf_path: str, lang: str = 'eng', dpi: int = 300,
                on_progress: Optional[Callable[[str], None]] = None) -> str:
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
            
            convert_kwargs = {'dpi': dpi}
            if sys.platform == 'win32' and self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path
            
            try:
                images = convert_from_path(pdf_path, **convert_kwargs)
            except Exception as e:
                log(f"PDF conversion error: {e}")
                if dpi > 150:
                    log("Retrying with lower DPI (150)...")
                    convert_kwargs['dpi'] = 150
                    try:
                        images = convert_from_path(pdf_path, **convert_kwargs)
                    except:
                        return ""
                else:
                    return ""
            
            if not images:
                log("No images extracted from PDF")
                return ""
            
            all_text = []
            total_pages = len(images)
            log(f"Extracted {total_pages} page(s)")
            
            use_lang = self._get_best_language(lang)
            log(f"Using language: {use_lang}")
            
            for i, image in enumerate(images):
                log(f"OCR page {i + 1}/{total_pages}")
                
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                try:
                    text = pytesseract.image_to_string(image, lang=use_lang)
                except:
                    try:
                        text = pytesseract.image_to_string(image)
                    except:
                        text = ""
                
                if text.strip():
                    all_text.append(f"=== Page {i + 1} ===\n{text.strip()}")
            
            log(f"OCR complete: {len(all_text)} pages with text")
            return "\n\n".join(all_text)
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def ocr_pptx_images(self, pptx_path: str, lang: str = 'eng') -> str:
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
                            text = self.ocr_image_from_bytes(shape.image.blob, lang)
                            if text:
                                ocr_texts.append(f"[Slide {slide_num} - Image]\n{text}")
                        except Exception as e:
                            print(f"[OCR] Error slide {slide_num}: {e}")
            
            return "\n\n".join(ocr_texts)
        except ImportError:
            return ""
        except Exception as e:
            print(f"[OCR] PPTX error: {e}")
            return ""
    
    def ocr_docx_images(self, docx_path: str, lang: str = 'eng') -> str:
        if not self.available:
            return ""
        
        try:
            from docx import Document
            doc = Document(docx_path)
            ocr_texts = []
            image_count = 0
            
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        text = self.ocr_image_from_bytes(rel.target_part.blob, lang)
                        if text:
                            image_count += 1
                            ocr_texts.append(f"[Image {image_count}]\n{text}")
                    except Exception as e:
                        print(f"[OCR] DOCX image error: {e}")
            
            return "\n\n".join(ocr_texts)
        except ImportError:
            return ""
        except Exception as e:
            print(f"[OCR] DOCX error: {e}")
            return ""
