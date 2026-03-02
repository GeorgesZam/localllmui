# OCR (Optical Character Recognition) Skill

## Description
Extracts text from images using OCR. Supports multiple image formats and languages. Integrates with Tesseract OCR engine for accurate text recognition.

## Use Cases
- Extract text from scanned documents
- Process screenshots
- Read text from images
- Digitize printed documents
- Extract text from photos
- Preprocess images for better OCR accuracy

## Implementation
Install required packages:
```bash
pip install pytesseract pillow
brew install tesseract  # macOS
apt-get install tesseract-ocr  # Linux
```

Example usage:
```python
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

class OCRProcessor:
    def __init__(self, lang='eng'):
        """Initialize OCR processor with language setting."""
        self.lang = lang
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']

    def preprocess_image(self, image_path):
        """Preprocess image for better OCR accuracy."""
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Apply sharpening
        img = img.filter(ImageFilter.SHARPEN)

        # Resize if too small
        width, height = img.size
        if width < 300:
            new_width = 300
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return img

    def extract_text(self, image_path, preprocess=True):
        """Extract text from image file."""
        # Validate file format
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}")

        try:
            # Preprocess image if enabled
            if preprocess:
                img = self.preprocess_image(image_path)
            else:
                img = Image.open(image_path)

            # Extract text using Tesseract
            text = pytesseract.image_to_string(img, lang=self.lang)

            # Extract additional data
            data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)

            return {
                "text": text.strip(),
                "confidence": self._calculate_confidence(data),
                "word_count": len(text.split()),
                "language": self.lang
            }

        except Exception as e:
            raise ValueError(f"OCR processing failed: {str(e)}")

    def extract_text_with_boxes(self, image_path):
        """Extract text with bounding box coordinates."""
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)

        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Confidence threshold
                results.append({
                    "text": data['text'][i],
                    "confidence": int(data['conf'][i]),
                    "bbox": {
                        "left": data['left'][i],
                        "top": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                })

        return results

    def _calculate_confidence(self, data):
        """Calculate average confidence score."""
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        return sum(confidences) / len(confidences) if confidences else 0

    def get_available_languages(self):
        """Get list of available Tesseract languages."""
        return pytesseract.get_languages()
```

## Parameters
- `lang`: Language code for OCR (default: 'eng' for English)
- `preprocess`: Enable image preprocessing (default: True)
- `confidence_threshold`: Minimum confidence for text extraction (default: 60)

## Supported Languages
Common language codes:
- `eng`: English
- `spa`: Spanish
- `fra`: French
- `deu`: German
- `chi_sim`: Chinese (Simplified)
- `jpn`: Japanese
- `kor`: Korean
- `ara`: Arabic

## Returns
For `extract_text()`:
- `text`: Extracted text content
- `confidence`: Average confidence score (0-100)
- `word_count`: Number of words extracted
- `language`: Language used

For `extract_text_with_boxes()`:
- List of text objects with coordinates and confidence scores

## Error Handling
- Validates image file formats
- Handles missing image files gracefully
- Manages OCR engine errors
- Returns empty string for images with no text

## Performance Tips
- Preprocess images for better accuracy
- Use appropriate language settings
- Ensure good image quality (300+ DPI recommended)
- Consider image size and resolution
- Use confidence thresholds to filter results
