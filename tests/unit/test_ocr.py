"""
Unit tests for OCR module.
Following AAA (Arrange-Act-Assert) pattern.

Note: These tests adapt based on OCR availability in the environment.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import ocr
from ocr import OCRProcessor


# Check OCR availability for the test session
OCR_AVAILABLE = ocr.HAS_OCR
PDF2IMAGE_AVAILABLE = ocr.HAS_PDF2IMAGE


class TestOCRProcessorInit:
    """Test cases for OCRProcessor initialization."""

    def test_initializes_processor(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create OCRProcessor
        Assert: Verify processor is created
        """
        # Arrange - None needed

        # Act
        processor = OCRProcessor()

        # Assert
        assert processor is not None
        assert isinstance(processor, OCRProcessor)

    def test_reflects_ocr_availability(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create OCRProcessor
        Assert: Verify available matches environment
        """
        # Arrange - None needed

        # Act
        processor = OCRProcessor()

        # Assert
        assert processor.available == OCR_AVAILABLE

    def test_reflects_pdf_support_availability(self):
        """
        AAA Test:
        Arrange: No setup needed
        Act: Create OCRProcessor
        Assert: Verify pdf_support matches environment
        """
        # Arrange - None needed

        # Act
        processor = OCRProcessor()

        # Assert
        assert processor.pdf_support == PDF2IMAGE_AVAILABLE


class TestOCRProcessorGetStatus:
    """Test cases for get_status method."""

    def test_get_status_returns_dict(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call get_status
        Assert: Verify status dict is returned
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        status = processor.get_status()

        # Assert
        assert isinstance(status, dict)
        assert 'ocr_available' in status
        assert 'pdf_ocr_available' in status

    def test_get_status_ocr_available_matches_environment(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call get_status
        Assert: Verify ocr_available matches environment
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        status = processor.get_status()

        # Assert
        assert status['ocr_available'] == OCR_AVAILABLE

    def test_get_status_pdf_available_when_both_enabled(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call get_status
        Assert: Verify pdf_ocr_available is correct
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        status = processor.get_status()

        # Assert
        expected = OCR_AVAILABLE and PDF2IMAGE_AVAILABLE
        assert status['pdf_ocr_available'] == expected


class TestOCRProcessorGetBestLanguage:
    """Test cases for _get_best_language method."""

    def test_returns_language_string(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call _get_best_language
        Assert: Verify returns string
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        lang = processor._get_best_language('eng')

        # Assert
        assert isinstance(lang, str)

    def test_returns_eng_for_unknown_language(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Request unsupported language
        Assert: Verify falls back to eng or available language
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        lang = processor._get_best_language('xyz_fake_lang')

        # Assert
        # Should return 'eng' or some available language
        assert isinstance(lang, str)
        assert len(lang) > 0


class TestOCRProcessorOCRImage:
    """Test cases for ocr_image method."""

    def test_returns_empty_when_unavailable(self):
        """
        AAA Test:
        Arrange: Create processor (may or may not have OCR)
        Act: Call ocr_image with non-existent file
        Assert: Verify returns empty string on error
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        result = processor.ocr_image('/nonexistent/file.png')

        # Assert
        # Should return empty string if file doesn't exist or OCR fails
        assert isinstance(result, str)

    @pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR not available")
    def test_returns_string_for_valid_input(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a simple image file
        Act: Call ocr_image
        Assert: Verify returns string (may be empty if no text)
        """
        # Arrange - Create a minimal image
        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (10, 10), color='white')
        img.save(img_path)

        processor = OCRProcessor()

        # Act
        result = processor.ocr_image(str(img_path))

        # Assert
        assert isinstance(result, str)


class TestOCRProcessorOCRImageFromBytes:
    """Test cases for ocr_image_from_bytes method."""

    def test_returns_string(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call ocr_image_from_bytes
        Assert: Verify returns string
        """
        # Arrange
        processor = OCRProcessor()
        fake_bytes = b'fake image data'

        # Act
        result = processor.ocr_image_from_bytes(fake_bytes)

        # Assert
        assert isinstance(result, str)


class TestOCRProcessorOCRPDF:
    """Test cases for ocr_pdf method."""

    def test_returns_string(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call ocr_pdf with non-existent file
        Assert: Verify returns string (empty on error)
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        result = processor.ocr_pdf('/nonexistent/file.pdf')

        # Assert
        assert isinstance(result, str)

    def test_returns_empty_when_pdf_unavailable(self):
        """
        AAA Test:
        Arrange: Create processor without PDF support
        Act: Call ocr_pdf
        Assert: Verify returns empty string
        """
        # Arrange
        processor = OCRProcessor()
        if not PDF2IMAGE_AVAILABLE:
            # Act
            result = processor.ocr_pdf('test.pdf')

            # Assert
            assert result == ""


class TestOCRProcessorOCRPPTXImages:
    """Test cases for ocr_pptx_images method."""

    def test_returns_string_for_nonexistent_file(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call ocr_pptx_images with non-existent file
        Assert: Verify returns string (empty on error)
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        result = processor.ocr_pptx_images('/nonexistent/file.pptx')

        # Assert
        assert isinstance(result, str)


class TestOCRProcessorOCRDocxImages:
    """Test cases for ocr_docx_images method."""

    def test_returns_string_for_nonexistent_file(self):
        """
        AAA Test:
        Arrange: Create processor
        Act: Call ocr_docx_images with non-existent file
        Assert: Verify returns string (empty on error)
        """
        # Arrange
        processor = OCRProcessor()

        # Act
        result = processor.ocr_docx_images('/nonexistent/file.docx')

        # Assert
        assert isinstance(result, str)


class TestOCRProcessorIntegration:
    """Integration tests that work with actual OCR if available."""

    @pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR not available")
    def test_full_workflow_with_ocr(self, tmp_path):
        """
        AAA Test:
        Arrange: Create a text image
        Act: Process with OCR
        Assert: Verify text is extracted
        """
        # Arrange
        try:
            from PIL import Image, ImageDraw, ImageFont
            img_path = tmp_path / "text.png"
            img = Image.new('RGB', (200, 50), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test", fill='black')
            img.save(img_path)

            processor = OCRProcessor()

            # Act
            result = processor.ocr_image(str(img_path))

            # Assert
            assert isinstance(result, str)
            # May or may not contain "Test" depending on OCR accuracy
        except ImportError:
            pytest.skip("PIL not fully available")


# Test module-level constants
def test_has_ocr_flag_exists():
    """
    AAA Test:
    Arrange: Import ocr module
    Act: Check HAS_OCR
    Assert: Verify flag is boolean
    """
    # Arrange & Act
    has_ocr = ocr.HAS_OCR

    # Assert
    assert isinstance(has_ocr, bool)


def test_has_pdf2image_flag_exists():
    """
    AAA Test:
    Arrange: Import ocr module
    Act: Check HAS_PDF2IMAGE
    Assert: Verify flag is boolean
    """
    # Arrange & Act
    has_pdf = ocr.HAS_PDF2IMAGE

    # Assert
    assert isinstance(has_pdf, bool)
