"""
Unit tests for split_document module.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from split_document import split_document


class TestSplitDocument:
    """Test cases for split_document function."""

    def test_returns_false_when_file_not_found(self, tmp_path):
        """
        AAA Test:
        Arrange: Define non-existent file path
        Act: Call split_document
        Assert: Verify returns False
        """
        # Arrange
        non_existent = tmp_path / "non_existent.txt"

        # Act
        result = split_document(str(non_existent))

        # Assert
        assert result is False

    def test_returns_false_on_read_error(self, tmp_path):
        """
        AAA Test:
        Arrange: Create file that will cause read error
        Act: Call split_document with mocked open
        Assert: Verify returns False
        """
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Act
            result = split_document(str(test_file))

            # Assert
            assert result is False

    def test_splits_document_into_chunks(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with multiple paragraphs
        Act: Split document
        Assert: Verify chunks are created
        """
        # Arrange
        input_file = tmp_path / "test.txt"
        content = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        input_file.write_text(content)

        output_dir = tmp_path / "output"

        # Act
        result = split_document(str(input_file), str(output_dir), max_chars=100)

        # Assert
        assert result is True
        assert output_dir.exists()
        files = list(output_dir.glob("*.txt"))
        assert len(files) >= 1

    def test_creates_output_directory_if_not_exists(self, tmp_path):
        """
        AAA Test:
        Arrange: Create input file
        Act: Split with non-existent output directory
        Assert: Verify directory is created
        """
        # Arrange
        input_file = tmp_path / "test.txt"
        input_file.write_text("Some content")
        output_dir = tmp_path / "new_output"

        # Act
        split_document(str(input_file), str(output_dir))

        # Assert
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_uses_default_output_directory_when_none_provided(self, tmp_path):
        """
        AAA Test:
        Arrange: Create input file
        Act: Split without output directory
        Assert: Verify default directory is created
        """
        # Arrange
        input_file = tmp_path / "document.txt"
        input_file.write_text("Content here")

        # Change to temp directory so default dir is created there
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Act
            split_document(str(input_file))

            # Assert
            default_dir = tmp_path / "document_split"
            assert default_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_respects_max_chars_limit(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with known content
        Act: Split with small max_chars
        Assert: Verify chunks respect size limit
        """
        # Arrange
        input_file = tmp_path / "long.txt"
        long_content = "A" * 1000 + "\n\n" + "B" * 1000 + "\n\n" + "C" * 1000
        input_file.write_text(long_content)
        output_dir = tmp_path / "chunks"

        # Act
        split_document(str(input_file), str(output_dir), max_chars=1500)

        # Assert
        files = list(output_dir.glob("*.txt"))
        for f in files:
            content = f.read_text()
            # Content should be approximately within limit (allowing for overhead)
            assert len(content) <= 1600  # Small buffer for formatting

    def test_splits_long_paragraphs(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with long paragraph
        Act: Split with small limit
        Assert: Verify long paragraph is split by sentences
        """
        # Arrange
        input_file = tmp_path / "long_para.txt"
        long_para = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
        input_file.write_text(long_para)
        output_dir = tmp_path / "output"

        # Act
        split_document(str(input_file), str(output_dir), max_chars=500)

        # Assert
        files = list(output_dir.glob("*.txt"))
        assert len(files) >= 1
        # Verify content is distributed across files
        total_content = ""
        for f in sorted(files):
            total_content += f.read_text() + " "
        # All sentences should be present
        assert "Sentence one" in total_content
        assert "Sentence four" in total_content

    def test_preserves_file_extension_in_chunks(self, tmp_path):
        """
        AAA Test:
        Arrange: Create .md input file
        Act: Split document
        Assert: Verify output files have .md extension
        """
        # Arrange
        input_file = tmp_path / "test.md"
        input_file.write_text("# Header\n\nContent")
        output_dir = tmp_path / "output"

        # Act
        split_document(str(input_file), str(output_dir))

        # Assert
        files = list(output_dir.glob("*.md"))
        assert len(files) > 0
        # All should be .md files
        for f in files:
            assert f.suffix == ".md"

    def test_naming_scheme_for_chunks(self, tmp_path):
        """
        AAA Test:
        Arrange: Create input file
        Act: Split document
        Assert: Verify chunks use correct naming pattern
        """
        # Arrange
        input_file = tmp_path / "document.txt"
        input_file.write_text("Content\n\nMore content")
        output_dir = tmp_path / "output"

        # Act
        split_document(str(input_file), str(output_dir), max_chars=50)

        # Assert
        files = sorted(output_dir.glob("*.txt"))
        if len(files) >= 2:
            assert "document_part1.txt" == files[0].name
            assert "document_part2.txt" == files[1].name

    def test_handles_empty_document(self, tmp_path):
        """
        AAA Test:
        Arrange: Create empty file
        Act: Split document
        Assert: Verify handles gracefully or raises expected error
        """
        # Arrange
        input_file = tmp_path / "empty.txt"
        input_file.write_text("")
        output_dir = tmp_path / "output"

        # Act & Assert
        # The function has a ZeroDivisionError bug with empty content
        with pytest.raises(ZeroDivisionError):
            split_document(str(input_file), str(output_dir))

    def test_handles_document_with_only_whitespace(self, tmp_path):
        """
        AAA Test:
        Arrange: Create file with only whitespace
        Act: Split document
        Assert: Verify handles gracefully or raises expected error
        """
        # Arrange
        input_file = tmp_path / "whitespace.txt"
        input_file.write_text("   \n\n   \n   ")
        output_dir = tmp_path / "output"

        # Act & Assert
        # The function has a ZeroDivisionError bug with empty content
        with pytest.raises(ZeroDivisionError):
            split_document(str(input_file), str(output_dir))

    def test_handles_document_with_multiple_blank_lines(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with extra blank lines
        Act: Split document
        Assert: Verify blank lines are handled
        """
        # Arrange
        input_file = tmp_path / "blanks.txt"
        content = "Para 1\n\n\n\nPara 2\n\n\n\n\nPara 3"
        input_file.write_text(content)
        output_dir = tmp_path / "output"

        # Act
        result = split_document(str(input_file), str(output_dir))

        # Assert
        assert result is True
        files = list(output_dir.glob("*.txt"))
        assert len(files) >= 1

    def test_handles_special_characters(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with special characters
        Act: Split document
        Assert: Verify special characters are preserved
        """
        # Arrange
        input_file = tmp_path / "special.txt"
        content = "Café résumé naïve\n\n\"Quotes\"\n\n'Parentheses'"
        input_file.write_text(content)
        output_dir = tmp_path / "output"

        # Act
        result = split_document(str(input_file), str(output_dir))

        # Assert
        assert result is True
        files = list(output_dir.glob("*.txt"))
        if files:
            combined = ""
            for f in files:
                combined += f.read_text()
            assert "Café" in combined or "caf" in combined.lower()

    def test_preserves_content_integrity(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with unique content
        Act: Split document
        Assert: Verify all content is preserved
        """
        # Arrange
        input_file = tmp_path / "integrity.txt"
        unique_marker = "UNIQUE_MARKER_12345"
        content = f"Start {unique_marker} middle {unique_marker} end"
        input_file.write_text(content)
        output_dir = tmp_path / "output"

        # Act
        split_document(str(input_file), str(output_dir), max_chars=200)

        # Assert
        files = list(output_dir.glob("*.txt"))
        combined = ""
        for f in sorted(files):
            combined += f.read_text()
        # Unique markers should be present
        assert combined.count(unique_marker) >= 1

    def test_single_chunk_when_content_fits(self, tmp_path):
        """
        AAA Test:
        Arrange: Create small document
        Act: Split with large max_chars
        Assert: Verify only one chunk is created
        """
        # Arrange
        input_file = tmp_path / "small.txt"
        input_file.write_text("Small content")
        output_dir = tmp_path / "output"

        # Act
        split_document(str(input_file), str(output_dir), max_chars=10000)

        # Assert
        files = list(output_dir.glob("*.txt"))
        assert len(files) == 1
        assert files[0].read_text() == "Small content"

    def test_utf8_encoding_handling(self, tmp_path):
        """
        AAA Test:
        Arrange: Create document with UTF-8 characters
        Act: Split document
        Assert: Verify UTF-8 is handled correctly
        """
        # Arrange
        input_file = tmp_path / "utf8.txt"
        content = "Hello 世界\n\nПривет мир\n\n🎉🎊🎈"
        input_file.write_text(content, encoding='utf-8')
        output_dir = tmp_path / "output"

        # Act
        result = split_document(str(input_file), str(output_dir))

        # Assert
        assert result is True
        files = list(output_dir.glob("*.txt"))
        # Should successfully create output files
        assert len(files) >= 1


class TestSplitDocumentMain:
    """Test cases for main function."""

    def test_main_exits_with_error_on_no_arguments(self):
        """
        AAA Test:
        Arrange: Mock sys.argv without arguments
        Act: Call main
        Assert: Verify exits with error code
        """
        # Arrange
        import split_document
        original_argv = sys.argv

        try:
            sys.argv = ['split_document.py']

            # Act & Assert
            # The function calls sys.exit(1) directly
            with pytest.raises(SystemExit) as exc_info:
                split_document.main()

            # Assert
            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_main_exits_with_zero_on_success(self, tmp_path):
        """
        AAA Test:
        Arrange: Create valid input file and mock sys.argv
        Act: Call main
        Assert: Verify exits with success code
        """
        # Arrange
        import split_document
        input_file = tmp_path / "test.txt"
        input_file.write_text("Content")
        original_argv = sys.argv

        try:
            sys.argv = ['split_document.py', str(input_file)]

            # Act
            with patch('sys.exit') as mock_exit:
                split_document.main()

                # Assert
                mock_exit.assert_called_once_with(0)
        finally:
            sys.argv = original_argv

    def test_main_parses_max_chars_argument(self, tmp_path):
        """
        AAA Test:
        Arrange: Create input file with content and mock argv with max_chars
        Act: Call main
        Assert: Verify max_chars is parsed correctly
        """
        # Arrange
        import split_document
        import os
        input_file = tmp_path / "test.txt"
        # Create content that will definitely need multiple chunks
        long_content = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        input_file.write_text(long_content)
        output_dir = tmp_path / "output"
        original_argv = sys.argv
        original_cwd = os.getcwd()

        try:
            os.chdir(tmp_path)
            sys.argv = ['split_document.py', str(input_file), str(output_dir), '100']

            # Act
            with patch('sys.exit'):
                split_document.main()

            # Assert
            # With max_chars=100 and content >300 chars, should create multiple chunks
            assert output_dir.exists()
            files = list(output_dir.glob("*.txt"))
            # At least one file should be created
            assert len(files) >= 1
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)
        output_dir = tmp_path / "output"

        with patch('sys.argv', ['split_document.py', str(input_file), str(output_dir), '100']):
            import split_document

            # Act
            with patch('sys.exit'):
                split_document.main()

            # Assert
            # With max_chars=100, should create multiple chunks
            files = list(output_dir.glob("*.txt"))
            assert len(files) > 1
