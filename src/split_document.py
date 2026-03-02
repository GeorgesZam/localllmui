#!/usr/bin/env python3
"""
Helper script to split large documents into smaller chunks for better RAG performance.

Usage:
    python src/split_document.py <input_file> [output_dir]

Example:
    python src/split_document.py large_document.txt split_docs/
"""

import sys
import os
from pathlib import Path


def split_document(input_file: str, output_dir: str = None, max_chars: int = 2000):
    """
    Split a large document into smaller chunks.

    Args:
        input_file: Path to the input document
        output_dir: Directory for output files (default: input_file_split)
        max_chars: Maximum characters per chunk (default: 2000)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return False

    # Set output directory
    if output_dir is None:
        output_dir = input_path.stem + "_split"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Splitting: {input_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Max chars per chunk: {max_chars}")
    print()

    # Read the document
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # Get file extension
    ext = input_path.suffix

    # Split by paragraphs first for better chunks
    paragraphs = content.split('\n\n')

    chunks = []
    current_chunk = ""
    chunk_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph itself is too long, split by sentences
        if len(para) > max_chars:
            sentences = para.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) + 1 < max_chars:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        chunk_num += 1
                    current_chunk = sentence + ". "
        else:
            # Check if adding this paragraph would exceed limit
            if len(current_chunk) + len(para) + 2 < max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    chunk_num += 1
                current_chunk = para + "\n\n"

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Write chunks to files
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        output_file = output_path / f"{input_path.stem}_part{i}{ext}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        print(f"  {output_file.name} ({len(chunk)} chars)")

    print()
    print(f"✓ Successfully split {len(chunks)} chunks")
    print(f"  Original: {len(content)} chars")
    print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    print()
    print(f"Move these files to 'src/data/' to use them with RAG:")
    print(f"  mv {output_dir}/* src/data/")

    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python split_document.py <input_file> [output_dir] [max_chars]")
        print()
        print("Example:")
        print("  python split_document.py large_doc.txt")
        print("  python split_document.py large_doc.txt split_docs/")
        print("  python split_document.py large_doc.txt split_docs/ 1500")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    max_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    success = split_document(input_file, output_dir, max_chars)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
