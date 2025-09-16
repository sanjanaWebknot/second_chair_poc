#!/usr/bin/env python3
"""Debug script to see what's in the PDF"""

import re

def debug_pdf_text():
    """Debug what text we're actually getting from the PDF"""
    
    # Try to read the PDF with basic text extraction first
    try:
        import fitz
        doc = fitz.open("Deposition_Joseph_Nadeau.pdf")
        print("=== RAW PDF TEXT (first 50 lines) ===")
        for page_num in range(min(3, len(doc))):  # First 3 pages
            page = doc[page_num]
            text = page.get_text("text")
            lines = text.split('\n')
            print(f"\n--- Page {page_num + 1} ---")
            for i, line in enumerate(lines[:20]):  # First 20 lines per page
                print(f"{i+1:2d}: {repr(line)}")
                if 'Q.' in line or 'A.' in line:
                    print(f"    ^^^ FOUND Q/A PATTERN!")
            
    except ImportError:
        print("PyMuPDF not available, trying alternative...")
        
    # Try with our current extraction function
    try:
        from pdf_utils import extract_pdf_text, clean_text
        
        print("\n=== EXTRACTED TEXT (first 50 lines) ===")
        raw_text = extract_pdf_text("Deposition_Joseph_Nadeau.pdf")
        lines = raw_text.split('\n')
        for i, line in enumerate(lines[:300]):
            print(f"{i+1:2d}: {repr(line)}")
            if 'Q.' in line or 'A.' in line:
                print(f"    ^^^ FOUND Q/A PATTERN!")
        
        print("\n=== CLEANED TEXT (first 50 lines) ===")
        cleaned = clean_text(raw_text)
        lines = cleaned.split('\n')
        for i, line in enumerate(lines[:300]):
            print(f"{i+1:2d}: {repr(line)}")
            if 'Q.' in line or 'A.' in line:
                print(f"    ^^^ FOUND Q/A PATTERN!")
                
        # Test our regex patterns
        print("\n=== TESTING REGEX PATTERNS ===")
        for i, line in enumerate(lines[:100]):
            if re.match(r'^\s*\d+\s+Q\.\s*(.*)', line):
                print(f"Q MATCH on line {i+1}: {repr(line)}")
            if re.match(r'^\s*\d+\s+A\.\s*(.*)', line):
                print(f"A MATCH on line {i+1}: {repr(line)}")
                
    except Exception as e:
        print(f"Error with extraction: {e}")

if __name__ == "__main__":
    debug_pdf_text()
