import fitz
import re
import uuid
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
import tempfile
from PIL import Image

# Patterns for repetitive headers/footers you don’t want
FOOTER_PATTERNS = [
    r"Vivian Dafoulas.*",        # reporter info
    r"CDALEDEP\d+",              # Bates number
    r"[A-F0-9-]{36}",            # UUID-like strings
]

def is_footer_or_header(text: str) -> bool:
    """Return True if the line matches a known header/footer pattern."""
    return any(re.search(p, text) for p in FOOTER_PATTERNS)


def extract_pdf_text(pdf_path: str) -> str:
    """Try text extraction with PyMuPDF; fallback to OCR if too little text is extracted."""
    doc = fitz.open(pdf_path)
    full_text = []

    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        if raw_text:
            full_text.append(raw_text)

    text = "\n".join(full_text)

    # If very little text, try OCR
    if len(text.strip()) < 200:  # heuristic: adjust threshold as needed
        print("Very little text extracted, running OCR fallback...")
        text = perform_ocr(pdf_path)

    return text


def perform_ocr(pdf_path: str) -> str:
    """Convert PDF to images and extract text with OCR."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(open(pdf_path, "rb").read())
        temp_pdf_path = temp_pdf.name

    images = convert_from_path(temp_pdf_path, dpi=300)
    for image in images:
        text += pytesseract.image_to_string(image)
    return text


def clean_text(text: str) -> str:
    """Remove known footers, headers, and junk lines."""
    cleaned_lines = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if is_footer_or_header(t):
            continue
        if t.isdigit() and len(t) < 4:  # standalone page numbers
            continue
        cleaned_lines.append(t)
    return "\n".join(cleaned_lines)


def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[Dict]:
    """Chunk cleaned text with LangChain RecursiveCharacterTextSplitter, keep metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for idx, chunk in enumerate(splitter.split_text(text)):
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": idx,
            "clean_text": chunk,
            "type": detect_block_type(chunk)
        })
    return chunks


def detect_block_type(text_block: str) -> str:
    """Classify block type by regex rules."""
    if re.search(r"^\s*(Q\.|A\.)", text_block, re.MULTILINE):
        return "testimony"
    if "APPEARANCES" in text_block or re.match(r"FOR .*?:", text_block):
        return "appearance"
    if re.search(r"\bINDEX\b", text_block):
        return "index"
    if re.search(r"UNITED STATES DISTRICT COURT|DEPOSITION TRANSCRIPT", text_block, re.IGNORECASE):
        return "header"
    return "body"


def extract_qna_chunks(text: str) -> List[Dict]:
    """
    Extract Q&A based chunks from deposition text.
    Each chunk contains 3 Q&A pairs: previous, current, and next for context.
    """
    chunks = []
    lines = text.split('\n')
    
    # Find all Q&A pairs
    qna_pairs = []
    current_q = ""
    current_a = ""
    current_q_line = None
    current_a_line = None
    mode = None  # 'q' or 'a' or None
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is just "Q."
        if line == 'Q.':
            # Save previous Q&A pair if we have both
            if current_q.strip() and current_a.strip():
                qna_pairs.append({
                    'q': current_q.strip(),
                    'a': current_a.strip(),
                    'q_line': current_q_line,
                    'a_line': current_a_line,
                    'line_range': (current_q_line, current_a_line)
                })
            
            # Start new question
            current_q = ""
            current_a = ""
            current_q_line = line_num + 1
            current_a_line = None
            mode = 'q'
            
        # Check if this line is just "A."
        elif line == 'A.':
            mode = 'a'
            current_a_line = line_num + 1
            
        # Skip footer patterns and other junk
        elif any(re.search(p, line) for p in [r'\(\d{3}\) \d{3}-\d{4}', r'[A-F0-9-]{36}', r'MR\. \w+:', r'MS\. \w+:', r'Objection\.']):
            continue
            
        # Add content to current question or answer
        elif mode == 'q':
            if current_q:
                current_q += " " + line
            else:
                current_q = line
        elif mode == 'a':
            if current_a:
                current_a += " " + line
            else:
                current_a = line
            current_a_line = line_num + 1
    
    # Don't forget the last Q&A pair
    if current_q and current_a:
        qna_pairs.append({
            'q': current_q,
            'a': current_a,
            'q_line': current_q_line,
            'a_line': current_a_line,
            'line_range': (current_q_line, current_a_line)
        })
    
    print(f"  - Found {len(qna_pairs)} Q&A pairs")
    
    # Create chunks with 3 Q&A context (previous, current, next)
    for i, qna in enumerate(qna_pairs):
        chunk_qnas = []
        start_line = qna['q_line']
        end_line = qna['a_line']
        
        # Add previous Q&A for context (if exists)
        if i > 0:
            prev_qna = qna_pairs[i-1]
            chunk_qnas.append(f"Q. {prev_qna['q']}")
            chunk_qnas.append(f"A. {prev_qna['a']}")
            start_line = prev_qna['q_line']
        
        # Add current Q&A
        chunk_qnas.append(f"Q. {qna['q']}")
        chunk_qnas.append(f"A. {qna['a']}")
        
        # Add next Q&A for context (if exists)
        if i < len(qna_pairs) - 1:
            next_qna = qna_pairs[i+1]
            chunk_qnas.append(f"Q. {next_qna['q']}")
            chunk_qnas.append(f"A. {next_qna['a']}")
            end_line = next_qna['a_line']
        
        chunk_text = "\n\n".join(chunk_qnas)
        
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i,
            "clean_text": chunk_text,
            "type": "testimony",
            "qna_count": len(chunk_qnas) // 2,  # Number of Q&A pairs
            "central_qna": i,  # Index of the central Q&A pair
            "start_line": start_line,
            "end_line": end_line,
            "context": "3qna" if i > 0 and i < len(qna_pairs) - 1 else "partial"
        }
        chunks.append(chunk)
    
    return chunks

def process_pdf(pdf_path: str, chunk_size=1000, overlap=200, use_qna_chunking=True):
    """Full pipeline: extract → clean → chunk."""
    raw_text = extract_pdf_text(pdf_path)
    cleaned = clean_text(raw_text)
    
    if use_qna_chunking:
        print("  - Using Q&A-aware chunking for deposition")
        return extract_qna_chunks(cleaned)
    else:
        print("  - Using standard text chunking")
        return chunk_text(cleaned, chunk_size, overlap)
