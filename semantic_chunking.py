"""
Semantic text chunking using semantic-text-splitter for meaningful content chunks.
"""

import uuid
import re
try:
    from semantic_text_splitter import TextSplitter
    SEMANTIC_SPLITTER_AVAILABLE = True
except ImportError:
    SEMANTIC_SPLITTER_AVAILABLE = False
    print("⚠️  semantic-text-splitter not installed. Using fallback chunking.")

def detect_block_type(text_block: str) -> str:
    """Classify block type by regex rules for legal depositions."""
    # Q&A sections
    if re.search(r'^\s*(Q\.|A\.)', text_block, re.MULTILINE):
        return "testimony"
    
    # Appearances section
    if "APPEARANCES" in text_block or re.match(r"FOR .*?:", text_block):
        return "appearance"
    
    # Index/table of contents
    if re.search(r'\bINDEX\b', text_block):
        return "index"
    
    # Court headers
    if re.search(r"UNITED STATES DISTRICT COURT|DEPOSITION TRANSCRIPT", text_block, re.IGNORECASE):
        return "header"
    
    # Certificate/signature pages
    if re.search(r"CERTIFICATE|NOTARY|SIGNATURE", text_block, re.IGNORECASE):
        return "certificate"
    
    # Exhibit markers
    if re.search(r"EXHIBIT|MARKED FOR IDENTIFICATION", text_block, re.IGNORECASE):
        return "exhibit"
    
    return "body"

def semantic_chunk_pages(pages, chunk_size=1000, overlap=100):
    """
    Use semantic text splitter to create meaningful chunks from deposition pages.
    
    Args:
        pages: List of page objects with lines
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of chunk objects with metadata
    """
    
    if not SEMANTIC_SPLITTER_AVAILABLE:
        print("⚠️  Falling back to simple text chunking")
        return fallback_chunk_pages(pages, chunk_size)
    
    # Initialize semantic splitter
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=chunk_size, overlap=overlap)
    
    chunks = []
    
    for page in pages:
        # Only use lines with numbers (skip headers, etc.)
        lines = [l for l in page["lines"] if l["line_no"] is not None]
        if not lines:
            continue
        
        # Combine all lines from the page into one text block
        page_text = "\n".join([l["text"] for l in lines])
        
        if not page_text.strip():
            continue
        
        # Use semantic splitter to split the page text
        try:
            semantic_chunks = splitter.chunks(page_text)
            
            for i, chunk_text in enumerate(semantic_chunks):
                if not chunk_text.strip():
                    continue
                
                # Try to determine line range for this chunk
                chunk_lines = chunk_text.split('\n')
                start_line = None
                end_line = None
                
                # Find the line numbers by matching text
                for line in lines:
                    if chunk_lines[0].strip() in line["text"]:
                        start_line = line["line_no"]
                        break
                
                for line in reversed(lines):
                    if chunk_lines[-1].strip() in line["text"]:
                        end_line = line["line_no"]
                        break
                
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "page": page["page_num"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "raw_text": chunk_text,
                    "clean_text": chunk_text.strip(),
                    "type": detect_block_type(chunk_text),
                    "chunk_method": "semantic",
                    "chunk_index": i
                }
                chunks.append(chunk)
                
        except Exception as e:
            print(f"⚠️  Error with semantic splitting on page {page['page_num']}: {e}")
            # Fallback to simple chunking for this page
            fallback_chunks = fallback_chunk_page(page, chunk_size)
            chunks.extend(fallback_chunks)
    
    return chunks

def fallback_chunk_pages(pages, chunk_size=1000):
    """
    Fallback chunking method when semantic splitter is not available.
    Uses sentence-aware chunking instead of line-based.
    """
    chunks = []
    
    for page in pages:
        page_chunks = fallback_chunk_page(page, chunk_size)
        chunks.extend(page_chunks)
    
    return chunks

def fallback_chunk_page(page, chunk_size=1000):
    """
    Chunk a single page using sentence-aware splitting.
    """
    chunks = []
    
    # Only use lines with numbers (skip headers, etc.)
    lines = [l for l in page["lines"] if l["line_no"] is not None]
    if not lines:
        return chunks
    
    # Combine all text from the page
    page_text = " ".join([l["text"] for l in lines])
    
    if not page_text.strip():
        return chunks
    
    # Split by sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', page_text)
    
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            
            # Try to find line range
            start_line = lines[0]["line_no"] if lines else None
            end_line = lines[-1]["line_no"] if lines else None
            
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "page": page["page_num"],
                "start_line": start_line,
                "end_line": end_line,
                "raw_text": chunk_text,
                "clean_text": chunk_text,
                "type": detect_block_type(chunk_text),
                "chunk_method": "fallback_sentence",
                "chunk_index": len(chunks)
            }
            chunks.append(chunk)
            
            # Start new chunk
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        
        start_line = lines[0]["line_no"] if lines else None
        end_line = lines[-1]["line_no"] if lines else None
        
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "page": page["page_num"],
            "start_line": start_line,
            "end_line": end_line,
            "raw_text": chunk_text,
            "clean_text": chunk_text,
            "type": detect_block_type(chunk_text),
            "chunk_method": "fallback_sentence",
            "chunk_index": len(chunks)
        }
        chunks.append(chunk)
    
    return chunks

def smart_chunk_pages(pages, chunk_size=1000, overlap=100, prefer_semantic=True):
    """
    Main chunking function that chooses the best available method.
    
    Args:
        pages: List of page objects
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        prefer_semantic: Whether to prefer semantic splitting when available
    
    Returns:
        List of chunk objects
    """
    if prefer_semantic and SEMANTIC_SPLITTER_AVAILABLE:
        print(f"  - Using semantic text splitter (chunk_size={chunk_size}, overlap={overlap})")
        return semantic_chunk_pages(pages, chunk_size, overlap)
    else:
        print(f"  - Using sentence-aware fallback chunking (chunk_size={chunk_size})")
        return fallback_chunk_pages(pages, chunk_size)
