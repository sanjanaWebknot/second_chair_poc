import fitz
import re
import uuid

# Patterns for repetitive headers/footers you don’t want
FOOTER_PATTERNS = [
    r"Vivian Dafoulas.*",        # reporter info
    r"CDALEDEP\d+",              # Bates number
    r"[A-F0-9-]{36}",            # UUID-like strings
]

def is_footer_or_header(text: str) -> bool:
    """Return True if the line matches a known header/footer pattern."""
    return any(re.search(p, text) for p in FOOTER_PATTERNS)


def extract_pages_with_lines(pdf_path: str):
    """
    Extract text by page, keeping line numbers and page numbers,
    but removing common headers/footers.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        lines = []
        for raw_line in raw_text.splitlines():
            text = raw_line.strip()

            # Skip empty lines
            if not text:
                continue

            # Skip headers/footers
            if is_footer_or_header(text):
                continue

            # Skip standalone page numbers (small integers alone)
            if text.isdigit() and len(text) < 4:
                continue

            # Try to capture line numbers at start
            m = re.match(r"^\s*(\d{1,3})\s+(.*)$", raw_line)
            if m:
                line_no = int(m.group(1))
                text = m.group(2).strip()
            else:
                line_no = None

            if text:
                lines.append({
                    "page": i + 1,         # 1-based page index
                    "line_no": line_no,    # can be None
                    "text": text
                })

        pages.append({
            "page_num": i + 1,
            "lines": lines
        })
    return pages


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


def chunk_by_line_ranges(pages, lines_per_chunk=10):
    """
    Group lines into chunks of N lines, preserving metadata.
    Returns chunks with both raw and cleaned text.
    Improved to try to respect sentence boundaries when possible.
    """
    chunks = []
    for p in pages:
        # Only use lines with numbers (skip headers, etc.)
        lines = [l for l in p["lines"] if l["line_no"] is not None]
        if not lines:
            continue

        i = 0
        while i < len(lines):
            # Start with target chunk size
            end_idx = min(i + lines_per_chunk, len(lines))
            window = lines[i:end_idx]
            
            # If we're not at the end of lines, try to extend to complete sentences
            if end_idx < len(lines):
                # Look ahead up to 5 more lines to find sentence ending
                for j in range(end_idx, min(end_idx + 5, len(lines))):
                    line_text = lines[j]["text"]
                    if line_text.endswith(('.', '!', '?', '"')) or line_text.startswith(('Q.', 'A.')):
                        end_idx = j + 1
                        break
                window = lines[i:end_idx]
            
            raw_text_block = "\n".join([f"{l['line_no']}: {l['text']}" for l in window])

            # Remove leading line numbers for clean text
            clean_text_block = re.sub(r"^\d+:\s*", "", raw_text_block, flags=re.MULTILINE)

            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "page": p["page_num"],
                "start_line": window[0]["line_no"],
                "end_line": window[-1]["line_no"],
                "raw_text": raw_text_block,
                "clean_text": clean_text_block,
                "type": detect_block_type(clean_text_block)
            }
            chunks.append(chunk)
            
            # Move to next chunk
            i = end_idx
    return chunks
