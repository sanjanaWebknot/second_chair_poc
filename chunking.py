import uuid


def chunk_turns(turns, chunk_chars, overlap_chars):
    chunks = []
    for idx, t in enumerate(turns):
        text = t["text"].strip()
        if not text:
            continue
        if len(text) <= chunk_chars:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": text,
                    "speaker": t["speaker"],
                    "page": t["page"],
                    "turn_index": idx,
                    "char_start": t["char_start"],
                    "char_end": t["char_end"],
                }
            )
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                sub = text[start:end].strip()
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "text": sub,
                        "speaker": t["speaker"],
                        "page": t["page"],
                        "turn_index": idx,
                        "char_start": t["char_start"] + start,
                        "char_end": t["char_start"] + end,
                    }
                )
                if end == len(text):
                    break
                start = max(0, end - overlap_chars)
    return chunks
