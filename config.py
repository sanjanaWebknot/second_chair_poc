import os

# Paths
PDF_DEFAULT = "/mnt/data/Deposition_Joseph_Nadeau.pdf"
CHROMA_PERSIST_DIR = "./chroma_db_second_chair"

# Chunking
CHUNK_TOKENS = 300  # target tokens per chunk
OVERLAP_TOKENS = 75  # overlap tokens
TOKEN_CHAR_RATIO = 4  # approx: 1 token ~= 4 chars
CHUNK_CHARS = int(CHUNK_TOKENS * TOKEN_CHAR_RATIO)
OVERLAP_CHARS = int(OVERLAP_TOKENS * TOKEN_CHAR_RATIO)

COLLECTION_NAME = "second_chair_depositions"
