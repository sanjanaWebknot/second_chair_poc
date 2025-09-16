#!/usr/bin/env python3
"""
Terminal-based claim checker for deposition analysis.
Processes the Deposition_Joseph_Nadeau.pdf file and checks claims against it.
"""

import os
import sys

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pdf_utils import extract_pages_with_lines, chunk_by_line_ranges
from embeddings import get_sentence_transformer
from chroma_utils import (
    create_chroma_client,
    upsert_chunks_to_chroma,
    query_negative_statement,
)
from claim_checker import check_claim_with_ollama

# Configuration
PDF_FILE = "Deposition_Joseph_Nadeau.pdf"
CHROMA_PERSIST_DIR = "./chroma_db_second_chair"
COLLECTION_NAME = "second_chair_depositions"
LINES_PER_CHUNK = 30  # Increased chunk size for better context

def setup_database():
    """Extract PDF, chunk it, and store in Chroma database."""
    print("🔍 Setting up database...")
    
    if not os.path.exists(PDF_FILE):
        print(f"❌ Error: PDF file '{PDF_FILE}' not found!")
        return False
    
    print(f"📄 Processing {PDF_FILE}...")
    
    # Extract pages
    print("  - Extracting pages...")
    pages = extract_pages_with_lines(PDF_FILE)
    print(f"    Found {len(pages)} pages")
    
    # Create chunks
    print(f"  - Creating chunks ({LINES_PER_CHUNK} lines each)...")
    chunks = chunk_by_line_ranges(pages, lines_per_chunk=LINES_PER_CHUNK)
    print(f"    Created {len(chunks)} chunks")
    
    # Calculate chunk statistics
    chunk_lengths = [len(c.get("clean_text", "")) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"    Average chunk length: {avg_length:.0f} characters")
    
    # Load embedding model
    print("  - Loading embedding model...")
    embedder = get_sentence_transformer("all-MiniLM-L6-v2")
    
    # Create Chroma client and store chunks
    print("  - Storing in Chroma database...")
    chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
    upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, chunks, embedder, PDF_FILE)
    
    print("✅ Database setup complete!")
    return chroma_client, embedder

def check_claim(chroma_client, embedder, statement, top_k=5, model="llama3.1"):
    """Check a claim against the database and get Ollama's verdict."""
    print(f"\n🔍 Checking claim: '{statement}'")
    print(f"  - Searching for top {top_k} relevant chunks...")
    
    # Query the database
    hits = query_negative_statement(
        chroma_client, COLLECTION_NAME, statement, embedder, top_k=top_k
    )
    
    print(f"  - Found {len(hits)} relevant chunks")
    for i, hit in enumerate(hits):
        distance = hit['distance']
        relevance = 1 - distance
        page = hit['metadata'].get('page')
        lines = f"{hit['metadata'].get('start_line')}-{hit['metadata'].get('end_line')}"
        print(f"    {i+1}. Page {page}, Lines {lines} (relevance: {relevance:.3f})")
    
    # Send to Ollama
    print(f"\n🤖 Analyzing with Ollama ({model})...")
    verdict = check_claim_with_ollama(statement, hits, model=model)
    
    # Display results
    print(f"\n{'='*60}")
    print("📊 ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    verdict_emoji = {
        "SUPPORT": "🟢",
        "REFUTE": "🔴", 
        "NOT_FOUND": "🟡",
        "ERROR": "⚫",
        "UNKNOWN": "⚪"
    }.get(verdict['verdict'], "⚪")
    
    print(f"Verdict: {verdict_emoji} {verdict['verdict']}")
    print(f"Confidence: {verdict['confidence']}%")
    print(f"Explanation: {verdict['explanation']}")
    print(f"{'='*60}")
    
    return verdict

def main():
    """Main function to run the terminal claim checker."""
    print("🏛️  Second Chair - Terminal Claim Checker")
    print("=" * 50)
    
    # Check if database exists, if not create it
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("🆕 Database not found. Setting up...")
        result = setup_database()
        if not result:
            return
        chroma_client, embedder = result
    else:
        print("📚 Loading existing database...")
        try:
            chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
            embedder = get_sentence_transformer("all-MiniLM-L6-v2")
            print("✅ Database loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            print("🔄 Recreating database...")
            result = setup_database()
            if not result:
                return
            chroma_client, embedder = result
    
    # Interactive loop
    print("\n🎯 Ready to check claims!")
    print("Type 'quit' or 'exit' to stop, 'setup' to recreate database")
    
    while True:
        try:
            print("\n" + "-" * 50)
            statement = input("📝 Enter statement to check: ").strip()
            
            if statement.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if statement.lower() == 'setup':
                result = setup_database()
                if result:
                    chroma_client, embedder = result
                continue
            
            if not statement:
                print("⚠️  Please enter a statement to check.")
                continue
            
            # Ask for additional parameters
            try:
                top_k = input(f"🔢 Number of chunks to retrieve (default: 5): ").strip()
                top_k = int(top_k) if top_k else 5
            except ValueError:
                top_k = 5
            
            model = input(f"🤖 Ollama model (default: llama3.1): ").strip()
            model = model if model else "llama3.1"
            
            # Check the claim
            check_claim(chroma_client, embedder, statement, top_k, model)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
