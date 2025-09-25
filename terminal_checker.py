#!/usr/bin/env python3
"""
Terminal-based claim checker for deposition analysis.
Processes the Deposition_Joseph_Nadeau.pdf file and checks claims against it.
"""

import os
import sys

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pdf_utils import process_pdf
from embeddings import get_sentence_transformer
from chroma_utils import (
    create_chroma_client,
    upsert_chunks_to_chroma,
    query_negative_statement,
)
from claim_checker import check_claim_with_ollama_chain

# Configuration
PDF_FILE = "Deposition_Joseph_Nadeau.pdf"
CHROMA_PERSIST_DIR = "./chroma_db_second_chair"
COLLECTION_NAME = "second_chair_depositions"
CHUNK_SIZE = 1000  # Target chunk size in characters
CHUNK_OVERLAP = 100  # Overlap between chunks

def clear_database():
    """Clear the existing ChromaDB database."""
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Removing existing database at {CHROMA_PERSIST_DIR}")
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Database cleared!")

def setup_database(force_rebuild=False):
    """Extract PDF, chunk it, and store in Chroma database."""
    print("Setting up database...")
    
    if force_rebuild:
        clear_database()
    
    if not os.path.exists(PDF_FILE):
        print(f"Error: PDF file '{PDF_FILE}' not found!")
        return False
    
    print(f"Processing {PDF_FILE}...")
    
    # Extract and process PDF directly into chunks
    print("  - Processing PDF and creating chunks...")
    chunks = process_pdf(PDF_FILE, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"    Created {len(chunks)} chunks")
    
    # Calculate chunk statistics
    chunk_lengths = [len(c.get("clean_text", "")) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"    Average chunk length: {avg_length:.0f} characters")
    
    # Show what's being embedded (first few chunks)
    print(f"\nSHOWING FIRST 5 CHUNKS TO BE EMBEDDED:")
    print("=" * 80)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Page: {chunk.get('page')}, Lines: {chunk.get('start_line')}-{chunk.get('end_line')}")
        print(f"Type: {chunk.get('type')}")
        print(f"Length: {len(chunk.get('clean_text', ''))} chars")
        print("Content:")
        print(f"'{chunk.get('clean_text', '')[:300]}{'...' if len(chunk.get('clean_text', '')) > 300 else ''}")
    
    print("=" * 80)
    
    # Ask user if they want to see all chunks
    response = input(f"\nWant to see ALL {len(chunks)} chunks? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print(f"\nALL {len(chunks)} CHUNKS:")
        print("=" * 80)
        for i, chunk in enumerate(chunks):
            print(f"\n--- CHUNK {i+1} ---")
            print(f"Page: {chunk.get('page')}, Lines: {chunk.get('start_line')}-{chunk.get('end_line')}")
            print(f"Type: {chunk.get('type')}")
            print(f"Length: {len(chunk.get('clean_text', ''))} chars")
            print("Content:")
            print(f"'{chunk.get('clean_text', '')}'")
        print("=" * 80)
    
    # Load embedding model
    print("  - Loading embedding model...")
    embedder = get_sentence_transformer("all-MiniLM-L6-v2")
    
    # Create Chroma client and store chunks
    print("  - Storing in Chroma database...")
    try:
        chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
        upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, chunks, embedder, PDF_FILE, verbose=True)
    except Exception as e:
        print(f"Database error: {e}")
        print("Clearing database and retrying...")
        clear_database()
        chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
        upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, chunks, embedder, PDF_FILE, verbose=True)
    
    print("Database setup complete!")
    return chroma_client, embedder

def check_claim(chroma_client, embedder, statement, top_k=5, model="phi3:mini"):
    """Check a claim against the database and get Ollama's verdict."""
    print(f"\nChecking claim: '{statement}'")
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
        qna_count = hit['metadata'].get('qna_count', 1)
        context = hit['metadata'].get('context', 'unknown')
        print(f"    {i+1}. Page {page}, Lines {lines} ({qna_count} Q&A pairs, context: {context}, relevance: {relevance:.3f})")
    
    # Send to Ollama using LangChain chain
    print(f"\nAnalyzing with Ollama ({model}) using LangChain chain...")
    verdict = check_claim_with_ollama_chain(statement, hits, model=model)
    
    # Display results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    verdict_symbol = {
        "SUPPORT": "[SUPPORT]",
        "REFUTE": "[REFUTE]", 
        "NOT_FOUND": "[NOT_FOUND]",
        "ERROR": "[ERROR]",
        "UNKNOWN": "[UNKNOWN]"
    }.get(verdict.verdict, "[UNKNOWN]")
    
    print(f"Verdict: {verdict_symbol}")
    print(f"Confidence: {verdict.confidence}%")
    print(f"Explanation: {verdict.explanation}")
    print(f"{'='*60}")
    
    return verdict

def main():
    """Main function to run the terminal claim checker."""
    print("Second Chair - Terminal Claim Checker")
    print("=" * 50)
    
    # Check if database exists, if not create it
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("Database not found. Setting up...")
        result = setup_database()
        if not result:
            return
        chroma_client, embedder = result
    else:
        print("Loading existing database...")
        try:
            chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
            embedder = get_sentence_transformer("all-MiniLM-L6-v2")
            print("Database loaded successfully!")
        except Exception as e:
            print(f"Error loading database: {e}")
            print("Recreating database...")
            result = setup_database()
            if not result:
                return
            chroma_client, embedder = result
    
    # Interactive loop
    print("\nReady to check claims!")
    print("Commands: 'quit'/'exit' to stop, 'setup' to rebuild database, 'clear' to clear database")
    
    while True:
        try:
            print("\n" + "-" * 50)
            statement = input("Enter statement to check: ").strip()
            
            if statement.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if statement.lower() == 'setup':
                result = setup_database(force_rebuild=True)
                if result:
                    chroma_client, embedder = result
                continue
            
            if statement.lower() == 'clear':
                clear_database()
                print("Database cleared. Type 'setup' to rebuild.")
                continue
            
            if not statement:
                print("Please enter a statement to check.")
                continue
            
            # Ask for additional parameters
            try:
                top_k = input(f"Number of chunks to retrieve (default: 5): ").strip()
                top_k = int(top_k) if top_k else 5
            except ValueError:
                top_k = 5
            
            model = input(f"Ollama model (default: phi3:mini): ").strip()
            model = model if model else "phi3:mini"
            
            # Check the claim
            check_claim(chroma_client, embedder, statement, top_k, model)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
