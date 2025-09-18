import os
import chromadb
from embeddings import embed_texts

def create_chroma_client(persist_directory):
    return chromadb.PersistentClient(path=persist_directory)

def upsert_chunks_to_chroma(chroma_client, collection_name, chunks, embedder, source_file, verbose=True):
    if verbose:
        print(f"    Preparing to embed {len(chunks)} chunks...")
    
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        if verbose:
            print(f"    Using existing collection '{collection_name}'")
        collection = chroma_client.get_collection(name=collection_name)
    else:
        if verbose:
            print(f"    Creating new collection '{collection_name}'")
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"source": "second_chair_poc"}
        )

    ids = [c.get("chunk_id") for c in chunks]

    metadatas = [{
        "page": c.get("page") or 1,
        "start_line": c.get("start_line") or 1,
        "end_line": c.get("end_line") or 1,
        "type": c.get("type") or "unknown",
        "chunk_method": c.get("chunk_method") or "qna_chunking",
        "chunk_index": c.get("chunk_index") or 0,
        "qna_count": c.get("qna_count") or 1,
        "central_qna": c.get("central_qna") or 0,
        "context": c.get("context") or "unknown",
        "source_file": os.path.basename(source_file)
    } for c in chunks]

    # Use clean_text if available, else fall back to raw_text
    documents = [c.get("clean_text") or c.get("raw_text") for c in chunks]
    
    if verbose:
        print(f"    Sample documents being embedded:")
        for i, doc in enumerate(documents[:3]):
            print(f"      Doc {i+1}: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")

    if verbose:
        print(f"    Generating embeddings for {len(documents)} documents...")
    embeddings = embed_texts(embedder, documents, verbose=verbose)

    if verbose:
        print(f"    Storing {len(embeddings)} embeddings in ChromaDB...")
    
    try:
        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            embeddings=embeddings
        )
        if verbose:
            print(f"    Successfully stored all chunks!")
    except Exception as e:
        if verbose:
            print(f"    Error storing chunks: {e}")
        raise
    
    return collection

def query_negative_statement(chroma_client, collection_name, query_text, embedder, top_k=5):
    collection = chroma_client.get_collection(name=collection_name)
    q_vec = embed_texts(embedder, [query_text])[0]
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
            "document": results["documents"][0][i]
        })
    return hits
