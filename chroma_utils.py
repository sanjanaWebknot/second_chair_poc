import os
import chromadb
from embeddings import embed_texts

def create_chroma_client(persist_directory):
    return chromadb.PersistentClient(path=persist_directory)

def upsert_chunks_to_chroma(chroma_client, collection_name, chunks, embedder, source_file):
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"source": "second_chair_poc"}
        )

    ids = [c.get("chunk_id") for c in chunks]

    metadatas = [{
        "page": c.get("page"),
        "start_line": c.get("start_line"),
        "end_line": c.get("end_line"),
        "type": c.get("type"),
        "raw_text": c.get("raw_text"),
        "source_file": os.path.basename(source_file)
    } for c in chunks]

    # Use clean_text if available, else fall back to raw_text
    documents = [c.get("clean_text") or c.get("raw_text") for c in chunks]

    embeddings = embed_texts(embedder, documents)

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        embeddings=embeddings
    )
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
