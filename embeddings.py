from sentence_transformers import SentenceTransformer


def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def embed_texts(embedder, texts, verbose=False):
    if verbose:
        print(f"      Encoding {len(texts)} texts")
        for i, text in enumerate(texts[:2]):  # Show first 2 texts
            print(f"      Text {i+1}: '{text[:80]}{'...' if len(text) > 80 else ''}'")
    
    embeddings = embedder.encode(
        texts, show_progress_bar=verbose, convert_to_numpy=True
    ).tolist()
    
    if verbose:
        print(f"      Generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
    
    return embeddings
