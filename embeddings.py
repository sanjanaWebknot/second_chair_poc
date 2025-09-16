from sentence_transformers import SentenceTransformer


def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def embed_texts(embedder, texts):
    return embedder.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    ).tolist()
