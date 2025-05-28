import chromadb
import numpy as np
from typing import List, Tuple
from config_loader import load_config

config = load_config()

DB_PATH = config["vector_store"]["db_path"]
COLLECTION_NAME = config["vector_store"]["collection_name"]
SIMILARITY_METRIC = config["vector_store"]["similarity_metric"]
CLEAR_EXISTING = config["vector_store"]["clear_existing"]
TOP_K = config["retrieval"]["top_k"]
MIN_SCORE = config["retrieval"]["min_score"]
MAX_BATCH_SIZE = config["vector_store"]["max_batch_size"]
ADAPTIVE_TOP_K = config["retrieval"]["adaptive_top_k"]
STRATEGY = config["retrieval"]["strategy"]

chroma_client = chromadb.PersistentClient(path=DB_PATH)

existing_collections = chroma_client.list_collections()
if CLEAR_EXISTING and COLLECTION_NAME in existing_collections:
    print(f"Deleting existing collection '{COLLECTION_NAME}' to avoid dimension conflicts...")
    chroma_client.delete_collection(COLLECTION_NAME)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": SIMILARITY_METRIC}
)


def store_embeddings(text_chunks: List[str], embeddings: List[np.ndarray], sources: List[str]):
    global collection

    if CLEAR_EXISTING:
        existing_data = collection.get()
        existing_ids = existing_data.get("ids", []) if existing_data else []
        if existing_ids:
            print(f"Clearing {len(existing_ids)} old embeddings from ChromaDB...")

            for i in range(0, len(existing_ids), 5000):
                batch_ids = existing_ids[i:i + 5000]
                collection.delete(ids=batch_ids)

            print("Old data cleared.")
        else:
            print("No existing embeddings found. Skipping deletion.")

    total_chunks = len(text_chunks)

    for i in range(0, total_chunks, MAX_BATCH_SIZE):
        batch_texts = text_chunks[i:i + MAX_BATCH_SIZE]
        batch_embeddings = embeddings[i:i + MAX_BATCH_SIZE]
        batch_ids = [str(i + j) for j in range(len(batch_texts))]

        batch_sources = sources[i:i + MAX_BATCH_SIZE]
        metadatas = [{"text": chunk, "source": source} for chunk, source in zip(batch_texts, batch_sources)]

        collection.add(
            ids=batch_ids,
            embeddings=[emb.tolist() for emb in batch_embeddings],
            metadatas=metadatas
        )

        print(f"Stored batch {i + len(batch_texts)}/{total_chunks} in ChromaDB.")

    print(f"Successfully stored {total_chunks} embeddings in ChromaDB.")


def retrieve_similar_text(query_embedding: np.ndarray, filters: dict = None) -> List[Tuple[str, float]]:
    all_ids = collection.get(where=filters)['ids'] if filters else collection.get()['ids']
    total_chunks = len(all_ids)

    n_results = _determine_n_results(total_chunks)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=filters
    )

    if not results["ids"] or not results["metadatas"][0]:
        return [(" No relevant results found.", 0.0)]

    raw_results = list(zip(results["metadatas"][0], results["distances"][0]))
    raw_results.sort(key=lambda x: x[1])

    seen_sources = set()
    diverse_results = []
    for meta, dist in raw_results:
        similarity = 1 - dist
        if similarity < MIN_SCORE:
            continue

        source = meta.get("source", "unknown")
        if source not in seen_sources:
            diverse_results.append((meta["text"], 1 - dist))
            seen_sources.add(source)

        if len(diverse_results) >= TOP_K:
            break

    return diverse_results if diverse_results else [("No diverse results found.", 0.0)]


def _determine_n_results(total_chunks: int) -> int:
    if STRATEGY == "top_k":
        if ADAPTIVE_TOP_K:
            return min(int(total_chunks * 0.01 * TOP_K), total_chunks)
        else:
            return TOP_K
    elif STRATEGY == "threshold":
        return min(total_chunks, 100)
    else:
        raise ValueError("Invalid retrieval strategy.")
