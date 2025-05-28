import openai
import numpy as np
import concurrent.futures
import time
import os
from typing import List

from dotenv import load_dotenv
from config_loader import load_config
from vector_store import store_embeddings
from extractor import extract_text_from_pdf, extract_text_from_ppt, extract_text_from_website, \
    extract_text_from_docx
from chunker import chunk_text_by_sentence

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

config = load_config()

BATCH_SIZE = config["embeddings"]["batch_size"]
MAX_WORKERS = config["embeddings"]["max_workers"]
EMBEDDING_MODEL = config["embeddings"]["embedding_model"]
WEBSITES = config["web_extraction"]["websites"]


def generate_embeddings(text_chunks: List[str]) -> List[np.ndarray]:
    embeddings = []
    total_chunks = len(text_chunks)
    start_time = time.time()

    def embed_batch(batch):
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        return [np.array(emb.embedding) for emb in response.data]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(embed_batch, text_chunks[i:i + BATCH_SIZE]): i
                           for i in range(0, total_chunks, BATCH_SIZE)}

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                embeddings.extend(future.result())
                print(f"Embedded {batch_index + BATCH_SIZE}/{total_chunks} chunks.")
            except Exception as e:
                print(f"API Error: {e}")

    print(f"Embedding complete in {time.time() - start_time:.2f} seconds.")
    return embeddings


def extract_all_text():
    folders = {
        "pdf": ("documents/pdf/", extract_text_from_pdf),
        "docx": ("documents/docx/", extract_text_from_docx),
        "pptx": ("documents/pptx/", extract_text_from_ppt)
    }

    extracted_texts = []

    for file_ext, (folder_path, extractor) in folders.items():
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (folder not found).")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith(f".{file_ext}"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    text = extractor(file_path)
                    print(f"Extracted {len(text)} characters from {file_name}.")
                    extracted_texts.append(text)
                except Exception as e:
                    print(f"Error extracting from {file_name}: {e}")

    for url in WEBSITES:
        try:
            text = extract_text_from_website(url)
            print(f"Extracted {len(text)} characters from {url}.")
            extracted_texts.append(text)
        except Exception as e:
            print(f"Error extracting from {url}: {e}")

    return "\n\n".join(extracted_texts)


def extract_all_text_and_sources():
    folders = {
        "pdf": ("documents/pdf/", extract_text_from_pdf),
        "docx": ("documents/docx/", extract_text_from_docx),
        "pptx": ("documents/pptx/", extract_text_from_ppt)
    }

    all_chunks = []
    all_sources = []

    for file_ext, (folder_path, extractor) in folders.items():
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (folder not found).")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith(f".{file_ext}"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    text = extractor(file_path)
                    print(f"Extracted {len(text)} characters from {file_name}.")
                    chunks = chunk_text_by_sentence(text)
                    all_chunks.extend(chunks)
                    all_sources.extend([file_name] * len(chunks))
                except Exception as e:
                    print(f"Error extracting from {file_name}: {e}")

    for url in WEBSITES:
        try:
            text = extract_text_from_website(url)
            print(f"Extracted {len(text)} characters from {url}.")
            chunks = chunk_text_by_sentence(text)
            all_chunks.extend(chunks)
            all_sources.extend([url] * len(chunks))
        except Exception as e:
            print(f"Error extracting from {url}: {e}")

    return all_chunks, all_sources


if __name__ == "__main__":
    print("Starting text extraction...")

    text_chunks, sources = extract_all_text_and_sources()
    print(f"Split text into {len(text_chunks)} chunks from {len(set(sources))} sources.")

    embeddings = generate_embeddings(text_chunks)

    store_embeddings(text_chunks, embeddings, sources)

    print("Embeddings successfully stored in ChromaDB!")
