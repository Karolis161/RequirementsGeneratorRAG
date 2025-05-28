from typing import List

import openai
import random
import numpy as np
import os
from dotenv import load_dotenv
from config_loader import load_config
from vector_store import retrieve_similar_text
from prompts import PROMPT_TEMPLATES
from evaluation_metrics import (
    compute_retrieval_quality,
    compute_faithfulness,
    compute_recall_at_k
)
from sentence_transformers import SentenceTransformer, util

config = load_config()
GPT_MODEL = config["gpt"]["model"]
EMBEDDING_MODEL = config["embeddings"]["embedding_model"]
TEMPERATURE = config["retrieval"]["temperature"]
RETRIEVAL_ENABLED = config["retrieval"]["retrieval_enabled"]
INJECT_NOISE = config["retrieval"]["inject_noise"]
POSITION = config["retrieval"]["position"]
CONTEXT_MAX_TOKENS = config["retrieval"]["context_max_tokens"]
PROMPT_STYLE = config["prompt"]["style"]

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> np.ndarray:
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    embedding_vector = np.array(response.data[0].embedding)
    return embedding_vector / np.linalg.norm(embedding_vector)


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def generate_requirements(query: str, retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return "No relevant text retrieved to generate requirements."

    unique_chunks = []
    embeddings = semantic_model.encode([text for text, _ in retrieved_chunks])
    seen_indices = set()

    for i in range(len(retrieved_chunks)):
        if i not in seen_indices:
            unique_chunks.append(retrieved_chunks[i])
            for j in range(i + 1, len(retrieved_chunks)):
                if j not in seen_indices:
                    similarity = util.cos_sim(embeddings[i], embeddings[j])[0][0]
                    if similarity > 0.95:
                        seen_indices.add(j)

    context_chunks = [f"- {chunk}" for chunk, _ in unique_chunks]

    current_tokens = 0
    trimmed_chunks = []
    for chunk in context_chunks:
        chunk_tokens = estimate_token_count(chunk)
        if current_tokens + chunk_tokens > CONTEXT_MAX_TOKENS:
            break
        trimmed_chunks.append(chunk)
        current_tokens += chunk_tokens

    context = "\n".join(trimmed_chunks)

    if PROMPT_STYLE not in PROMPT_TEMPLATES:
        return f"Unknown prompt style: {PROMPT_STYLE}"

    base_prompt = PROMPT_TEMPLATES[PROMPT_STYLE](query, context)

    if POSITION == "before":
        prompt = base_prompt
    else:
        prompt = base_prompt.replace("**Query:**", "**Retrieved Text:** [AFTER-INSERT]").replace("**Retrieved Text:**",
                                                                                                 "**Query:**").replace(
            "[AFTER-INSERT]", "**Query:**")

    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI software analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = "Identify at least 5 measurable requirements (both functional and non-functional) for payment authorization and authentication workflows."

    print(f"Query: {query}")

    query_embedding = get_embedding(query)
    similar_texts = []
    if RETRIEVAL_ENABLED:
        similar_texts = retrieve_similar_text(query_embedding)
    else:
        print("Retrieval disabled. Skipping similarity search.")
        similar_texts = [("", 1.0)]

    final_output = generate_requirements(query, similar_texts)

    print("\nGenerated Functional & Non-Functional Requirements:")
    print(final_output)

    from evaluation_metrics import (
        count_used_documents,
        compute_token_length,
        compute_technical_term_coverage,
        average_cosine_similarity
    )

    retrieved_texts = [text for text, score in similar_texts]


    def extract_quotes_from_output(text: str, min_word_count: int = 4) -> List[str]:
        import re
        quotes = re.findall(r'\*"(.*?)"\*', text)
        return [q for q in quotes if len(q.split()) >= min_word_count]


    relevant_docs = extract_quotes_from_output(final_output)

    generated_lines = [line.strip() for line in final_output.split("\n") if
                       line.strip().startswith("1.") or line.strip()[0:2].isdigit()]
    similarity_score = average_cosine_similarity(generated_lines, retrieved_texts)

    doc_count = count_used_documents(similar_texts)
    token_length = compute_token_length(final_output)
    specificity = compute_technical_term_coverage(final_output)
    faithfulness = compute_faithfulness(final_output, retrieved_texts)
    retrieval_quality = compute_retrieval_quality(query, retrieved_texts)
    recall_at_k = compute_recall_at_k(relevant_docs, retrieved_texts)

    print("\n--- Evaluation Metrics ---")
    print(f"Answer Relevance: {similarity_score:.3f}")
    print(f"Faithfulness: {faithfulness:.3f}")
    print(f"Recall@k: {recall_at_k:.3f}")
    print(f"Technical Term Coverage: {specificity}")
