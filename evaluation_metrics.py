import numpy as np
from typing import List, Tuple
from transformers import GPT2TokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

KEYWORD_GROUPS = {
    "security": ["multi-factor authentication", "MFA", "two-factor authentication", "2FA", "step-up authentication",
                 "trusted platform module", "hardware root of trust", "cryptographic module", "session key"
                 ],
    "access": ["OAuth", "OAuth2", "OpenID", "OIDC", "SAML", "JWT", "access token", "refresh token", "scope",
               "authorization server", "identity provider", "IdP", "redirect URI", "authentication flow"],
    "compliance": ["PSD2", "GDPR", "eIDAS", "SOX", "HIPAA", "FFIEC", "FATF", "AML", "KYC", "EBA guidelines", "TSP",
                   "PCI"]
}


def count_used_documents(similar_chunks: List[Tuple[str, float]]) -> int:
    return len([text for text, score in similar_chunks if score > 0.0])


def compute_token_length(text: str) -> int:
    return len(tokenizer.encode(text))


def compute_technical_term_coverage(text: str) -> float:
    group_coverage_scores = []
    for category, keywords in KEYWORD_GROUPS.items():
        matched = sum(1 for kw in keywords if kw.lower() in text.lower())
        group_score = matched / len(keywords)
        group_coverage_scores.append(group_score)

    score = sum(group_coverage_scores) / len(group_coverage_scores)

    return round(score, 2)


def average_cosine_similarity(generated_lines: List[str], retrieved_chunks: List[str]) -> float:
    if not generated_lines or not retrieved_chunks:
        return 0.0

    retrieved_embeddings = embedding_model.encode(retrieved_chunks)
    requirement_embeddings = embedding_model.encode(generated_lines)

    sims = []
    for req_emb in requirement_embeddings:
        max_sim = max(cosine_similarity([req_emb], retrieved_embeddings)[0])
        sims.append(max_sim)

    return np.mean(sims)


def compute_retrieval_quality(query: str, retrieved_chunks: List[str]) -> float:
    if not retrieved_chunks:
        return 0.0
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(retrieved_chunks, convert_to_tensor=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    return float(sims.mean())


def compute_context_precision(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    if not ground_truth_chunks:
        return 0.0
    retrieved_embeddings = embedding_model.encode(retrieved_chunks, convert_to_tensor=True)
    gt_embeddings = embedding_model.encode(ground_truth_chunks, convert_to_tensor=True)

    precision_scores = []
    for i, gt in enumerate(gt_embeddings):
        sims = util.cos_sim(gt, retrieved_embeddings)[0]
        sorted_scores = sorted(sims, reverse=True)
        correct_at_k = [1 if s > 0.7 else 0 for s in sorted_scores[:len(retrieved_chunks)]]
        if any(correct_at_k):
            k = correct_at_k.index(1) + 1
            precision_scores.append(1 / k)
    return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0


def compute_context_recall(retrieved_chunks: List[str], ground_truth_claims: List[str]) -> float:
    if not ground_truth_claims:
        return 0.0
    context = " ".join(retrieved_chunks).lower()
    matched = sum(1 for claim in ground_truth_claims if claim.lower() in context)
    return matched / len(ground_truth_claims)


def compute_faithfulness(generated_answer: str, retrieved_chunks: List[str], threshold: float = 0.65) -> float:
    answer_sentences = [s.strip() for s in generated_answer.split("\n") if
                        len(s.strip()) > 15 and s.strip()[0].isdigit()]
    if not answer_sentences or not retrieved_chunks:
        return 0.0

    answer_embeddings = embedding_model.encode(answer_sentences, convert_to_tensor=True)
    context_embeddings = embedding_model.encode(retrieved_chunks, convert_to_tensor=True)

    faithful_count = 0
    for ans_emb in answer_embeddings:
        sims = util.cos_sim(ans_emb, context_embeddings)[0]
        if sims.max() >= threshold:
            faithful_count += 1

    return faithful_count / len(answer_sentences)


def compute_recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], threshold: float = 0.6) -> float:
    if not relevant_docs or not retrieved_docs:
        return 0.0

    relevant_embeddings = embedding_model.encode(relevant_docs, convert_to_tensor=True)
    retrieved_embeddings = embedding_model.encode(retrieved_docs, convert_to_tensor=True)

    matched = 0
    for rel_emb in relevant_embeddings:
        sims = util.cos_sim(rel_emb, retrieved_embeddings)[0]
        if sims.max() >= threshold:
            matched += 1

    recall = matched / len(relevant_docs)
    return recall
