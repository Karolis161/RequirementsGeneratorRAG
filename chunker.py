from typing import List
from transformers import GPT2TokenizerFast
from config_loader import load_config
from nltk.tokenize import sent_tokenize

config = load_config()
CHUNK_SIZE = config["data_processing"]["chunk_size"]
OVERLAP = config["data_processing"]["overlap"]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text.strip())
        start += chunk_size - overlap

    return chunks


def chunk_text_by_sentence(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)

        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                overlap_length = max(1, int(overlap * len(current_chunk) / max(1, current_length)))
                overlap_sentences = current_chunk[-overlap_length:]
                overlap_string = " ".join(overlap_sentences).strip()
                overlap_tokens = tokenizer.encode(overlap_string)

                current_chunk = [sentence]
                current_length = sentence_length + len(overlap_tokens)
            else:
                chunks.append(sentence.strip())
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks
