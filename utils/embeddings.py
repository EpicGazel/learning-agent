from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from time import sleep


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def get_embedding(text: str, model="text-embedding-ada-002", max_retries=3) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            embeddings = OpenAIEmbeddings(model=model)
            result = embeddings.embed_query(text.replace("\n", " "))
            return np.array(result)

        except Exception as e:
            if attempt < max_retries - 1:
                sleep(1)
            else:
                raise e
