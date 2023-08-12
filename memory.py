from datetime import datetime, timedelta, tzinfo
import numpy as np
from uuid import UUID, uuid4
import utils.embeddings
import math
import pytz

IMPORTANCE_WEIGHT = 1
SIMILARITY_WEIGHT = 1
RECENCY_WEIGHT = 1


class Memory:
    @property
    def recency(self) -> float:
        last_retrieved_hours_ago = (datetime.now(tz=pytz.utc) - self.last_accessed) / timedelta(hours=1)

        decay_factor = 0.99
        return math.pow(decay_factor, last_retrieved_hours_ago)

    def __init__(self, person, description, embedding, creation_time=datetime.now(tz=pytz.utc)):
        self.id = uuid4()
        self.person = person
        self.description = description
        self.created_at = creation_time
        self.embedding: np.ndarray = embedding
        self.importance: int = 0
        self.last_accessed = creation_time
        self.related_memory_ids: list[UUID] = []

    def __str__(self):
        return f"[{self.person}] - {self.description} ({round(self.importance, 1)})"

    def update_last_accessed(self):
        self.last_accessed = datetime.now(tz=pytz.utc)

    def similarity(self, query: str) -> float:
        query_embedding = utils.embeddings.get_embedding(query)
        return utils.embeddings.cosine_similarity(self.embedding, query_embedding)

    def relevance(self, query: str) -> float:
        return (IMPORTANCE_WEIGHT * self.importance
                + SIMILARITY_WEIGHT * self.similarity(query)
                + RECENCY_WEIGHT * self.recency)


class RelatedMemory:
    memory: Memory
    relevance: float

    def __init__(self, memory: Memory, relevance: float):
        self.memory = memory
        self.relevance = relevance

    def __str__(self) -> str:
        return f"Memory: {self.memory.description}, Relevance: {self.relevance}"


def get_relevant_memories(query: str, memories: list[Memory], k: int = 5) -> list[Memory]:
    """Returns a list of the top k most relevant memories based on the query string."""
    memories_with_relevance = [
        RelatedMemory(memory=memory, relevance=memory.relevance(query))
        for memory in memories
    ]

    # Sort the list of dictionaries based on the 'relevance' key in descending order
    sorted_by_relevance = sorted(
        memories_with_relevance, key=lambda x: x.relevance, reverse=True
    )

    # get the top k memories, as a list of SingleMemory object
    top_memories = [memory.memory for memory in sorted_by_relevance[:k]]

    # now sort the list based on the created_at field, with the oldest memories first
    sorted_by_created_at = sorted(
        top_memories, key=lambda x: x.created_at, reverse=False
    )

    # update last accessed times
    for memory in sorted_by_created_at:
        memory.update_last_accessed()

    return sorted_by_created_at
