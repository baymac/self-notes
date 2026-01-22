"""Simple file-based vector store using numpy. No external vector DB dependencies."""
import json
import numpy as np
from pathlib import Path
from typing import Optional


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.embeddings_file = path / "embeddings.npy"
        self.metadata_file = path / "metadata.json"
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: list[dict] = []
        self._load()

    def _load(self):
        if self.embeddings_file.exists() and self.metadata_file.exists():
            self.embeddings = np.load(self.embeddings_file)
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)

    def _save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(self.embeddings_file, self.embeddings)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def clear(self):
        self.embeddings = None
        self.metadata = []
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def add(self, embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        new_embeddings = np.array(embeddings, dtype=np.float32)

        # Combine metadata with document text
        new_metadata = []
        for doc, meta in zip(documents, metadatas):
            entry = {**meta, "document": doc}
            new_metadata.append(entry)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.metadata = new_metadata
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.metadata.extend(new_metadata)

        self._save()

    def query(self, query_embedding: list[float], n_results: int = 4) -> dict:
        if self.embeddings is None or len(self.metadata) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = np.dot(self.embeddings, query_vec) / (norms * query_norm + 1e-10)

        # Get top k indices
        k = min(n_results, len(self.metadata))
        top_indices = np.argsort(similarities)[-k:][::-1]

        documents = [self.metadata[i]["document"] for i in top_indices]
        metadatas = [{k: v for k, v in self.metadata[i].items() if k != "document"} for i in top_indices]
        distances = [1 - similarities[i] for i in top_indices]  # Convert similarity to distance

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }

    def get_all_metadata(self) -> list[dict]:
        return [{k: v for k, v in m.items() if k != "document"} for m in self.metadata]

    def count(self) -> int:
        return len(self.metadata)
