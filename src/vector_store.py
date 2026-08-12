import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self, chunks=None, path="data/vector_store", force_rebuild=False):
        self.path = path
        self.chunks = chunks or []

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.npy")

        has_saved_index = os.path.exists(index_path) and os.path.exists(chunks_path)

        if chunks and (force_rebuild or not has_saved_index):
            # New report content was given (or nothing is cached yet) ->
            # always rebuild from the current chunks, never reuse old data.
            self.build()

        elif has_saved_index:
            self.index = faiss.read_index(index_path)
            self.chunks = np.load(
                chunks_path,
                allow_pickle=True
            ).tolist()

        else:
            raise ValueError(
                "No chunks were provided and no saved vector store was found "
                f"at '{path}'. Pass `chunks=...` to build one."
            )

    def build(self):
        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

        os.makedirs(self.path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(self.path, "index.faiss")
        )

        np.save(
            os.path.join(self.path, "chunks.npy"),
            np.array(self.chunks, dtype=object)
        )

    def search(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                # FAISS returns -1 when there are fewer chunks than top_k;
                # skip these instead of wrapping to the last chunk.
                continue
            results.append({
                "score": float(score),
                "text": self.chunks[index]
            })

        return results