import os
import pickle
import numpy as np
import faiss
from fastembed import TextEmbedding

# ---------- FIXED PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

class Retriever:
    def __init__(self):
        self.chunk_data = pickle.load(open(os.path.join(OUTPUT_DIR, "chunked_data.pkl"), "rb"))
        self.bm25 = pickle.load(open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "rb"))
        self.faiss_index = faiss.read_index(os.path.join(OUTPUT_DIR, "vector_index.faiss"))
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def query_router(self, query):
        q = query.lower()
        if any(term in q for term in ["contact", "email", "reach out", "coordinator"]):
            return "contacts"
        elif any(term in q for term in ["event", "this week", "seminar", "fair"]):
            return "events"
        elif any(term in q for term in ["credits", "minor", "requirement", "prerequisite"]):
            return "requirements"
        elif any(term in q for term in ["club", "organization", "society"]):
            return "student_life"
        return "general"

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        scores = {}
        for rank, idx in enumerate(dense_results):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += 1 / (k + rank + 1)
        for rank, idx in enumerate(sparse_results):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += 1 / (k + rank + 1)
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices

    def retrieve(self, query, top_k=5, router_override=True):
        intent = self.query_router(query)

        # 1. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        if router_override and intent != "general":
            for i, chunk in enumerate(self.chunk_data):
                if chunk["category"] == intent:
                    bm25_scores[i] *= 1.5

        sparse_top_n = np.argsort(bm25_scores)[::-1][:15]

        # 2. Dense Search (FAISS)
        emb = np.array(list(self.model.embed([query])))
        distances, dense_top_n = self.faiss_index.search(emb, 15)
        dense_top_n = dense_top_n[0]

        # 3. Combine via RRF
        fused_indices = self.reciprocal_rank_fusion(dense_top_n, sparse_top_n)
        final_indices = fused_indices[:top_k]

        results = [self.chunk_data[i] for i in final_indices]
        return results, intent

if __name__ == "__main__":
    r = Retriever()
    res, intent = r.retrieve("Who is the contact for the MITA program?")
    print(f"Router intent: {intent}")
    print("Top result:", res[0]['title'])