from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognizer:
    def __init__(self, embedder, database, threshold=0.5):
        self.embedder = embedder
        self.database = database
        self.threshold = threshold

    def _get_mean_embeddings(self):
        records = self.database.fetch_all_embeddings()
        grouped = defaultdict(list)

        cleaned = []
        for name, embs in grouped.items():
            try:
                arr = np.array(embs)
                if len(arr.shape) == 2 and arr.shape[1] == 512:
                    mean = np.mean(arr, axis=0)
                    cleaned.append((name, mean))
                else:
                    print(f"[!] Skipping {name}, invalid shape: {arr.shape}")
            except Exception as e:
                print(f"[!] Error processing {name}: {e}")
        return cleaned

    def recognize(self, face):
        new_embedding = self.embedder.get_embedding(face)
        mean_embeddings = self._get_mean_embeddings()
        if len(mean_embeddings) == 0:
            return "Unknown", 0.0, []

        scores = []
        for label, stored_emb in mean_embeddings:
            score = cosine_similarity([new_embedding], [stored_emb])[0][0]
            scores.append((label, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        print("\n[📊 All Similarity Scores]")
        for label, score in scores[:3]:
            print(f"  - {label}: {score:.4f}")

        best_label, best_score = scores[0]

        if best_score >= self.threshold:
            return best_label, best_score, scores[:3]

        return "Unknown", best_score, scores[:3]
