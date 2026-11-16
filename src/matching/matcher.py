import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.matching.preprocess import clean


class HybridMatcher:
    """Buy kataloğunu bir kez indeksler; sorgu başlığı için hibrit skorla top-k döndürür."""

    def __init__(self, model_path: str, catalog_csv: str, abt_csv: str, alpha: float = 0.5):
        self.alpha = alpha
        self.model = SentenceTransformer(model_path)  # cihazı otomatik seçer (GPU varsa GPU)

        buy = pd.read_csv(catalog_csv, encoding="latin-1")
        abt = pd.read_csv(abt_csv, encoding="latin-1")
        self.ids = buy["id"].to_numpy()
        self.names = buy["name"].fillna("").tolist()
        texts = buy["name"].apply(clean).tolist()

        # TF-IDF: eğitimdeki gibi Abt + Buy üzerinde fit edilir
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.vec.fit(pd.concat([abt["name"].apply(clean), buy["name"].apply(clean)]))
        self.tfidf_matrix = self.vec.transform(texts)  # (n_buy, n_features), L2 normalize

        self.emb_matrix = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True
        )  # (n_buy, 384)

    def match(self, title: str, top_k: int = 5):
        q = clean(title)
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        q_tfidf = self.vec.transform([q])

        sim_emb = self.emb_matrix @ q_emb
        sim_tfidf = (self.tfidf_matrix @ q_tfidf.T).toarray().ravel()
        score = self.alpha * sim_emb + (1 - self.alpha) * sim_tfidf

        top = np.argsort(-score)[:top_k]
        return [
            {
                "id": int(self.ids[i]),
                "name": self.names[i],
                "score": float(score[i]),
                "embedding_score": float(sim_emb[i]),
                "tfidf_score": float(sim_tfidf[i]),
            }
            for i in top
        ]