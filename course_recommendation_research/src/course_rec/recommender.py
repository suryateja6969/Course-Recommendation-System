from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfRecommender:
    vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    course_ids: np.ndarray  # aligned to tfidf_matrix rows


def build_tfidf_recommender(df: pd.DataFrame, text_col: str = "course_title", max_features: int = 20000) -> TfidfRecommender:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf = vec.fit_transform(df[text_col].astype(str))
    return TfidfRecommender(vectorizer=vec, tfidf_matrix=tfidf.tocsr(), course_ids=df["course_id"].to_numpy())


def recommend_similar_courses(
    df: pd.DataFrame,
    rec: TfidfRecommender,
    seed_course_id: int,
    top_k: int = 10,
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Item-to-item recommendation: returns top_k courses similar to the seed course title.
    """
    if seed_course_id not in set(rec.course_ids):
        raise ValueError(f"course_id {seed_course_id} not found in TF-IDF index")

    idx = int(np.where(rec.course_ids == seed_course_id)[0][0])
    seed_vec = rec.tfidf_matrix[idx]
    sims = cosine_similarity(seed_vec, rec.tfidf_matrix).ravel()

    # Exclude itself
    sims[idx] = -1

    out = df.copy()
    out["similarity"] = sims

    # Ensure derived features expected by downstream code exist.
    # Some pipelines expect `title_len` (created in make_modeling_frame); compute if missing.
    if "title_len" not in out.columns and "course_title" in out.columns:
        out["title_len"] = out["course_title"].astype(str).str.len()

    if filters:
        for k, v in filters.items():
            if k in out.columns:
                if isinstance(v, (list, tuple, set)):
                    out = out[out[k].isin(list(v))]
                else:
                    out = out[out[k] == v]

    return out.sort_values("similarity", ascending=False).head(top_k)


def recommend_by_query(
    df: pd.DataFrame,
    rec: TfidfRecommender,
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Query-to-item recommendation: uses TF-IDF on course titles.
    """
    qv = rec.vectorizer.transform([query])
    sims = cosine_similarity(qv, rec.tfidf_matrix).ravel()

    out = df.copy()
    out["similarity"] = sims

    # Ensure derived features expected by downstream code exist.
    if "title_len" not in out.columns and "course_title" in out.columns:
        out["title_len"] = out["course_title"].astype(str).str.len()

    if filters:
        for k, v in filters.items():
            if k in out.columns:
                if isinstance(v, (list, tuple, set)):
                    out = out[out[k].isin(list(v))]
                else:
                    out = out[out[k] == v]

    return out.sort_values("similarity", ascending=False).head(top_k)


def hybrid_rerank(
    candidates: pd.DataFrame,
    predicted_profit_col: str = "predicted_profit",
    similarity_col: str = "similarity",
    alpha: float = 0.65,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Hybrid score = alpha * normalized_similarity + (1-alpha) * normalized_predicted_profit.
    """
    out = candidates.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-12:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    out["_sim"] = _minmax(out[similarity_col].astype(float))
    out["_pp"] = _minmax(out[predicted_profit_col].astype(float))

    out["hybrid_score"] = alpha * out["_sim"] + (1 - alpha) * out["_pp"]
    return out.sort_values("hybrid_score", ascending=False).head(top_k).drop(columns=["_sim", "_pp"])
