from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# Optional (app still works without SHAP)
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


def _project_root() -> Path:
    """Find the repo root that contains the `models/` folder."""
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, *here.parents]:
        if (p / "models").exists():
            return p
    # Fallback: use parent; a clear error is raised later if artifacts are missing.
    return here.parent


BASE_DIR = _project_root()
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_artifacts() -> Tuple[pd.DataFrame, object, sparse.spmatrix, object, list[str], list[str], str]:
    """Load artifacts produced by the notebooks."""
    required = [
        MODELS_DIR / "course_index.parquet",
        MODELS_DIR / "tfidf_vectorizer.joblib",
        MODELS_DIR / "tfidf_matrix.npz",
        MODELS_DIR / "profit_model.joblib",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Missing required artifact(s). Run notebooks 03/04/06 first:\n" + "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(msg)

    index_df = pd.read_parquet(MODELS_DIR / "course_index.parquet")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_matrix = sparse.load_npz(MODELS_DIR / "tfidf_matrix.npz")

    bundle = joblib.load(MODELS_DIR / "profit_model.joblib")
    pipe = bundle["pipeline"]
    num_cols = list(bundle["numeric_cols"])
    cat_cols = list(bundle["categorical_cols"])
    model_name = str(bundle.get("model_name", "model"))

    # Defensive: columns used by UI.
    if "url" not in index_df.columns:
        index_df["url"] = ""

    # Normalize a few common dtypes.
    if "price" in index_df.columns:
        index_df["price"] = pd.to_numeric(index_df["price"], errors="coerce").fillna(0).astype(int)
    if "is_paid" in index_df.columns:
        # Some exports store as strings/ints.
        if index_df["is_paid"].dtype != bool:
            index_df["is_paid"] = index_df["is_paid"].astype(str).str.lower().isin(["true", "1", "paid", "yes"])

    return index_df, vectorizer, tfidf_matrix, pipe, num_cols, cat_cols, model_name


def _minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)


def _ensure_title_len(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    if "title_len" in required_cols and "title_len" not in df.columns and "course_title" in df.columns:
        df = df.copy()
        df["title_len"] = df["course_title"].astype(str).str.len()
    return df


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _get_hours(df: pd.DataFrame) -> pd.Series:
    # Depending on notebook version, you may have either of these.
    if "content_duration_hours" in df.columns:
        return _safe_num(df, "content_duration_hours")
    if "content_duration" in df.columns:
        return _safe_num(df, "content_duration")
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def student_value_components(cand: pd.DataFrame) -> pd.DataFrame:
    """Compute a student-facing score using only dataset proxies (no extra training)."""
    out = cand.copy()

    subs = _safe_num(out, "num_subscribers")
    revs = _safe_num(out, "num_reviews")
    price = _safe_num(out, "price")
    hours = _get_hours(out)
    lectures = _safe_num(out, "num_lectures")

    out["hours"] = hours
    out["lectures"] = lectures

    # Social proof / quality proxies
    out["quality_reviews"] = np.log1p(revs)
    out["quality_popularity"] = np.log1p(subs)
    out["review_rate"] = (revs / (subs + 1.0)) * 1000.0  # reviews per 1k subscribers

    # Value-for-money proxy
    denom = price.clip(lower=1.0)
    out["hours_per_dollar"] = np.where(price <= 0, hours, hours / denom)

    # Normalize components to [0,1]
    q1 = _minmax(out["quality_reviews"].to_numpy())
    q2 = _minmax(out["quality_popularity"].to_numpy())
    q3 = _minmax(out["review_rate"].to_numpy())
    v1 = _minmax(out["hours"].to_numpy())
    v2 = _minmax(out["hours_per_dollar"].to_numpy())

    out["student_quality"] = 0.45 * q1 + 0.35 * q2 + 0.20 * q3
    out["student_value"] = 0.55 * v2 + 0.45 * v1
    out["student_score"] = 0.65 * out["student_quality"] + 0.35 * out["student_value"]

    return out


def why_student_text(row: pd.Series) -> str:
    reasons: list[str] = []

    if float(row.get("similarity", 0.0)) >= 0.35:
        reasons.append("High match to your query")

    if float(row.get("num_reviews", 0.0)) >= 200:
        reasons.append("Strong social proof (many reviews)")

    if float(row.get("num_subscribers", 0.0)) >= 10_000:
        reasons.append("Popular with learners (many subscribers)")

    if float(row.get("hours", 0.0)) >= 10.0:
        reasons.append("Substantial content (10+ hours)")

    price = float(row.get("price", 0.0))
    hpd = float(row.get("hours_per_dollar", 0.0))
    if price == 0:
        reasons.append("Free course (low risk to try)")
    elif hpd >= 0.20:
        reasons.append("Good value-for-money (hours per dollar)")

    if not reasons:
        reasons.append("Balanced relevance + learning value")

    return " • ".join(reasons[:3])


def recommend(
    index_df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    pipe,
    num_cols,
    cat_cols,
    query: str,
    top_k: int,
    subject: Optional[str],
    level: Optional[str],
    is_paid: Optional[bool],
    max_price: Optional[int],
    alpha: float,
    profit_weight: float,
) -> pd.DataFrame:
    """Hybrid ranking: (query relevance + student value) plus optional profit weight."""

    # Similarity
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix).ravel()

    cand = index_df.copy()
    cand["similarity"] = sims

    # Filters
    if subject and subject != "Any" and "subject" in cand.columns:
        cand = cand[cand["subject"] == subject]
    if level and level != "Any" and "level" in cand.columns:
        cand = cand[cand["level"] == level]
    if is_paid is not None and "is_paid" in cand.columns:
        cand = cand[cand["is_paid"] == is_paid]
    if max_price is not None and "price" in cand.columns:
        cand = cand[cand["price"] <= max_price]

    if cand.empty:
        return cand

    # Profit prediction (platform utility)
    required = list(num_cols) + list(cat_cols)
    cand = _ensure_title_len(cand, required)
    X = cand[num_cols + cat_cols]
    pred_log = pipe.predict(X)  # trained on log1p(profit)
    cand["predicted_profit"] = np.expm1(pred_log)

    # Student-facing score and explanation text
    cand = student_value_components(cand)
    cand["why_student"] = cand.apply(why_student_text, axis=1)

    # Ranking
    sim_n = _minmax(cand["similarity"].to_numpy())
    stud_n = _minmax(cand["student_score"].to_numpy())
    profit_n = _minmax(cand["predicted_profit"].to_numpy())

    student_rank = alpha * sim_n + (1.0 - alpha) * stud_n
    profit_weight = float(np.clip(profit_weight, 0.0, 0.5))
    cand["score"] = (1.0 - profit_weight) * student_rank + profit_weight * profit_n

    out_cols = [
        "course_id",
        "course_title",
        "subject",
        "level",
        "is_paid",
        "price",
        "similarity",
        "student_score",
        "predicted_profit",
        "score",
        "why_student",
        "url",
    ]

    # Keep only columns that exist.
    out_cols = [c for c in out_cols if c in cand.columns]

    return cand.sort_values("score", ascending=False).head(top_k)[out_cols]


@st.cache_resource
def shap_background(index_df: pd.DataFrame, _pipe, num_cols, cat_cols, n: int = 200):
    """Representative SHAP background. `_pipe` avoids Streamlit hashing errors."""
    if shap is None:
        return None

    pre = _pipe.named_steps["pre"]

    bg_df = index_df.sample(n=min(n, len(index_df)), random_state=42).copy()
    bg_df = _ensure_title_len(bg_df, list(num_cols) + list(cat_cols))
    X_bg = bg_df[num_cols + cat_cols]
    bg = pre.transform(X_bg)

    try:
        if sparse.issparse(bg):
            bg = bg.toarray()
    except Exception:
        pass

    try:
        feature_names = np.array(pre.get_feature_names_out())
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(bg.shape[1])])

    return bg, feature_names


def shap_explain_one(pipe, num_cols, cat_cols, row: pd.DataFrame, model_name: str, bg, feature_names):
    """Return (feature_names, shap_values_vector) for the selected row."""
    if shap is None or bg is None:
        return None

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    row = _ensure_title_len(row, list(num_cols) + list(cat_cols))
    X = row[num_cols + cat_cols]
    Xt = pre.transform(X)

    try:
        if sparse.issparse(Xt):
            Xt = Xt.toarray()
    except Exception:
        pass

    name = str(model_name).lower()

    try:
        if "ridge" in name or "linear" in name:
            explainer = shap.LinearExplainer(model, bg, feature_names=feature_names)
            sv = explainer(Xt)
            vals = sv.values[0] if hasattr(sv, "values") else sv[0]
            return feature_names, np.asarray(vals, dtype=float)

        # Tree models
        explainer = shap.TreeExplainer(model, data=bg, feature_names=feature_names)
        sv = explainer(Xt)
        vals = sv.values[0] if hasattr(sv, "values") else sv[0]
        return feature_names, np.asarray(vals, dtype=float)

    except Exception:
        try:
            explainer = shap.Explainer(model, bg, feature_names=feature_names)
            sv = explainer(Xt)
            vals = sv.values[0] if hasattr(sv, "values") else sv[0]
            return feature_names, np.asarray(vals, dtype=float)
        except Exception:
            return None


def _render_recs_table(df: pd.DataFrame):
    disp = df.copy()

    for col in ["similarity", "student_score", "predicted_profit", "score"]:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors="coerce").fillna(0.0).round(4)

    # Prefer native link column if available
    try:
        if "url" in disp.columns:
            st.dataframe(
                disp,
                use_container_width=True,
                column_config={"url": st.column_config.LinkColumn("url", display_text="link")},
            )
        else:
            st.dataframe(disp, use_container_width=True)
    except Exception:
        # Fallback for older Streamlit: markdown links
        if "url" in disp.columns:
            disp["url"] = disp["url"].apply(lambda u: f"[link]({u})" if isinstance(u, str) and u else "")
        st.dataframe(disp, use_container_width=True)


def main():
    st.set_page_config(page_title="Course Recommender (Student + Explainable)", layout="wide")
    st.title("Course Recommendation System")

    try:
        index_df, vectorizer, tfidf_matrix, pipe, num_cols, cat_cols, model_name = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Query")
        query = st.text_input("What do you want to learn?", value="javascript").strip()
        top_k = st.slider("Top-K recommendations", 5, 30, 10)

        st.header("Filters")
        subject_opts = ["Any"] + (sorted(index_df["subject"].dropna().unique().tolist()) if "subject" in index_df.columns else [])
        level_opts = ["Any"] + (sorted(index_df["level"].dropna().unique().tolist()) if "level" in index_df.columns else [])

        subject = st.selectbox("Subject", subject_opts, index=0)
        level = st.selectbox("Level", level_opts, index=0)

        paid_choice = st.radio("Paid / Free", ["Any", "Paid", "Free"], horizontal=True)
        is_paid = None if paid_choice == "Any" else (paid_choice == "Paid")

        max_p = int(index_df["price"].max()) if "price" in index_df.columns else 500
        max_price = st.slider("Max price", 0, max_p, min(200, max_p))

        st.header("Ranking")
        alpha = st.slider("α: relevance vs student value", 0.0, 1.0, 0.65, 0.05)
        profit_weight = st.slider("Profit influence (optional)", 0.0, 0.5, 0.15, 0.05)

        st.caption(f"Profit model: **{model_name}**")

    if not query:
        st.info("Enter a query to get recommendations.")
        st.stop()

    recs = recommend(
        index_df=index_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        pipe=pipe,
        num_cols=num_cols,
        cat_cols=cat_cols,
        query=query,
        top_k=top_k,
        subject=subject,
        level=level,
        is_paid=is_paid,
        max_price=max_price,
        alpha=alpha,
        profit_weight=profit_weight,
    )

    if recs.empty:
        st.warning("No results match your filters. Try relaxing filters (e.g., Level → Any).")
        st.stop()

    # Quick note if similarity is zero across results (usually due to restrictive filters)
    if "similarity" in recs.columns and float(recs["similarity"].max()) <= 1e-12:
        st.warning("All similarities are 0. This usually means the query doesn't match any titles under the current filters.")

    st.subheader("Recommendations")
    with st.expander("How ranking works (student-first)"):
        st.markdown(
            """
- **Relevance**: TF-IDF cosine similarity between your query and course title.
- **Student value score** (proxy): combines social proof (**reviews/subscribers**), engagement (**review rate**), and learning value (**hours**, **hours-per-dollar**).
- **Profit influence**: optional (defaults low). Set it to **0** if you want purely student-focused ranking.
            """
        )

    _render_recs_table(recs)

    # Explanation panel
    st.subheader("Explain a recommendation")

    chosen = st.selectbox(
        "Pick a course",
        options=recs["course_id"].tolist() if "course_id" in recs.columns else list(range(len(recs))),
        format_func=lambda cid: recs.loc[recs["course_id"] == cid, "course_title"].iloc[0]
        if "course_id" in recs.columns and "course_title" in recs.columns else str(cid),
    )

    row = index_df[index_df["course_id"] == chosen].head(1) if "course_id" in index_df.columns else index_df.head(1)
    if row.empty:
        st.warning("Could not find the selected course row.")
        st.stop()

    # Recompute student components for this row (for display)
    row_plus = student_value_components(row.assign(similarity=0.0))

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Why this is good for a student")

        # Match the selected row to the displayed recs row (for similarity + why text)
        sel = recs[recs["course_id"] == chosen].head(1) if "course_id" in recs.columns else recs.head(1)
        why = sel["why_student"].iloc[0] if "why_student" in sel.columns else ""
        if why:
            st.write(why)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        if "similarity" in sel.columns:
            m1.metric("Relevance", f"{float(sel['similarity'].iloc[0]):.3f}")
        if "student_score" in sel.columns:
            m2.metric("Student score", f"{float(sel['student_score'].iloc[0]):.3f}")
        if "predicted_profit" in sel.columns:
            m3.metric("Pred. profit", f"{float(sel['predicted_profit'].iloc[0]):.0f}")
        if "price" in sel.columns:
            m4.metric("Price", f"{int(sel['price'].iloc[0])}")

        # Breakdown table (student proxies)
        breakdown_cols = [
            c for c in [
                "num_subscribers",
                "num_reviews",
                "hours",
                "num_lectures",
                "review_rate",
                "hours_per_dollar",
                "student_quality",
                "student_value",
                "student_score",
            ] if c in row_plus.columns
        ]

        if breakdown_cols:
            bd = row_plus[breakdown_cols].copy()
            for c in bd.columns:
                bd[c] = pd.to_numeric(bd[c], errors="coerce")
            st.dataframe(bd.round(3), use_container_width=True)

    with right:
        st.markdown("### Why the model predicts high success (profit) — SHAP")
        if shap is None:
            st.info("Install `shap` to enable this explanation.")
        else:
            bg_pack = shap_background(index_df, pipe, num_cols, cat_cols, n=200)
            if bg_pack is None:
                st.warning("Could not build SHAP background.")
            else:
                bg, feature_names = bg_pack
                explained = shap_explain_one(pipe, num_cols, cat_cols, row, model_name=model_name, bg=bg, feature_names=feature_names)
                if explained is None:
                    st.warning("Could not compute SHAP values for this model/row.")
                else:
                    fn, vals = explained
                    # Top contributors by absolute magnitude
                    top_n = 12
                    idx = np.argsort(np.abs(vals))[::-1][:top_n]
                    plot_df = pd.DataFrame({
                        "feature": np.array(fn)[idx],
                        "shap_value": np.array(vals)[idx],
                        "abs_shap": np.abs(np.array(vals)[idx]),
                    }).sort_values("abs_shap", ascending=True)

                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(plot_df["feature"], plot_df["shap_value"])
                    plt.title("Top SHAP contributions (profit model)")
                    plt.xlabel("SHAP value (impact on log1p(profit))")
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()