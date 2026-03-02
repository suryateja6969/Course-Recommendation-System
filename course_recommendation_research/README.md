# Explainable Course Recommendation System (Research Repo)

**Goal:** Build a reproducible research workflow on the Udemy course dataset:
- Train **multiple models** (baselines + ML models)
- Provide **recommendations** (content-based + hybrid re-ranking)
- Add **explainability** using **SHAP**
- Ship a lightweight **Streamlit prototype**
- Keep **all experimentation in Jupyter notebooks** (export artifacts for the app)

## Quickstart (uv)

```bash
uv sync
uv run python -m pip install --upgrade pip  # optional
```

Run notebooks in `notebooks/` (recommended order below).

Run Streamlit app:

```bash
uv run streamlit run app/app.py
```

## Notebook order

1. `00_data_audit_and_cleaning.ipynb`
2. `01_eda.ipynb`
3. `02_feature_engineering.ipynb`
4. `03_train_multiple_models.ipynb`
5. `04_recommender_models.ipynb`
6. `05_shap_explainability.ipynb`
7. `06_export_artifacts_for_streamlit.ipynb`

## Outputs / artifacts

- Clean dataset: `data/processed/courses_clean.parquet`
- Trained quality model: `models/profit_model.joblib`
- TF-IDF vectorizer + matrix: `models/tfidf_vectorizer.joblib`, `models/tfidf_matrix.npz`
- Final course index: `models/course_index.parquet` (id/title/url + key fields)

## Research paper

A detailed paper outline + reporting checklist is included in `reports/paper_outline.md`.
