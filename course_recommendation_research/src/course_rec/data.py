from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


DURATION_RE = re.compile(r"^\s*(?P<hours>[0-9]*\.?[0-9]+)\s*hours?\s*$", re.IGNORECASE)


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load the raw Udemy course dataset (CSV)."""
    return pd.read_csv(path)


def parse_content_duration_to_hours(s: str) -> Optional[float]:
    """
    Parse strings like '1.5 hours' -> 1.5 (float).
    Returns None if parsing fails.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = DURATION_RE.match(str(s))
    if not m:
        return None
    return float(m.group("hours"))


def clean_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal but robust cleaning:
    - drop exact duplicates by course_id
    - coerce/clean categorical noise in 'level'
    - parse duration and timestamps
    """
    out = df.copy()

    # Drop duplicate course_id rows (identical records in this dataset)
    out = out.drop_duplicates(subset=["course_id"]).reset_index(drop=True)

    # Fix rare noise category in level (e.g., '52')
    out["level"] = out["level"].astype(str).str.strip()
    out.loc[~out["level"].isin(["All Levels", "Beginner Level", "Intermediate Level", "Expert Level"]), "level"] = "Unknown"

    # Duration in hours (float)
    out["content_duration_hours"] = out["content_duration"].apply(parse_content_duration_to_hours)

    # Timestamps
    out["published_timestamp"] = pd.to_datetime(out["published_timestamp"], errors="coerce", utc=True)
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")

    # Handle missing published_time (rare)
    if "published_time" in out.columns:
        out["published_time"] = out["published_time"].fillna("00:00:00")

    # Guard against negative values
    for col in ["price", "num_subscribers", "num_reviews", "num_lectures", "profit"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)

    return out


def make_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select + derive fields used for modeling/recommendation.
    (You can expand this as part of your feature engineering notebook.)
    """
    out = df.copy()
    out["title_len"] = out["course_title"].astype(str).str.len()
    out["is_free"] = (~out["is_paid"]).astype(int)
    out["log_price"] = (out["price"] + 1).apply(lambda x: float(__import__('numpy').log(x)))  # backward compat for notebooks
    return out
