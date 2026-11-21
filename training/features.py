# File: training/features.py
"""Single source of truth for the lead-scoring feature contract.

Training (``training/train.py``), real-time serving (``serving/``) and the
batch scorer (``batch/score_unscored.py``) all build feature frames through
this module so the schema can never drift between offline and online paths.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Iterable, Mapping, Sequence

import pandas as pd

FEATURE_SCHEMA_VERSION: Final[str] = "v1"
TARGET_COLUMN: Final[str] = "converted"
LEAD_ID_COLUMN: Final[str] = "lead_id"


class FeatureSchemaError(ValueError):
    """Raised when an input frame violates the shared feature contract."""


class FeatureRole(str, Enum):
    """How a feature is pre-processed inside the sklearn pipeline."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class QualityBand(str, Enum):
    """Discretised conversion propensity consumed by the routing engine."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Score thresholds (inclusive lower bounds). Shared by webhook serving and
# batch scoring so a lead is banded identically in both paths.
HIGH_BAND_MIN_SCORE: Final[float] = 0.65
MEDIUM_BAND_MIN_SCORE: Final[float] = 0.35

# Ordered (name -> role) contract. Order is significant: it defines the
# column order of every feature frame handed to the model pipeline.
_FEATURE_CONTRACT: Final[tuple[tuple[str, FeatureRole], ...]] = (
    ("company_size", FeatureRole.CATEGORICAL),
    ("industry", FeatureRole.CATEGORICAL),
    ("source", FeatureRole.CATEGORICAL),
    ("region", FeatureRole.CATEGORICAL),
    ("funding_stage", FeatureRole.CATEGORICAL),
    ("engagement_score", FeatureRole.NUMERIC),
    ("emails_opened", FeatureRole.NUMERIC),
    ("website_visits", FeatureRole.NUMERIC),
)

NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    name for name, role in _FEATURE_CONTRACT if role is FeatureRole.NUMERIC
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = tuple(
    name for name, role in _FEATURE_CONTRACT if role is FeatureRole.CATEGORICAL
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name, _role in _FEATURE_CONTRACT
)


def band_for_score(score: float) -> QualityBand:
    """Map a conversion probability to its routing quality band."""
    if score >= HIGH_BAND_MIN_SCORE:
        return QualityBand.HIGH
    if score >= MEDIUM_BAND_MIN_SCORE:
        return QualityBand.MEDIUM
    return QualityBand.LOW


def build_feature_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build an ordered, validated feature frame from raw record mappings.

    Raises:
        FeatureSchemaError: if any required feature is missing or extra
            columns are present (drift protection).
    """
    frame = pd.DataFrame(list(records))
    if frame.empty:
        frame = pd.DataFrame(columns=list(FEATURE_COLUMNS))
    validate_feature_frame(frame)
    return frame.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Fail fast on schema drift between training and serving paths.

    Raises:
        FeatureSchemaError: on missing or unexpected columns.
    """
    expected = set(FEATURE_COLUMNS)
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    problems: list[str] = []
    if missing:
        problems.append(f"missing feature(s): {missing}")
    if unexpected:
        problems.append(f"unexpected column(s): {unexpected}")
    if problems:
        raise FeatureSchemaError(
            f"feature contract {FEATURE_SCHEMA_VERSION} violated: "
            + "; ".join(problems)
        )


def describe_contract() -> dict[str, Sequence[str]]:
    """Machine-readable description of the contract (used by /health)."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
    }
