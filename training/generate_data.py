# File: training/generate_data.py
"""Generate a seeded, realistic mock B2B lead dataset for model training.

Signal design (encoded as a latent logistic score):
  * intent sources (`demo_request`, `referral`) convert far better than
    purchased lists;
  * engagement metrics (emails opened, website visits, engagement score)
    carry strong positive signal;
  * enterprise company sizes and later funding stages help;
  * industry/region add mild lift or drag;
  * label noise is injected via a probabilistic floor so the task is not
    trivially separable.

Usage:
    python -m training.generate_data --rows 20000 --out data/leads.csv --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger("lead_engine.generate_data")

RANDOM_SEED: Final[int] = 42
DEFAULT_ROWS: Final[int] = 20_000
DEFAULT_OUT: Final[Path] = Path("data") / "leads.csv"

INDUSTRIES: Final[tuple[str, ...]] = (
    "saas",
    "fintech",
    "healthcare",
    "ecommerce",
    "manufacturing",
    "media",
    "logistics",
)
SOURCES: Final[tuple[str, ...]] = (
    "demo_request",
    "referral",
    "inbound_web",
    "event",
    "partner",
    "purchased_list",
)
REGIONS: Final[tuple[str, ...]] = (
    "us_west",
    "us_east",
    "emea",
    "apac",
    "latam",
)
COMPANY_SIZES: Final[tuple[str, ...]] = (
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1000+",
)
FUNDING_STAGES: Final[tuple[str, ...]] = (
    "bootstrap",
    "seed",
    "series_a",
    "series_b",
    "series_c_plus",
    "unknown",
)

# Multiplicative log-odds contributions, tuned so base conversion lands
# around 18-22% while keeping per-feature effects interpretable.
SOURCE_LOG_ODDS: Final[dict[str, float]] = {
    "demo_request": 1.30,
    "referral": 1.05,
    "inbound_web": 0.35,
    "event": 0.20,
    "partner": 0.10,
    "purchased_list": -1.10,
}
INDUSTRY_LOG_ODDS: Final[dict[str, float]] = {
    "saas": 0.35,
    "fintech": 0.25,
    "healthcare": 0.00,
    "ecommerce": -0.10,
    "logistics": -0.20,
    "manufacturing": -0.30,
    "media": -0.45,
}
SIZE_LOG_ODDS: Final[dict[str, float]] = {
    "1-10": -0.55,
    "11-50": -0.25,
    "51-200": 0.00,
    "201-500": 0.30,
    "501-1000": 0.45,
    "1000+": 0.60,
}
FUNDING_LOG_ODDS: Final[dict[str, float]] = {
    "bootstrap": -0.45,
    "seed": -0.20,
    "series_a": 0.05,
    "series_b": 0.30,
    "series_c_plus": 0.55,
    "unknown": -0.15,
}
REGION_LOG_ODDS: Final[dict[str, float]] = {
    "us_west": 0.15,
    "us_east": 0.10,
    "emea": 0.00,
    "apac": -0.10,
    "latam": -0.20,
}

BASE_INTERCEPT: Final[float] = -2.05
NOISE_SIGMA: Final[float] = 0.85


def _sample_categoricals(
    rng: np.random.Generator, rows: int
) -> dict[str, np.ndarray]:
    """Sample correlated categoricals; bigger companies skew enterprise."""
    size_idx = rng.choice(len(COMPANY_SIZES), size=rows, p=(0.22, 0.26, 0.20, 0.16, 0.09, 0.07))
    company_size = np.array(COMPANY_SIZES, dtype=object)[size_idx]
    is_enterprise = size_idx >= 3

    industry_idx = rng.choice(len(INDUSTRIES), size=rows)
    # Enterprise leads skew toward saas/fintech (first two entries).
    enterprise_mask = rng.random(rows) < 0.5
    industry_idx = np.where(is_enterprise & enterprise_mask, rng.integers(0, 2, rows), industry_idx)
    industry = np.array(INDUSTRIES, dtype=object)[industry_idx]

    source = rng.choice(SOURCES, size=rows, p=(0.14, 0.12, 0.34, 0.14, 0.11, 0.15))
    region = rng.choice(REGIONS, size=rows, p=(0.28, 0.24, 0.26, 0.13, 0.09))
    funding = rng.choice(FUNDING_STAGES, size=rows, p=(0.14, 0.20, 0.26, 0.18, 0.10, 0.12))

    return {
        "company_size": company_size,
        "industry": industry,
        "source": source,
        "region": region,
        "funding_stage": funding,
    }


def generate_leads(rows: int, seed: int) -> pd.DataFrame:
    """Return a typed DataFrame of synthetic B2B leads with labels."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")

    rng = np.random.default_rng(seed)

    cats = _sample_categoricals(rng, rows)
    engagement_score = np.clip(rng.normal(loc=48.0, scale=21.0, size=rows), 0.0, 100.0)
    source_lift = np.vectorize(SOURCE_LOG_ODDS.get)(cats["source"]).astype(float)
    emails_opened = rng.poisson(lam=np.clip(2.0 + source_lift * 2.4 + engagement_score / 25.0, 0.4, 40.0))
    website_visits = rng.poisson(lam=np.clip(3.0 + source_lift * 3.0 + engagement_score / 18.0, 0.5, 90.0))

    frame = pd.DataFrame(
        {
            "lead_id": [f"LEAD-{seed}-{i:06d}" for i in range(rows)],
            "company_size": cats["company_size"],
            "industry": cats["industry"],
            "source": cats["source"],
            "region": cats["region"],
            "funding_stage": cats["funding_stage"],
            "engagement_score": np.round(engagement_score, 2),
            "emails_opened": emails_opened.astype(np.int32),
            "website_visits": website_visits.astype(np.int32),
        }
    )

    logits = pd.Series(BASE_INTERCEPT, index=frame.index)
    logits += frame["source"].map(SOURCE_LOG_ODDS).astype(float)
    logits += frame["industry"].map(INDUSTRY_LOG_ODDS).astype(float)
    logits += frame["company_size"].map(SIZE_LOG_ODDS).astype(float)
    logits += frame["funding_stage"].map(FUNDING_LOG_ODDS).astype(float)
    logits += frame["region"].map(REGION_LOG_ODDS).astype(float)
    logits += 0.032 * (frame["engagement_score"] - 48.0)
    logits += 0.085 * frame["emails_opened"]
    logits += 0.048 * np.log1p(frame["website_visits"]) * 6.0
    logits += rng.normal(loc=0.0, scale=NOISE_SIGMA, size=rows)

    probs = 1.0 / (1.0 + np.exp(-logits))
    frame["converted"] = rng.binomial(n=1, p=probs.to_numpy()).astype(np.int8)

    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of leads to generate.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="PRNG seed for reproducibility.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        frame = generate_leads(rows=args.rows, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
    except (ValueError, OSError) as exc:
        logger.error("Generation failed: %s", exc)
        return 1

    rate = float(frame["converted"].mean())
    logger.info(
        "Wrote %d leads to %s (positive rate=%.3f, columns=%s)",
        len(frame), args.out, rate, list(frame.columns),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
