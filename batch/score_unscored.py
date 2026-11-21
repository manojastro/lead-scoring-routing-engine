# File: batch/score_unscored.py
"""Backfill scoring for un-scored leads living in Snowflake.

Loop (resumable by construction):
    1. SELECT the next chunk of leads WHERE SCORE IS NULL (oldest first);
    2. build feature frames through ``training.features`` (training parity);
    3. batch-predict probabilities via the MLflow registry pyfunc model;
    4. UPDATE score/band/model_version/scored_at inside a per-chunk
       transaction - committed chunks are durable, so an interrupted run
       simply resumes where the last commit landed.
    5. Repeat until fewer than ``--chunk-size`` rows remain or the
       ``--limit`` budget is exhausted.

Usage:
    python -m batch.score_unscored --limit 5000 --chunk-size 500
    python -m batch.score_unscored --dry-run --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.config import AppSettings, SnowflakeSettings, get_app_settings, get_snowflake_settings  # noqa: E402
from training.features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    QualityBand,
    band_for_score,
    build_feature_frame,
)

logger = logging.getLogger("lead_engine.batch")

SELECT_COLUMNS: tuple[str, ...] = (
    "LEAD_ID",
    "EMAIL",
    "COMPANY_NAME",
    "COMPANY_SIZE",
    "INDUSTRY",
    "SOURCE",
    "REGION",
    "FUNDING_STAGE",
    "ENGAGEMENT_SCORE",
    "EMAILS_OPENED",
    "WEBSITE_VISITS",
)

UPDATE_SQL_TEMPLATE: str = (
    "UPDATE {database}.{schema}.{table} "
    "SET SCORE = %s, "
    "    QUALITY_BAND = %s, "
    "    MODEL_VERSION = %s, "
    "    SCORED_AT = CURRENT_TIMESTAMP(), "
    "    FEATURE_SCHEMA_VERSION = %s "
    "WHERE LEAD_ID = %s AND SCORE IS NULL"
)


@dataclass(frozen=True, slots=True)
class BatchStats:
    chunks_committed: int = 0
    rows_scored: int = 0
    high_band: int = 0
    medium_band: int = 0
    low_band: int = 0
    elapsed_s: float = 0.0


def select_chunk_sql(settings: SnowflakeSettings, limit: int) -> str:
    columns = ", ".join(SELECT_COLUMNS)
    return (
        f"SELECT {columns} FROM {settings.database}.{settings.schema_name}.LEADS "
        f"WHERE SCORE IS NULL ORDER BY CREATED_AT LIMIT {int(limit)}"
    )


def fetch_chunk(conn: Any, settings: SnowflakeSettings, limit: int) -> list[dict[str, Any]]:
    """Pull the next resumable chunk as plain dicts keyed by lowercase names."""
    with conn.cursor() as cursor:
        cursor.execute(select_chunk_sql(settings, limit))
        raw_rows = cursor.fetchall()
        column_names = [c[0] for c in cursor.description]
    keys = [name.lower() for name in column_names]
    return [dict(zip(keys, row)) for row in raw_rows]


def update_chunk(
    conn: Any,
    settings: SnowflakeSettings,
    lead_ids: list[str],
    scores: np.ndarray,
    bands: list[QualityBand],
    model_version: str,
) -> int:
    """Write scores for one chunk inside a single transaction."""
    sql = UPDATE_SQL_TEMPLATE.format(
        database=settings.database,
        schema=settings.schema_name,
        table="LEADS",
    )
    params = [
        (float(score), band.value, model_version, FEATURE_SCHEMA_VERSION, lead_id)
        for lead_id, score, band in zip(lead_ids, scores, bands, strict=True)
    ]
    try:
        conn.autocommit(False)
        with conn.cursor() as cursor:
            cursor.executemany(sql, params)
        conn.commit()
        return len(params)
    except Exception:
        conn.rollback()
        raise


def run_batch(
    conn_factory: Any,
    model: Any,
    settings: SnowflakeSettings,
    *,
    limit: int,
    chunk_size: int,
    model_version: str,
    dry_run: bool = False,
) -> BatchStats:
    """Execute the full backfill loop and return summary statistics."""
    if chunk_size <= 0 or limit <= 0:
        raise ValueError("limit and chunk_size must be positive")

    scored_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    chunks_committed = 0
    rows_total = 0
    started = time.perf_counter()
    remaining = limit

    while remaining > 0:
        conn = conn_factory()
        try:
            rows = fetch_chunk(conn, settings, min(chunk_size, remaining))
        except Exception:
            conn.close()
            raise

        if not rows:
            logger.info("no further un-scored leads found; stopping")
            conn.close()
            break

        feature_frame = build_feature_frame(rows)  # contract-checked parity
        probabilities = np.asarray(model.predict(feature_frame), dtype=float).ravel()
        if len(probabilities) != len(rows):
            raise RuntimeError(
                f"model returned {len(probabilities)} scores for {len(rows)} rows"
            )
        bands = [band_for_score(float(p)) for p in probabilities]

        if not dry_run:
            try:
                update_chunk(
                    conn=conn,
                    settings=settings,
                    lead_ids=[str(r["lead_id"]) for r in rows],
                    scores=probabilities,
                    bands=bands,
                    model_version=model_version,
                )
            finally:
                conn.close()
            chunks_committed += 1
        else:
            conn.close()

        for band in bands:
            scored_counts[band.value] += 1
        rows_total += len(rows)
        remaining -= len(rows)
        logger.info(
            "%s chunk of %d rows (total=%d, remaining_budget=%d)",
            "[dry-run] would write" if dry_run else "committed", len(rows), rows_total, max(remaining, 0),
        )

    stats = BatchStats(
        chunks_committed=chunks_committed,
        rows_scored=rows_total,
        high_band=scored_counts["HIGH"],
        medium_band=scored_counts["MEDIUM"],
        low_band=scored_counts["LOW"],
        elapsed_s=round(time.perf_counter() - started, 3),
    )
    logger.info("batch summary: %r", stats)
    return stats


def resolve_model(app_settings: AppSettings, model_uri_override: str | None) -> tuple[Any, str]:
    uri = model_uri_override or f"models:/{app_settings.model_name}/{app_settings.model_stage}"
    logger.info("loading registry model: %s", uri)
    model = mlflow.pyfunc.load_model(uri)

    version = "unknown"
    try:
        client = MlflowClient(tracking_uri=app_settings.mlflow_tracking_uri)
        name, _, ver = uri.removeprefix("models:/").partition("/")
        versions = client.search_model_versions(f"name='{name}'")
        staged = [
            v.version for v in versions
            if getattr(v, "current_stage", "") == app_settings.model_stage
        ]
        version = str(max(staged)) if staged else (ver or "unknown")
    except Exception:  # noqa: BLE001 - version metadata is best-effort
        logger.exception("could not resolve model version; continuing")
    return model, version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000, help="Max leads to process this run.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Rows per fetch/update transaction.")
    parser.add_argument("--model-uri", default=None, help="Override registry URI (default models:/<name>/<stage>).")
    parser.add_argument("--dry-run", action="store_true", help="Score + log only; skip UPDATE.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app_settings = get_app_settings()
    snow_settings = get_snowflake_settings()
    mlflow.set_tracking_uri(app_settings.mlflow_tracking_uri)

    try:
        import snowflake.connector

        model, model_version = resolve_model(app_settings, args.model_uri)

        def conn_factory() -> Any:
            return snowflake.connector.connect(**snow_settings.connection_params())

        stats = run_batch(
            conn_factory=conn_factory,
            model=model,
            settings=snow_settings,
            limit=args.limit,
            chunk_size=args.chunk_size,
            model_version=model_version,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        logger.error("invalid arguments: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 - top-level operator-facing failure
        logger.exception("batch scoring failed")
        return 1

    mode = "DRY-RUN" if args.dry_run else "COMMITTED"
    logger.info(
        "[%s] rows=%d chunks=%d HIGH=%d MEDIUM=%d LOW=%d elapsed=%.1fs",
        mode, stats.rows_scored, stats.chunks_committed,
        stats.high_band, stats.medium_band, stats.low_band, stats.elapsed_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
