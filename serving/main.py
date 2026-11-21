# File: serving/main.py
"""FastAPI application factory.

Lifespan responsibilities:
  * configure structured JSON logging;
  * resolve + load the current ``models:/<name>/<stage>`` model once via
    ``mlflow.pyfunc`` (no per-request registry calls);
  * build the routing engine (Salesforce client, strategies, breakers).

Run locally:
    uvicorn serving.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.config import (  # noqa: E402
    AppSettings,
    SnowflakeSettings,
    get_app_settings,
    get_salesforce_settings,
    get_snowflake_settings,
)
from serving.routes_leads import ReplayCache, router as leads_router  # noqa: E402
from serving.salesforce_client import SalesforceClient  # noqa: E402
from serving.routing import build_default_engine  # noqa: E402
from training.features import describe_contract  # noqa: E402

logger = logging.getLogger("lead_engine.main")


class JsonFormatter(logging.Formatter):
    """Minimal structured-JSON log formatter (stdout-friendly)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))


def resolve_production_version(client: MlflowClient, model_name: str, stage: str) -> str | None:
    """Best-effort resolution of the numeric version currently in a stage."""
    try:
        versions = client.search_model_versions(
            f"name='{model_name}' and current_stage='{stage}'"
        )
        if not versions:
            return None
        return str(max(v.version for v in versions))
    except Exception:  # noqa: BLE001 - health must not crash on registry hiccups
        logger.exception("could not resolve %s version for '%s'", stage, model_name)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: AppSettings = app.state.settings
    configure_logging(settings.log_level)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    logger.info("loading model from %s (tracking_uri=%s)", model_uri, settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = resolve_production_version(client, settings.model_name, settings.model_stage)

    app.state.model = mlflow.pyfunc.load_model(model_uri)
    app.state.model_version = version or "unknown"
    app.state.schema_contract = describe_contract()

    sf_client = SalesforceClient.instance(get_salesforce_settings())
    app.state.router = build_default_engine(
        sf_client=sf_client,
        sf_settings=get_salesforce_settings(),
        snow_settings=get_snowflake_settings(),
        app_settings=settings,
    )
    app.state.replay_cache = ReplayCache(settings.webhook_replay_cache_size)

    logger.info("startup complete: model=%s v%s", settings.model_name, app.state.model_version)
    try:
        yield
    finally:
        logger.info("shutdown: releasing resources")
        SalesforceClient.reset_instance()


def create_app(config: AppSettings | None = None) -> FastAPI:
    """Build the FastAPI app; inject config explicitly for testability."""
    settings = config or get_app_settings()

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        async with lifespan(app):
            yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Automated Lead Scoring & Dynamic Routing Engine",
        lifespan=_lifespan,
    )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, object]:
        model_version = getattr(app.state, "model_version", None)
        contract = getattr(app.state, "schema_contract", None)
        return {
            "status": "ok" if model_version else "degraded",
            "model_version": model_version,
            "model_stage": settings.model_stage,
            "schema_version": contract["schema_version"] if contract else None,
        }

    app.include_router(leads_router)
    app.state.settings = settings
    return app


app = create_app()
