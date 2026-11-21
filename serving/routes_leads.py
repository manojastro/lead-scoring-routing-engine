# File: serving/routes_leads.py
"""POST /webhooks/lead - HMAC-verified, validated, scored, routed.

Security model:
  * caller sends ``X-Lead-Signature: sha256=<hex hmac-sha256(raw_body)>``;
  * signature is verified with ``hmac.compare_digest`` (constant time);
  * body is then parsed into strict Pydantic v2 ``LeadIn`` (extra=forbid);
  * ``lead_id`` acts as an idempotency key: replays within the cache window
    return the original decision without re-scoring or double-routing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Final

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from serving.config import AppSettings
from serving.routing import RoutingEngine, RoutingFailure, ScoredLead
from serving.schemas import LeadIn, WebhookAck
from training.features import FEATURE_COLUMNS, band_for_score, build_feature_frame

logger = logging.getLogger("lead_engine.routes")

SIGNATURE_HEADER: Final[str] = "X-Lead-Signature"
SIGNATURE_PREFIX: Final[str] = "sha256="


def verify_signature(raw_body: bytes, header_value: str | None, secret: bytes) -> bool:
    """Constant-time HMAC-SHA256 verification of the raw request body."""
    if not header_value or not header_value.startswith(SIGNATURE_PREFIX):
        return False
    provided = header_value[len(SIGNATURE_PREFIX):].strip().lower()
    if len(provided) != 64:  # sha256 hex digest length
        return False
    expected = hmac.new(secret, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided)


class ReplayCache:
    """Bounded LRU of lead_id -> ack payload for webhook idempotency."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, lead_id: str) -> dict[str, Any] | None:
        ack = self._store.get(lead_id)
        if ack is not None:
            self._store.move_to_end(lead_id)
        return ack

    def put(self, lead_id: str, ack: dict[str, Any]) -> None:
        self._store[lead_id] = ack
        self._store.move_to_end(lead_id)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


router = APIRouter(tags=["leads"])


def _positive_probability(pyfunc_model: Any, feature_frame: Any) -> float:
    """Extract P(converted) from an mlflow pyfunc predict output."""
    output = pyfunc_model.predict(feature_frame)
    array = np.asarray(output, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("model returned empty prediction vector")
    return float(array[0])


@router.post(
    "/webhooks/lead",
    response_model=WebhookAck,
    status_code=status.HTTP_200_OK,
    summary="Score an inbound lead and route it to Salesforce or nurture.",
)
async def receive_lead(request: Request) -> WebhookAck:
    started = time.perf_counter()
    settings: AppSettings = request.app.state.settings
    engine: RoutingEngine = request.app.state.router
    replay_cache: ReplayCache = request.app.state.replay_cache

    raw_body = await request.body()

    # 1. Authenticate the producer.
    if not verify_signature(
        raw_body,
        request.headers.get(SIGNATURE_HEADER),
        settings.lead_webhook_secret.get_secret_value().encode("utf-8"),
    ):
        logger.warning("rejected webhook: invalid or missing HMAC signature")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    # 2. Validate payload against the strict contract.
    try:
        lead = LeadIn.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"validation_errors": exc.errors(include_url=False)},
        ) from exc

    # 3. Idempotency: replayed lead_id returns the original decision.
    cached = replay_cache.get(lead.lead_id)
    if cached is not None:
        logger.info("replay suppressed for lead_id=%s", lead.lead_id)
        return WebhookAck(**{**cached, "duplicate": True})

    # 4. Score via the registry model loaded at startup.
    feature_frame = build_feature_frame([lead.model_dump(include=set(FEATURE_COLUMNS))])
    score = _positive_probability(request.app.state.model, feature_frame)

    scored = ScoredLead(
        lead_id=lead.lead_id,
        email=str(lead.email),
        company_name=lead.company_name,
        payload=lead.model_dump(),
        score=score,
        band=band_for_score(score),
        model_version=request.app.state.model_version,
    )

    # 5. Route (Salesforce assignment vs nurture queue).
    try:
        outcome = engine.route(scored)
    except RoutingFailure as exc:
        logger.error("routing failed for lead_id=%s: %s", lead.lead_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"routing failed: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    ack = WebhookAck(
        accepted=True,
        duplicate=False,
        decision_id=outcome.decision_id or f"dec_{uuid.uuid4().hex[:12]}",
        score=round(score, 6),
        quality_band=scored.band.value,
        assigned_queue=outcome.queue,
        action=outcome.action,
        elapsed_ms=elapsed_ms,
    )
    replay_cache.put(lead.lead_id, ack.model_dump())
    return ack


__all__ = ["ReplayCache", "receive_lead", "router", "verify_signature"]
