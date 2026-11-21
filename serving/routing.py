# File: serving/routing.py
"""Dynamic routing engine: quality band -> assignment strategy.

Flow per decision:
    1. ``RoutingEngine.route`` selects the strategy for the lead's band
       (HIGH -> Salesforce senior-rep queue; MEDIUM/LOW -> Snowflake nurture).
    2. The strategy call is wrapped by a **circuit breaker** (per destination)
       and **exponential backoff with full jitter** on retryable errors.
    3. Every outcome - success, failure, or short-circuit - is appended to an
       audit log as a structured JSON line.

Strategies implement a common Protocol (Strategy pattern) so adding e.g. an
SDR-round-robin or Slack-notify path requires no engine changes.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

from serving.config import SnowflakeSettings
from serving.salesforce_client import (
    SalesforceAuthError,
    SalesforceClient,
    SalesforceError,
    SalesforceRejectedError,
    SalesforceThrottledError,
    SalesforceUnavailableError,
)
from training.features import QualityBand

logger = logging.getLogger("lead_engine.routing")
audit_logger = logging.getLogger("lead_engine.audit")

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Decision model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScoredLead:
    """Everything the routing engine needs to act."""

    lead_id: str
    email: str
    company_name: str
    payload: dict[str, Any]  # raw LeadIn dump (includes feature fields)
    score: float
    band: QualityBand
    model_version: str


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    """Result of one routing attempt."""

    decision_id: str
    queue: str
    action: str
    ok: bool
    detail: str = ""


class RoutingFailure(RuntimeError):
    """Raised when no strategy could durably place the lead."""


# --------------------------------------------------------------------------- #
# Resilience primitives
# --------------------------------------------------------------------------- #
class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Minimal thread-safe circuit breaker (failure-count based)."""

    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout_s: float = 30.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state is CircuitState.OPEN
                and time.monotonic() - self._opened_at >= self.reset_timeout_s
            ):
                self._state = CircuitState.HALF_OPEN
            return self._state

    def allow_attempt(self) -> bool:
        """Whether a call may proceed right now."""
        return self.state is not CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning("circuit '%s' OPENED after %d failures", self.name, self._consecutive_failures)


def with_retry(
    operation: Callable[[], T],
    *,
    what: str,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    retry_on: tuple[type[BaseException], ...],
) -> T:
    """Run ``operation`` with exponential backoff + full jitter.

    Permanent errors (``SalesforceAuthError`` / ``SalesforceRejectedError``)
    propagate immediately.
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except retry_on as exc:  # transient
            last_error = exc
            if attempt == max_attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay = random.uniform(0.0, delay)  # full jitter
            logger.warning(
                "transient error on %s (attempt %d/%d): %s -- retrying in %.2fs",
                what, attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


RETRYABLE_SF_ERRORS: tuple[type[BaseException], ...] = (
    SalesforceThrottledError,
    SalesforceUnavailableError,
)

# Midpoint headcount per company-size band (Salesforce NumberOfEmployees is numeric).
_EMPLOYEE_BAND_MIDPOINT: dict[str, int] = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1000+": 2500,
}


def _employee_count(company_size: str | None) -> int | None:
    if company_size is None:
        return None
    return _EMPLOYEE_BAND_MIDPOINT.get(str(company_size))


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class AssignmentStrategy:
    """Interface + shared resilience plumbing for all routing targets."""

    name: str = "base"

    def __init__(
        self,
        *,
        max_attempts: int = 4,
        base_delay_s: float = 0.25,
        max_delay_s: float = 8.0,
        breaker_failure_threshold: int = 5,
        breaker_reset_timeout_s: float = 30.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.breaker = CircuitBreaker(
            self.name,
            failure_threshold=breaker_failure_threshold,
            reset_timeout_s=breaker_reset_timeout_s,
        )

    def assign(self, lead: ScoredLead) -> RouteOutcome:
        raise NotImplementedError

    def _protected_call(self, operation_name: str, operation: Callable[[], RouteOutcome]) -> RouteOutcome:
        """Circuit-breaker gate around a strategy operation."""
        if not self.breaker.allow_attempt():
            return RouteOutcome(
                decision_id="",
                queue=self.name,
                action="short_circuited",
                ok=False,
                detail=f"circuit open for {self.name}",
            )
        try:
            outcome = with_retry(
                operation,
                what=operation_name,
                max_attempts=self.max_attempts,
                base_delay_s=self.base_delay_s,
                max_delay_s=self.max_delay_s,
                retry_on=RETRYABLE_SF_ERRORS,
            )
        except SalesforceAuthError as exc:
            self.breaker.record_failure()
            return RouteOutcome(self._new_decision_id(), self.name, "auth_failed", False, str(exc))
        except (SalesforceThrottledError, SalesforceUnavailableError) as exc:
            self.breaker.record_failure()
            return RouteOutcome(self._new_decision_id(), self.name, "destination_unavailable", False, str(exc))
        except SalesforceRejectedError as exc:
            # Permanent: breaker stays closed; payload must be fixed upstream.
            return RouteOutcome(self._new_decision_id(), self.name, "rejected_permanent", False, str(exc))
        except Exception as exc:  # noqa: BLE001 - defensive catch-all
            self.breaker.record_failure()
            return RouteOutcome(self._new_decision_id(), self.name, "unexpected_error", False, repr(exc))
        else:
            self.breaker.record_success()
            return RouteOutcome(
                decision_id=outcome.decision_id or self._new_decision_id(),
                queue=outcome.queue,
                action=outcome.action,
                ok=True,
                detail=outcome.detail,
            )

    @staticmethod
    def _new_decision_id() -> str:
        return f"dec_{uuid.uuid4().hex[:12]}"


class SalesforceAssignmentStrategy(AssignmentStrategy):
    """HIGH-band leads: create a Salesforce Lead owned by the senior-rep queue."""

    name: str = "senior_reps"

    def __init__(self, client: SalesforceClient, queue_developer_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._queue_developer_name = queue_developer_name
        self._owner_id: str | None = None
        self._owner_lock = threading.Lock()

    def resolve_owner_id(self) -> str:
        """Resolve the queue OwnerId once and cache it for the process."""
        with self._owner_lock:
            if self._owner_id is None:

                def _resolve() -> str:
                    return self._client.get_queue_owner_id(self._queue_developer_name)

                self._owner_id = with_retry(
                    _resolve,
                    what="queue-owner lookup",
                    max_attempts=self.max_attempts,
                    base_delay_s=self.base_delay_s,
                    max_delay_s=self.max_delay_s,
                    retry_on=RETRYABLE_SF_ERRORS,
                )
            return self._owner_id

    def assign(self, lead: ScoredLead) -> RouteOutcome:
        def _operation() -> RouteOutcome:
            owner_id = self.resolve_owner_id()
            sf_fields = {
                "LastName": lead.payload.get("last_name") or lead.email.split("@", 1)[0],
                "FirstName": lead.payload.get("first_name", ""),
                "Email": lead.email,
                "Company": lead.company_name,
                "LeadSource": lead.payload["source"],
                "Industry": lead.payload["industry"],
                "OwnerId": owner_id,
                # Idempotency: duplicate webhook replays are deduped by SF.
                "External_Lead_Id__c": lead.lead_id,
                "Description": (
                    f"[auto-route] band={lead.band.value} score={lead.score:.4f} "
                    f"model_v{lead.model_version}"
                ),
                "NumberOfEmployees": _employee_count(lead.payload.get("company_size")),
            }
            lead_sf_id = self._client.create_lead(sf_fields)
            return RouteOutcome(
                decision_id=self._new_decision_id(),
                queue=self.name,
                action="salesforce_lead_created",
                ok=True,
                detail=f"sf_lead_id={lead_sf_id}",
            )

        return self._protected_call("salesforce.assign", _operation)


class NurtureQueueStrategy(AssignmentStrategy):
    """MEDIUM/LOW leads: write-through to the Snowflake low-priority table."""

    name: str = "nurture_low_priority"

    def __init__(self, settings: SnowflakeSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._conn: Any = None  # snowflake.connector.SnowflakeConnection
        self._conn_lock = threading.Lock()

    def _connection(self) -> Any:
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    import snowflake.connector

                    self._conn = snowflake.connector.connect(**self._settings.connection_params())
        return self._conn

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def assign(self, lead: ScoredLead) -> RouteOutcome:
        def _operation() -> RouteOutcome:
            conn = self._connection()
            # Idempotent insert: replays of the same LEAD_ID are no-ops.
            sql = (
                f"INSERT INTO {self._settings.database}.{self._settings.schema_name}."
                f"{self._settings.low_priority_table} "
                "(LEAD_ID, EMAIL, COMPANY_NAME, SCORE, QUALITY_BAND, MODEL_VERSION, "
                " SCORED_AT, ROUTED_AT) "
                "SELECT v.* FROM (SELECT %s AS LEAD_ID, %s AS EMAIL, %s AS COMPANY_NAME, "
                "%s AS SCORE, %s AS QUALITY_BAND, %s AS MODEL_VERSION, %s AS SCORED_AT, "
                "CURRENT_TIMESTAMP() AS ROUTED_AT) v "
                f"WHERE NOT EXISTS (SELECT 1 FROM {self._settings.low_priority_table} t "
                "WHERE t.LEAD_ID = v.LEAD_ID)"
            )
            params = (
                lead.lead_id, lead.email, lead.company_name,
                float(lead.score), lead.band.value, lead.model_version,
                datetime.now(timezone.utc).isoformat(),
            )
            with conn.cursor() as cur:
                cur.execute(sql, params)
            return RouteOutcome(
                decision_id=self._new_decision_id(),
                queue=self.name,
                action="queued_for_nurture",
                ok=True,
                detail=f"snowflake={self._settings.low_priority_table}",
            )

        return self._protected_call("nurture.write", _operation)


# --------------------------------------------------------------------------- #
# Engine facade
# --------------------------------------------------------------------------- #
class RoutingEngine:
    """Chooses a strategy per band, executes it, and audits every decision."""

    def __init__(self, strategies: dict[QualityBand, AssignmentStrategy]) -> None:
        if not strategies:
            raise ValueError("strategies mapping must not be empty")
        self._strategies = dict(strategies)

    def strategy_for(self, band: QualityBand) -> AssignmentStrategy:
        strategy = self._strategies.get(band)
        if strategy is None:
            raise RoutingFailure(f"no strategy registered for band={band.value}")
        return strategy

    def route(self, lead: ScoredLead) -> RouteOutcome:
        started = time.perf_counter()
        decision_id = AssignmentStrategy._new_decision_id()
        try:
            strategy = self.strategy_for(lead.band)
        except RoutingFailure as exc:
            self._audit(decision_id, lead, None, "no_strategy", False, str(exc), started)
            raise

        outcome = strategy.assign(lead)
        final = RouteOutcome(
            decision_id=decision_id or outcome.decision_id,
            queue=outcome.queue,
            action=outcome.action,
            ok=outcome.ok,
            detail=outcome.detail,
        )
        self._audit(final.decision_id, lead, strategy, final.action, final.ok, final.detail, started)
        if not final.ok:
            raise RoutingFailure(f"routing failed ({final.action}): {final.detail}")
        return final

    @staticmethod
    def _audit(
        decision_id: str,
        lead: ScoredLead,
        strategy: AssignmentStrategy | None,
        action: str,
        ok: bool,
        detail: str,
        started: float,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "lead_route_decision",
            "decision_id": decision_id,
            "lead_id": lead.lead_id,
            "score": round(lead.score, 6),
            "quality_band": lead.band.value,
            "model_version": lead.model_version,
            "strategy": strategy.name if strategy else None,
            "action": action,
            "ok": ok,
            "detail": detail,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        audit_logger.info(json.dumps(record, separators=(",", ":")))


def build_default_engine(
    sf_client: SalesforceClient,
    sf_settings: Any,
    snow_settings: SnowflakeSettings,
    app_settings: Any,
) -> RoutingEngine:
    """Wire the production strategy set from env-driven settings."""
    salesforce_strategy = SalesforceAssignmentStrategy(
        client=sf_client,
        queue_developer_name=sf_settings.senior_rep_queue_developer_name,
        max_attempts=app_settings.sf_max_attempts,
        base_delay_s=app_settings.retry_base_delay_s,
        max_delay_s=app_settings.retry_max_delay_s,
        breaker_failure_threshold=app_settings.breaker_failure_threshold,
        breaker_reset_timeout_s=app_settings.breaker_reset_timeout_s,
    )
    nurture_strategy = NurtureQueueStrategy(
        settings=snow_settings,
        max_attempts=app_settings.sf_max_attempts,
        base_delay_s=app_settings.retry_base_delay_s,
        max_delay_s=app_settings.retry_max_delay_s,
        breaker_failure_threshold=app_settings.breaker_failure_threshold,
        breaker_reset_timeout_s=app_settings.breaker_reset_timeout_s,
    )
    return RoutingEngine(
        {
            QualityBand.HIGH: salesforce_strategy,
            QualityBand.MEDIUM: nurture_strategy,
            QualityBand.LOW: nurture_strategy,
        }
    )


__all__ = [
    "AssignmentStrategy",
    "CircuitBreaker",
    "CircuitState",
    "NurtureQueueStrategy",
    "RouteOutcome",
    "RoutingEngine",
    "RoutingFailure",
    "ScoredLead",
    "SalesforceAssignmentStrategy",
    "build_default_engine",
]
