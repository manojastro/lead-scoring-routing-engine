# File: tests/test_routing.py
"""Unit tests for the routing engine with the Salesforce client mocked out.

No network access is required: strategies are exercised against fakes, and
time.sleep is stubbed so backoff paths run instantly.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import OrderedDict
from typing import Any

import pytest

from serving.config import SnowflakeSettings
from serving.routing import (
    CircuitState,
    NurtureQueueStrategy,
    RoutingEngine,
    SalesforceAssignmentStrategy,
    ScoredLead,
)
from serving.salesforce_client import (
    SalesforceRejectedError,
    SalesforceUnavailableError,
)
from training.features import QualityBand


def make_lead(score: float = 0.9, band: QualityBand = QualityBand.HIGH) -> ScoredLead:
    return ScoredLead(
        lead_id="LEAD-test-000001",
        email="vp.eng@acme.io",
        company_name="Acme Corp",
        payload={
            "lead_id": "LEAD-test-000001",
            "source": "demo_request",
            "industry": "saas",
            "company_size": "201-500",
        },
        score=score,
        band=band,
        model_version="7",
    )


class FakeSalesforceClient:
    """In-memory stand-in for serving.salesforce_client.SalesforceClient."""

    def __init__(self, failures_before_success: int = 0) -> None:
        self.queue_id = "00GFAKEQUEUE0001"
        self.created_leads: list[dict[str, Any]] = []
        self.owner_lookups = 0
        self.failures_remaining = failures_before_success

    def get_queue_owner_id(self, developer_name: str) -> str:
        self.owner_lookups += 1
        return self.queue_id

    def create_lead(self, fields: dict[str, Any]) -> str:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise SalesforceUnavailableError("simulated 503", status_code=503)
        self.created_leads.append(fields)
        return "500FAKELEADID1"


class FakeCursor:
    def __init__(self, sink: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._sink = sink

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._sink.append((sql, params or ()))


class FakeSnowflakeConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.executed)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("serving.routing.time.sleep", lambda _s: None)


def make_sf_strategy(client: FakeSalesforceClient, **kwargs: Any) -> SalesforceAssignmentStrategy:
    options: dict[str, Any] = {"max_attempts": 3, "base_delay_s": 0.01}
    options.update(kwargs)
    return SalesforceAssignmentStrategy(
        client=client,  # type: ignore[arg-type]
        queue_developer_name="Senior_Reps",
        **options,
    )


# --------------------------------------------------------------------------- #
# Salesforce strategy
# --------------------------------------------------------------------------- #
class TestSalesforceStrategy:
    def test_high_band_creates_lead_owned_by_senior_queue(self) -> None:
        sf = FakeSalesforceClient()
        outcome = make_sf_strategy(sf).assign(make_lead())

        assert outcome.ok
        assert outcome.action == "salesforce_lead_created"
        assert len(sf.created_leads) == 1
        fields = sf.created_leads[0]
        assert fields["OwnerId"] == sf.queue_id
        assert fields["Email"] == "vp.eng@acme.io"
        assert fields["External_Lead_Id__c"] == "LEAD-test-000001"

    def test_queue_owner_resolved_once_and_cached(self) -> None:
        sf = FakeSalesforceClient()
        strategy = make_sf_strategy(sf)

        for _ in range(3):
            strategy.assign(make_lead())

        assert sf.owner_lookups == 1

    def test_transient_errors_are_retried_with_backoff(self) -> None:
        sf = FakeSalesforceClient(failures_before_success=2)
        outcome = make_sf_strategy(sf).assign(make_lead())

        assert outcome.ok and outcome.action == "salesforce_lead_created"
        assert len(sf.created_leads) == 1

    def test_permanent_rejection_is_not_retried(self) -> None:
        class RejectingClient(FakeSalesforceClient):
            def create_lead(self, fields: dict[str, Any]) -> str:
                raise SalesforceRejectedError("malformed field", status_code=400)

        outcome = make_sf_strategy(RejectingClient()).assign(make_lead())

        assert not outcome.ok
        assert outcome.action == "rejected_permanent"

    def test_circuit_breaker_opens_after_threshold_and_short_circuits(self) -> None:
        class AlwaysUnavailable(FakeSalesforceClient):
            calls = 0

            def create_lead(self, fields: dict[str, Any]) -> str:
                type(self).calls += 1
                raise SalesforceUnavailableError("503", status_code=503)

        sf = AlwaysUnavailable()
        strategy = make_sf_strategy(
            sf, breaker_failure_threshold=2, breaker_reset_timeout_s=3600.0, max_attempts=1
        )

        assert not strategy.assign(make_lead()).ok
        assert not strategy.assign(make_lead()).ok
        assert strategy.breaker.state is CircuitState.OPEN

        third = strategy.assign(make_lead())
        assert not third.ok and third.action == "short_circuited"
        assert AlwaysUnavailable.calls == 2  # no further client attempts


# --------------------------------------------------------------------------- #
# Nurture (Snowflake write-through) strategy
# --------------------------------------------------------------------------- #
class TestNurtureStrategy:
    @pytest.fixture()
    def nurture_with_fake_conn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[NurtureQueueStrategy, FakeSnowflakeConn]:
        settings = SnowflakeSettings(
            account="fake",
            user="svc",
            password="pw",
            warehouse="WH",
            database="CRM",
            schema_name="LEADS",
        )
        strategy = NurtureQueueStrategy(settings=settings, max_attempts=2, base_delay_s=0.01)
        conn = FakeSnowflakeConn()
        monkeypatch.setattr(strategy, "_connection", lambda: conn)
        return strategy, conn

    def test_low_band_lead_is_written_to_nurture_table(
        self, nurture_with_fake_conn: tuple[NurtureQueueStrategy, FakeSnowflakeConn]
    ) -> None:
        strategy, conn = nurture_with_fake_conn
        outcome = strategy.assign(make_lead(score=0.10, band=QualityBand.LOW))

        assert outcome.ok and outcome.action == "queued_for_nurture"
        sql, params = conn.executed[0]
        assert "LOW_PRIORITY_LEADS" in sql.upper()
        assert params[0] == "LEAD-test-000001"
        assert params[4] == "LOW"

    def test_engine_maps_bands_to_strategies(
        self, nurture_with_fake_conn: tuple[NurtureQueueStrategy, FakeSnowflakeConn]
    ) -> None:
        strategy, _conn = nurture_with_fake_conn
        engine = RoutingEngine(
            {
                QualityBand.MEDIUM: strategy,
                QualityBand.LOW: strategy,
            }
        )

        medium = engine.route(make_lead(score=0.5, band=QualityBand.MEDIUM))
        low = engine.route(make_lead(score=0.1, band=QualityBand.LOW))
        assert medium.ok and low.ok

        with pytest.raises(Exception):  # RoutingFailure for unregistered HIGH band
            engine.route(make_lead(band=QualityBand.HIGH))


# --------------------------------------------------------------------------- #
# Webhook signature + idempotency helpers
# --------------------------------------------------------------------------- #
class TestWebhookSecurity:
    def test_valid_signature_is_accepted(self) -> None:
        from serving.routes_leads import verify_signature

        body = b'{"lead_id": "LEAD-2026-000123"}'
        secret = b"shhh"
        good = hmac.new(secret, body, hashlib.sha256).hexdigest()

        assert verify_signature(body, f"sha256={good}", secret)
        assert verify_signature(body, good, secret)  # bare hex also tolerated

    def test_tampered_or_missing_signature_is_rejected(self) -> None:
        from serving.routes_leads import verify_signature

        body = b'{"lead_id": "LEAD-2026-000123"}'
        secret = b"shhh"
        bad_sig = "a" * 64

        assert not verify_signature(body, f"sha256={bad_sig}", secret)
        assert not verify_signature(body, None, secret)
        assert not verify_signature(body, "not-hex", secret)
        assert not verify_signature(b"tampered", hmac.new(secret, body, hashlib.sha256).hexdigest(), secret)


class TestReplayCache:
    def test_lru_eviction_and_duplicate_flag_source(self) -> None:
        from serving.routes_leads import ReplayCache

        cache = ReplayCache(max_size=2)
        cache.put("L1", {"accepted": True})
        cache.put("L2", {"accepted": True})
        cache.put("L3", {"accepted": True})

        assert cache.get("L1") is None  # evicted
        assert cache.get("L3") == {"accepted": True}

