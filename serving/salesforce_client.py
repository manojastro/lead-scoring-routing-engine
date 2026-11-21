# File: serving/salesforce_client.py
"""Thin, typed wrapper around simple-salesforce.

Design points:
  * lazy singleton connection (thread-safe) built from env-only credentials;
  * all vendor exceptions are mapped onto a small internal error hierarchy so
    the routing layer can distinguish *retryable* from *permanent* failures;
  * queue OwnerIds are resolved once via a Group SOQL query and cached.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from simple_salesforce.api import Salesforce
from simple_salesforce.exceptions import (
    SalesforceAuthenticationFailed,
    SalesforceGeneralError,
    SalesforceRefusedFunctionError,
    SalesforceResourceNotFound,
)

from serving.config import SalesforceSettings

logger = logging.getLogger("lead_engine.salesforce")


class SalesforceError(Exception):
    """Base class for wrapped Salesforce failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SalesforceAuthError(SalesforceError):
    """Credentials rejected - permanent, never retried."""


class SalesforceThrottledError(SalesforceError):
    """Rate limit / API quota exhausted - retryable with backoff."""


class SalesforceUnavailableError(SalesforceError):
    """5xx or transport failure - retryable with backoff."""


class SalesforceRejectedError(SalesforceError):
    """4xx validation/malformed request - permanent, surfaced to caller."""


_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _map_error(exc: Exception) -> SalesforceError:
    """Translate simple-salesforce/transport exceptions into our hierarchy."""
    if isinstance(exc, SalesforceAuthenticationFailed):
        return SalesforceAuthError(f"authentication failed: {exc}", status_code=401)
    if isinstance(exc, SalesforceRefusedFunctionError):
        return SalesforceRejectedError(f"call refused: {exc}")
    if isinstance(exc, SalesforceResourceNotFound):
        return SalesforceRejectedError(f"resource not found: {exc}", status_code=404)
    if isinstance(exc, SalesforceGeneralError):
        status = getattr(exc, "status", None)
        if status == 429:
            return SalesforceThrottledError(
                f"rate limited (status=429): {exc}", status_code=429
            )
        if status in _RETRYABLE_STATUS_CODES:
            return SalesforceUnavailableError(
                f"salesforce unavailable (status={status}): {exc}", status_code=status
            )
        if status == 401:
            return SalesforceAuthError(f"session expired: {exc}", status_code=401)
        return SalesforceRejectedError(f"request rejected (status={status}): {exc}", status_code=status)

    # requests.Timeout / ConnectionError and anything else -> treat as transient.
    return SalesforceUnavailableError(f"transport failure: {exc!r}")


class SalesforceClient:
    """Thread-safe lazy singleton around ``simple_salesforce.Salesforce``."""

    _instance: SalesforceClient | None = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: SalesforceSettings) -> None:
        self._settings = settings
        self._sf: Salesforce | None = None
        self._connect_lock = threading.Lock()
        self._queue_owner_cache: dict[str, str] = {}

    @classmethod
    def instance(cls, settings: SalesforceSettings | None = None) -> SalesforceClient:
        """Return the process-wide client (creating it on first call)."""
        with cls._instance_lock:
            if cls._instance is None:
                if settings is None:
                    from serving.config import get_salesforce_settings

                    settings = get_salesforce_settings()
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton (used by tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _connection(self) -> Salesforce:
        if self._sf is None:
            with self._connect_lock:
                if self._sf is None:
                    try:
                        self._sf = Salesforce(
                            username=self._settings.username,
                            password=self._settings.password.get_secret_value(),
                            security_token=self._settings.security_token.get_secret_value(),
                            domain=self._settings.domain,
                            version=self._settings.api_version,
                        )
                        logger.info("Salesforce connection established (domain=%s)", self._settings.domain)
                    except Exception as exc:  # noqa: BLE001 - mapped below
                        raise _map_error(exc) from exc
        return self._sf

    def invalidate_connection(self) -> None:
        """Force reconnect on next use (after auth/session expiry)."""
        with self._connect_lock:
            self._sf = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def create_lead(self, fields: dict[str, Any]) -> str:
        """Create a Lead and return its Salesforce Id.

        Raises:
            SalesforceAuthError, SalesforceThrottledError,
            SalesforceUnavailableError, SalesforceRejectedError
        """
        try:
            result = self._connection().Lead.create(fields)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _map_error(exc) from exc

        if not isinstance(result, dict) or not result.get("success"):
            errors = result.get("errors") if isinstance(result, dict) else result
            raise SalesforceRejectedError(f"Lead.create failed: {errors!r}")
        lead_id = result.get("id")
        if not isinstance(lead_id, str) or not lead_id:
            raise SalesforceRejectedError(f"Lead.create returned no id: {result!r}")
        logger.debug("Created Lead id=%s company=%s", lead_id, fields.get("Company"))
        return lead_id

    def get_queue_owner_id(self, developer_name: str) -> str:
        """Resolve a Queue's owner id (Group.Id); cached per developer name."""
        cached = self._queue_owner_cache.get(developer_name)
        if cached:
            return cached

        soql = (
            "SELECT Id FROM Group "
            "WHERE Type = 'Queue' AND DeveloperName = "
            f"'{developer_name.replace(chr(39), '')}' LIMIT 1"
        )
        try:
            rows = self._connection().query_all(soql).get("records", [])
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _map_error(exc) from exc

        if not rows:
            raise SalesforceRejectedError(
                f"queue '{developer_name}' not found (Group query returned no rows)"
            )
        owner_id = str(rows[0]["Id"])
        self._queue_owner_cache[developer_name] = owner_id
        logger.info("Resolved queue %s -> OwnerId=%s (cached)", developer_name, owner_id)
        return owner_id


__all__ = [
    "SalesforceClient",
    "SalesforceError",
    "SalesforceAuthError",
    "SalesforceThrottledError",
    "SalesforceUnavailableError",
    "SalesforceRejectedError",
]
