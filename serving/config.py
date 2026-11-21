# File: serving/config.py
"""Environment-only configuration (12-factor).

All secrets are read exclusively from environment variables (optionally from
a git-ignored ``.env``). Required credentials have no defaults so the process
fails fast at startup instead of silently mis-routing leads.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """API + MLflow Model Registry settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "lead-scoring-routing-engine"
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "lead-quality-classifier"
    model_stage: str = Field(default="Production", pattern="^(Staging|Production)$")

    lead_webhook_secret: SecretStr = Field(
        description="HMAC-SHA256 secret used to sign inbound lead webhooks."
    )
    webhook_replay_cache_size: int = Field(default=10_000, ge=1)

    # Routing resilience knobs.
    sf_max_attempts: int = Field(default=4, ge=1)
    retry_base_delay_s: float = Field(default=0.25, gt=0)
    retry_max_delay_s: float = Field(default=8.0, gt=0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_timeout_s: float = Field(default=30.0, gt=0)


class SalesforceSettings(BaseSettings):
    """simple-salesforce credentials (username/password/security-token flow)."""

    model_config = SettingsConfigDict(
        env_prefix="SALESFORCE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    username: str
    password: SecretStr
    security_token: SecretStr
    domain: str = Field(default="login", pattern="^(login|test)$")
    api_version: str = "59.0"
    senior_rep_queue_developer_name: str = "Senior_Reps"


class SnowflakeSettings(BaseSettings):
    """Snowflake connection for the nurture write-through and batch scorer."""

    model_config = SettingsConfigDict(
        env_prefix="SNOWFLAKE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    account: str
    user: str
    password: SecretStr
    role: str = ""
    warehouse: str
    database: str
    schema_name: str = "LEADS"  # SNOWFLAKE_SCHEMA_NAME env var
    low_priority_table: str = "LOW_PRIORITY_LEADS"

    def connection_params(self) -> dict[str, str]:
        params = {
            "account": self.account,
            "user": self.user,
            "password": self.password.get_secret_value(),
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema_name,
        }
        if self.role:
            params["role"] = self.role
        return params


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings()  # type: ignore[call-arg]  # populated from env


@lru_cache(maxsize=1)
def get_salesforce_settings() -> SalesforceSettings:
    return SalesforceSettings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_snowflake_settings() -> SnowflakeSettings:
    return SnowflakeSettings()  # type: ignore[call-arg]
