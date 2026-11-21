# File: serving/schemas.py
"""Pydantic v2 request/response contracts for the lead webhook API.

Field constraints intentionally mirror ``training/features.py`` so malformed
or drifted payloads are rejected with 422 *before* they can poison inference.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

CompanySize = Literal["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
Industry = Literal[
    "saas", "fintech", "healthcare", "ecommerce", "manufacturing", "media", "logistics"
]
LeadSource = Literal[
    "demo_request", "referral", "inbound_web", "event", "partner", "purchased_list"
]
Region = Literal["us_west", "us_east", "emea", "apac", "latam"]
FundingStage = Literal[
    "bootstrap", "seed", "series_a", "series_b", "series_c_plus", "unknown"
]
QualityBandName = Literal["HIGH", "MEDIUM", "LOW"]


class LeadIn(BaseModel):
    """Inbound lead payload (identity + the exact model feature columns)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lead_id: str = Field(
        pattern=r"^LEAD-[A-Za-z0-9_-]{4,64}$",
        examples=["LEAD-2026-000123"],
        description="Idempotency key supplied by the source system.",
    )
    email: EmailStr = Field(examples=["vp.engineering@acme.io"])
    company_name: str = Field(min_length=1, max_length=200, examples=["Acme Corp"])

    # ---- feature columns (must match training/features.py exactly) ----
    company_size: CompanySize = Field(examples=["201-500"])
    industry: Industry = Field(examples=["saas"])
    source: LeadSource = Field(examples=["demo_request"])
    region: Region = Field(examples=["us_west"])
    funding_stage: FundingStage = Field(default="unknown", examples=["series_c_plus"])
    engagement_score: float = Field(ge=0.0, le=100.0, examples=[82.5])
    emails_opened: int = Field(ge=0, le=10_000, examples=[11])
    website_visits: int = Field(ge=0, le=100_000, examples=[27])


class ScoreOut(BaseModel):
    """Raw scoring result before routing."""

    model_config = ConfigDict(frozen=True)

    lead_id: str
    score: float = Field(ge=0.0, le=1.0, description="Conversion probability.")
    quality_band: QualityBandName
    model_version: str = Field(examples=["7"])


class WebhookAck(BaseModel):
    """Decision receipt returned to the caller."""

    accepted: bool
    duplicate: bool = False
    decision_id: str = Field(examples=["dec_2f8a1c94"])
    score: float | None = None
    quality_band: QualityBandName | None = None
    assigned_queue: str | None = Field(
        default=None,
        examples=["senior_reps"],
        description="Destination queue; None when routing failed.",
    )
    action: str | None = Field(
        default=None,
        examples=["salesforce_lead_created", "queued_for_nurture"],
    )
    elapsed_ms: float = Field(default=0.0, ge=0.0)


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None = None
    model_stage: str | None = None
    schema_version: str | None = None
