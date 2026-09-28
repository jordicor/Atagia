"""Resolve one calendar date with the selected Luna card and Python arithmetic."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, Self

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from atagia.services.llm_client import LLMClient


# Promoted verbatim from the selected clean_low experimental card.
CLEAN = """Read when the action happens. Python will do the calendar arithmetic.
Return one line: certainty|base|unit|value

Use ref as base when counting from the message date. Python already knows that date.
If the text says to count from a written date, copy that date as YYYY-MM-DD instead.
For a written final date, copy it and use days|0.
Never calculate a new date for base. Return analyze for an incomplete written date.

Use days, weeks, months, or years with one signed number: + means later, - means earlier.
Today, tomorrow, yesterday and similar words in any language give a usable number of days.
Combine years and months into months. Combine weeks and days into days. Include both amounts.
Convert decades, centuries and millennia to years.
Keep fractional months as decimals, unless the text defines fixed-length months; then convert everything to days.
For weekdays, use next_weekday or previous_weekday as unit and the English weekday name as value. Python finds that weekday strictly after or before the base.

Use exact for a definite date or time amount. Whole calendar months/years are exact, including at the end of a month. Fractions of years that equal whole months are exact.
Use uncertain for approximate numbers, numeric ranges, fractional calendar months, or a known month/year with no particular day.
Keep an approximate number; use the midpoint of a range. Next/last month or year without a day is uncertain and +1/-1.
Certainty describes the wording, not whether a future plan will really happen.
Return unknown alone if there is no usable timing, the amount is unspecified, or a fictional calendar has no known conversion.
Do not invent a number for a few, some or several. Zero means no change from the base date, never unknown.

The text may first describe an old plan and then give a new plan.
Use the new plan. Count from the message date, unless the new plan says to count from another date.

Examples:
The meeting is in eight days. -> exact|ref|days|+8
The repair was about seven weeks ago. -> uncertain|ref|weeks|-7
The visit was several months ago. -> unknown
Two days after April 19, 2037. -> exact|2037-04-19|days|+2"""


ReferenceDate = str | date | datetime


class DateOperation(BaseModel):
    """The model's arithmetic instruction, retained for audit and replay."""

    model_config = ConfigDict(extra="forbid")
    base: str
    unit: Literal["days", "weeks", "months", "years", "next_weekday", "previous_weekday"]
    value: str


class ParsedDateResolution(BaseModel):
    """Completed unknown, a usable date, and pending analysis are distinct."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "pending_analysis"]
    certainty: Literal["exact", "uncertain", "unknown"] | None = None
    resolved_date: str | None = None
    operation: DateOperation | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status == "pending_analysis":
            values = (self.certainty, self.resolved_date, self.operation)
            if any(value is not None for value in values):
                raise ValueError("Pending date analysis cannot contain a completed result")
        elif self.certainty == "unknown":
            if self.resolved_date is not None or self.operation is not None:
                raise ValueError("Unknown timing cannot contain a date or operation")
        elif self.certainty in {"exact", "uncertain"}:
            if self.resolved_date is None or self.operation is None:
                raise ValueError("Known timing requires a date and operation")
            _parse_calendar_date(self.resolved_date)
        else:
            raise ValueError("Completed date resolution requires certainty")
        return self


class DateResolution(ParsedDateResolution):
    """Versioned annotation bound to the exact source text and source date."""

    version: Literal[1] = 1
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_date: str | None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.reference_date is None:
            if self.status != "pending_analysis":
                raise ValueError("Completed date resolution requires a source reference date")
        else:
            _parse_calendar_date(self.reference_date)
        return self


def _parse_calendar_date(text: str) -> date:
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError("Expected exactly YYYY-MM-DD")
    return parsed


def reference_calendar_date(value: ReferenceDate) -> str:
    """Keep the source's calendar day, including its original timezone offset."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if len(value) == 10:
        return _parse_calendar_date(value).isoformat()
    return datetime.fromisoformat(value).date().isoformat()


def normalized_amount(unit: str, raw: str) -> tuple[str, Decimal]:
    """Normalize a model-produced numeric operation without interpreting text."""
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        raise ValueError("Invalid numeric amount") from None
    if not amount.is_finite() or abs(amount) > 100000:
        raise ValueError("Non-finite or excessive amount")
    if unit in {"days", "weeks"}:
        return "days", amount * (7 if unit == "weeks" else 1)
    if unit in {"months", "years"}:
        return "months", amount * (12 if unit == "years" else 1)
    raise ValueError("Expected days, weeks, months, or years")


def calculate(anchor: str, unit: str, raw: str) -> str:
    """Apply whole calendar months, then fractional months at 30 days/month.

    Remaining fractional days round to nearest, with ties away from zero.
    This is a representative-date convention, not a probability estimate.
    """
    start = _parse_calendar_date(anchor)
    dimension, amount = normalized_amount(unit, raw)
    if dimension == "months":
        whole_months = int(amount)
        start += relativedelta(months=whole_months)
        days = (amount - whole_months) * 30
    else:
        days = amount
    rounded_days = int(days.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return (start + timedelta(days=rounded_days)).isoformat()


def build_date_resolution_prompt(text: str, reference_date: ReferenceDate) -> str:
    """Build the selected single-user-message prompt without additional cards."""
    reference = reference_calendar_date(reference_date)
    return f"Reference date: {reference}\nText: {text}\n\n{CLEAN}\n\nReturn only the result."


def parse_date_resolution(text: str, reference_date: ReferenceDate) -> ParsedDateResolution:
    """Decode only the promoted smart wire format; malformed answers raise."""
    reference = reference_calendar_date(reference_date)
    answer = text.strip()
    if answer == "unknown":
        return ParsedDateResolution(status="completed", certainty="unknown")
    if answer == "analyze":
        return ParsedDateResolution(status="pending_analysis")
    fields = answer.split("|")
    if len(fields) != 4 or fields[0] not in {"exact", "uncertain"}:
        raise ValueError("Expected certainty|base|unit|amount, unknown, or analyze")
    certainty, base, unit, amount = fields
    anchor = reference if base == "ref" else base
    if unit in {"next_weekday", "previous_weekday"}:
        weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        amount = amount.lower()
        if amount not in weekdays:
            raise ValueError("Invalid weekday operation")
        start = _parse_calendar_date(anchor)
        direction = 1 if unit == "next_weekday" else -1
        distance = ((weekdays.index(amount) - start.weekday()) * direction) % 7 or 7
        actual = (start + timedelta(days=direction * distance)).isoformat()
    else:
        actual = calculate(anchor, unit, amount)
    return ParsedDateResolution(
        status="completed", certainty=certainty, resolved_date=actual,
        operation=DateOperation(base=base, unit=unit, value=amount),
    )


def pending_date_analysis(
    text: str, reference_date: ReferenceDate | None = None,
) -> DateResolution:
    """Represent missing source context without guessing an anchor or a date."""
    return DateResolution(
        status="pending_analysis", source_text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        reference_date=reference_calendar_date(reference_date) if reference_date is not None else None,
    )


async def resolve_date(
    llm_client: LLMClient,
    model: str,
    text: str,
    reference_date: ReferenceDate,
    metadata: dict[str, Any] | None = None,
) -> DateResolution:
    """Use ordinary routing, accounting, access policy, and technical recovery."""
    from atagia.services.llm_client import LLMCompletionRequest, LLMMessage

    reference = reference_calendar_date(reference_date)
    response = await llm_client.complete(LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="user", content=build_date_resolution_prompt(text, reference))],
        max_output_tokens=1024,
        metadata={**(metadata or {}), "purpose": "memory_date_resolution"},
    ))
    result = parse_date_resolution(response.output_text, reference)
    return DateResolution(
        **result.model_dump(), source_text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        reference_date=reference,
    )


def read_persisted_date_resolution(
    payload: dict[str, Any], text: str, reference_date: ReferenceDate | None,
) -> DateResolution | None:
    """Reuse a valid annotation only for the unchanged canonical text and anchor."""
    raw = payload.get("date_resolution")
    if not isinstance(raw, dict):
        return None
    try:
        annotation = DateResolution.model_validate(raw)
        reference = reference_calendar_date(reference_date) if reference_date is not None else None
    except (ValidationError, ValueError, TypeError):
        return None
    if annotation.reference_date != reference:
        return None
    if annotation.source_text_sha256 != sha256(text.encode("utf-8")).hexdigest():
        return None
    return annotation
