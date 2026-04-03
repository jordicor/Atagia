"""Clock utilities."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    """Abstract clock for testable time handling."""

    def now(self) -> datetime:
        """Return the current UTC timestamp."""


@dataclass(slots=True)
class SystemClock:
    """Production clock using the system UTC time."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


@dataclass(slots=True)
class FrozenClock:
    """Test helper clock with manual advancement."""

    current: datetime

    def now(self) -> datetime:
        return self.current

    def advance(self, *, seconds: float = 0.0) -> None:
        self.current = self.current + timedelta(seconds=seconds)

