"""Bind authorized reporting coordinates to both existing cache boundaries."""
from dataclasses import asdict, dataclass
import json
import re


@dataclass(frozen=True)
class ReportingCoordinates:
    """Canonical coordinates resolved by the report application before caching."""

    entity: str
    month: str
    population: str

    def __post_init__(self):
        if not isinstance(self.month, str) or not re.fullmatch(r"[1-9]\d{3}-(?:0[1-9]|1[0-2])", self.month, re.ASCII):
            raise ValueError("expected a calendar month in YYYY-MM format")
        for value in (self.entity, self.population):
            if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value, re.ASCII):
                raise ValueError("expected a canonical report identifier")

    def bind(self, body):
        context = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        system, user = body["messages"]
        if system["role"] != "system" or user["role"] != "user":
            raise ValueError("expected the reporting application's system and user messages")
        return {
            **body,
            "messages": [
                {**system, "content": system["content"] + "\nAuthorized report coordinates: " + context},
                user,
            ],
        }


def process_report(cache, body, *, provider, coordinates):
    """Call only with coordinates from an authorized report lookup."""
    if coordinates is None:
        return cache.process(body, provider=provider, bypass=True)
    return cache.process(coordinates.bind(body), provider=provider)
