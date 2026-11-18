"""Trip loading interfaces. P1 implements this module."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[list[int]] = None


def load_trip(split: str, trip_id: str, with_route: bool = True) -> Trip:
    """Load one trip. Implementation owned by P1."""
    raise NotImplementedError("P1: implement load_trip")


def list_trips(split: str) -> list[str]:
    """List available trip IDs for a split. Implementation owned by P1."""
    raise NotImplementedError("P1: implement list_trips")
