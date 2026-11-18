"""Feature construction interface. P1 implements this module."""

FEATURE_COLUMNS: list[str] = []


def build_features(df):
    """Build model features without using reference-route information."""
    raise NotImplementedError("P1: implement build_features")
