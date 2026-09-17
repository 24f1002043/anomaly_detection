"""
SIH26170 - AI-Driven Anomaly Detection in Component Burn-In & Screening.

Two modules against one checkpoint grid (0 h, 24 h, 96 h, 168 h):

  Module A  dynamic lot-relative outlier detection over the completed soak
  Module B  Value_0h + Value_24h -> Value_168h drift prediction with a
            safety-slope early-rejection decision at 24 h

Every flag carries a complete, automatically generated explanation record.
"""

__version__ = "1.0.0"
__problem_statement__ = "SIH26170"

from burnin.config import Config, load_config  # noqa: E402,F401
