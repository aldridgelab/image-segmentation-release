"""
Shared fluorescence feature extraction package.

Provides HADA (AF405) polar features, Bod493 puncta detection,
background subtraction, shell/core analysis, and midline intensity profiles.

Used by both the expanded analysis (analysis/) and deciphaer processing (processing/) pipelines.
"""

from fluorescence.config import FluorescenceConfig
from fluorescence.background import (
    compute_background_ring,
    background_stats,
    compute_shell_core,
    shell_core_stats,
)
from fluorescence.profiles import (
    compute_midline_profile,
    profile_scalar_summaries,
    reorient_profile,
)
from fluorescence.hada import (
    hada_polar_features,
    hada_state_classify,
)
from fluorescence.puncta import (
    detect_puncta,
    puncta_scalar_summary,
)
from fluorescence.sidecar import (
    write_profiles_parquet,
    write_puncta_parquet,
)

__all__ = [
    "FluorescenceConfig",
    "compute_background_ring",
    "background_stats",
    "compute_shell_core",
    "shell_core_stats",
    "compute_midline_profile",
    "profile_scalar_summaries",
    "reorient_profile",
    "hada_polar_features",
    "hada_state_classify",
    "detect_puncta",
    "puncta_scalar_summary",
    "write_profiles_parquet",
    "write_puncta_parquet",
]
