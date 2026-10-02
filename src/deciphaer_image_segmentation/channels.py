"""Per-channel feature-extraction plugin registry and channel auto-discovery.

This module is the single place that knows how to turn a list of channels
(detected next to a Phase TIFF) into a column schema and a set of per-cell
feature extractions. New channels added to the imaging setup get a baseline
set of features automatically. Channels that match a registered plugin
(``AF405`` → HADA polar features, ``Bod493`` → puncta features) additionally
get bespoke features.

The registry is the only place to add new per-channel feature panels. To
add a new bespoke panel:

    1. Subclass :class:`ChannelPlugin`.
    2. Append an instance to :data:`CHANNEL_PLUGINS`.

The base fluorescence-feature pass in :mod:`extract` is shared by every
detected non-phase channel; plugins layer on extras only.

Channel-name discovery from filenames is registry-driven: see
:data:`KNOWN_CHANNELS` and :func:`parse_channel_token`. This handles real
microscope export quirks like a common trailing tag (e.g. ``..._Cy5_ORG.tif``)
and channel-shaped substrings in project names (e.g. ``..._sCy5DA-01_...``)
without false-matching them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known-channel registry — drives filename → channel parsing.
# Add new channels here when the imaging setup changes; the parser anchors
# matches on ``_<channel>`` followed by a separator so ordering doesn't matter.
# ---------------------------------------------------------------------------
KNOWN_CHANNELS: tuple[str, ...] = (
    "Phase",
    "AF405",
    "Bod493",
    "Cy5",
    "OG514",
)

# Separators that legitimately terminate a channel token in a filename stem.
# An underscore (the usual delimiter), a dot (channel-at-end-before-extension),
# or a space (e.g. ``_Phase PH3_ORG`` — Zen-style channel labels with a
# space-separated qualifier we want to ignore as trailing junk).
_CHANNEL_TOKEN_SEPS: tuple[str, ...] = ("_", " ", ".")


# ---------------------------------------------------------------------------
# Common feature suffix families.
# These are emitted for EVERY detected channel by the shared extractor in
# :mod:`extract`.  The Phase channel only gets the intensity/profile pair
# (no fluorescence-specific pass).
# ---------------------------------------------------------------------------
INTENSITY_SUFFIXES: tuple[str, ...] = (
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "range",
    "q1",
    "q3",
    "iqr",
    "p5",
    "p95",
    "cv",
    "skewness",
    "kurtosis",
    "entropy",
)
PROFILE_SUFFIXES: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "gradient_mean",
    "gradient_max",
    "polar_ratio",
    "polar_diff",
    "peak_position",
    "peak_value",
)
FLUOR_COMMON_SUFFIXES: tuple[str, ...] = (
    "bg_mean",
    "bg_median",
    "bg_std",
    "mean_bgsub",
    "shell_mean",
    "core_mean",
    "shell_core_ratio",
    "midline_mean",
    "midline_max",
    "midline_gradient_mean",
    "midline_gradient_max",
    "midline_peak_position",
    "midline_peak_value",
)


def intensity_feature_names(channel: str) -> list[str]:
    return [f"intensity_{channel}_{suf}" for suf in INTENSITY_SUFFIXES]


def profile_feature_names(channel: str) -> list[str]:
    return [f"profile_{channel}_{suf}" for suf in PROFILE_SUFFIXES]


def fluor_common_feature_names(channel: str) -> list[str]:
    return [f"{channel}_{suf}" for suf in FLUOR_COMMON_SUFFIXES]


def empty_intensity_features(channel: str) -> dict[str, float]:
    return {name: np.nan for name in intensity_feature_names(channel)}


def empty_profile_features(channel: str) -> dict[str, float]:
    return {name: np.nan for name in profile_feature_names(channel)}


def empty_fluor_common_features(channel: str) -> dict[str, float]:
    return {name: np.nan for name in fluor_common_feature_names(channel)}


# ---------------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PluginContext:
    """Inputs available to a channel plugin during per-cell extraction.

    A plugin sees the per-cell crops of the mask/image plus precomputed
    auxiliaries (midline, shell/core/bg-ring masks, the background-subtracted
    midline profile and the raw cell pixels). Plugins are pure: they read
    these and return a dict of additional feature values.
    """

    channel: str
    image: np.ndarray  # (H, W) float for THIS channel (cropped)
    binary_mask: np.ndarray  # (H, W) uint8, 1 inside the cell
    cell_pixels: np.ndarray  # 1-D array of in-mask pixels (float)
    midline: np.ndarray | None  # (N, 2) midline coords, or None
    profile: np.ndarray | None  # background-subtracted midline profile, or None
    core_mask: np.ndarray | None
    shell_mask: np.ndarray | None
    bg_value: float
    fluor_cfg: Any  # FluorescenceConfig
    puncta_records_out: list[dict[str, Any]]  # plugins may append puncta rows


class ChannelPlugin:
    """Adds bespoke features to channels matching :meth:`matches`.

    Subclasses must declare:

    * :attr:`name`           – short identifier used for logging only
    * :attr:`extra_suffixes` – column-suffix family added on top of the
      shared fluorescence-common columns.  ``f"{channel}_{suffix}"`` is
      the emitted column name.
    * :meth:`matches`        – return True if ``channel`` should run this
      plugin.  Default is case-insensitive substring of ``self.name``.
    * :meth:`extract`        – compute the extra features.  Must return a
      complete dict for every entry in :attr:`extra_suffixes`.
    """

    name: str = ""
    extra_suffixes: tuple[str, ...] = ()

    def matches(self, channel: str) -> bool:
        if not self.name:
            return False
        return self.name.lower() in channel.lower()

    def feature_names(self, channel: str) -> list[str]:
        return [f"{channel}_{suf}" for suf in self.extra_suffixes]

    def empty_features(self, channel: str) -> dict[str, float]:
        return {name: np.nan for name in self.feature_names(channel)}

    def extract(
        self, ctx: PluginContext
    ) -> dict[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in plugins
# ---------------------------------------------------------------------------
class HadaPlugin(ChannelPlugin):
    """HADA polar / septal features for AF405-like channels."""

    name = "AF405"
    extra_suffixes = (
        "poleA",
        "poleB",
        "center",
        "polar_asym",
        "center_enrichment",
        "peak_count",
        "state_code",
    )

    def extract(self, ctx: PluginContext) -> dict[str, float]:
        from fluorescence.hada import hada_polar_features, hada_state_classify

        if ctx.profile is None:
            return self.empty_features(ctx.channel)

        feats = hada_polar_features(
            ctx.profile,
            pole_window=ctx.fluor_cfg.pole_window,
            center_window=ctx.fluor_cfg.center_window,
        )
        state_code, _ = hada_state_classify(
            feats["polar_asym"],
            feats["center_enrichment"],
        )
        out: dict[str, float] = {f"{ctx.channel}_{k}": v for k, v in feats.items()}
        out[f"{ctx.channel}_state_code"] = state_code
        return out


class PunctaPlugin(ChannelPlugin):
    """Puncta detection for Bod493-like channels."""

    name = "Bod493"
    extra_suffixes = (
        "puncta_count",
        "puncta_total_intensity",
        "puncta_max_intensity",
        "puncta_mean_sigma_px",
        "puncta_max_sigma_px",
        "puncta_fraction",
    )

    def extract(self, ctx: PluginContext) -> dict[str, float]:
        from fluorescence.puncta import (
            detect_puncta,
            puncta_axial_radial,
            puncta_scalar_summary,
        )

        core_mask = ctx.core_mask if ctx.fluor_cfg.puncta_use_core else None
        puncta = detect_puncta(
            ctx.image,
            ctx.binary_mask,
            core_mask=core_mask,
            min_distance=ctx.fluor_cfg.puncta_min_distance,
            threshold_rel=ctx.fluor_cfg.puncta_threshold_rel,
            gaussian_sigma=ctx.fluor_cfg.puncta_gaussian_sigma,
            log_sigma=ctx.fluor_cfg.puncta_log_sigma,
            threshold_abs=ctx.fluor_cfg.puncta_threshold_abs,
            exclude_border=ctx.fluor_cfg.puncta_exclude_border,
            bg_value=ctx.bg_value,
        )
        cell_total = float(np.sum(ctx.cell_pixels)) if ctx.cell_pixels.size > 0 else 0.0
        summary = puncta_scalar_summary(puncta, cell_total)
        out = {f"{ctx.channel}_{k}": v for k, v in summary.items()}

        # Side-effect: append per-punctum rows so downstream sidecar can
        # reconstruct lipid-body coordinates per cell.
        if puncta.size > 0 and ctx.midline is not None:
            try:
                axrad = puncta_axial_radial(puncta, ctx.midline)
            except Exception:
                from loguru import logger
                logger.exception(
                    "puncta_axial_radial failed for channel {} (cell n={}); falling back to NaN axial/radial",
                    ctx.channel,
                    len(puncta),
                )
                axrad = np.zeros((0, 2))
            for j in range(len(puncta)):
                ctx.puncta_records_out.append(
                    {
                        "channel": ctx.channel,
                        "punctum_id": j,
                        "y_px": float(puncta[j, 0]),
                        "x_px": float(puncta[j, 1]),
                        "sigma_px": float(abs(puncta[j, 3])),
                        "peak_intensity": float(puncta[j, 2]),
                        "axial_s": float(axrad[j, 0]) if j < len(axrad) else np.nan,
                        "radial_r": float(axrad[j, 1]) if j < len(axrad) else np.nan,
                    }
                )
        return out


# Registry: append new plugins here. Order is irrelevant — multiple plugins
# can match the same channel (their features will all be added).
CHANNEL_PLUGINS: tuple[ChannelPlugin, ...] = (
    HadaPlugin(),
    PunctaPlugin(),
)


def plugins_for_channel(
    channel: str, plugins: Iterable[ChannelPlugin] = CHANNEL_PLUGINS
) -> list[ChannelPlugin]:
    return [p for p in plugins if p.matches(channel)]


# ---------------------------------------------------------------------------
# Feature-column schema for a given channel set
# ---------------------------------------------------------------------------
# Morphology + centerline columns produced by regionprops/centerline. These
# are the same for every run regardless of which channels are present.
MORPHOLOGY_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_px",
    "area_um2",
    "perimeter_px",
    "perimeter_um",
    "major_axis_px",
    "major_axis_um",
    "minor_axis_px",
    "minor_axis_um",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
    "convex_area_px",
    "filled_area_px",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "compactness",
    "circularity",
    "rough_length_px",
    "rough_length_um",
    "aspect_ratio",
    "hu_moment_0",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "centroid_row",
    "centroid_col",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "length_px",
    "length_um",
    "sinuosity",
    "width_median_px",
    "width_median_um",
    "width_std_px",
    "width_std_um",
    "width_max_px",
    "width_max_um",
    "width_min_px",
    "width_min_um",
    "width_q1_px",
    "width_q1_um",
    "width_q3_px",
    "width_q3_um",
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "curvature_min",
)


def feature_columns_for_channels(
    channels: Iterable[str],
    *,
    phase_channel: str,
    plugins: Iterable[ChannelPlugin] = CHANNEL_PLUGINS,
) -> tuple[str, ...]:
    """Return the full ordered feature-column list for a set of channels.

    Layout (matches the legacy 176-column reference schema when
    ``channels == ("Phase", "AF405", "Bod493")``):

        morphology + centerline   (54 cols)
        for each channel in channels:
            intensity_<channel>_*  (16 cols)
        for each fluor channel:
            profile_<channel>_*    (11 cols)
        for each fluor channel:
            fluor common <channel>_*  (13 cols)
            plugin extras <channel>_*  (variable, e.g. HADA=7, puncta=6)
    """
    channels = list(channels)
    if phase_channel not in channels:
        channels.insert(0, phase_channel)
    fluor_channels = [c for c in channels if c != phase_channel]

    cols: list[str] = list(MORPHOLOGY_FEATURE_COLUMNS)
    for ch in channels:
        cols.extend(intensity_feature_names(ch))
    for ch in channels:
        cols.extend(profile_feature_names(ch))
    for ch in fluor_channels:
        cols.extend(fluor_common_feature_names(ch))
        for plugin in plugins_for_channel(ch, plugins):
            cols.extend(plugin.feature_names(ch))
    return tuple(cols)


# ---------------------------------------------------------------------------
# Channel auto-discovery from an image directory
# ---------------------------------------------------------------------------
_TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_channel_token(
    stem: str,
    *,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
    phase_channel: str | None = None,
) -> tuple[str | None, str]:
    """Pull the channel name out of a filename stem.

    Returns ``(channel, base)``. Matches the rightmost occurrence of
    ``_<channel>`` (where ``<channel>`` is in ``known_channels``) followed
    by a separator in :data:`_CHANNEL_TOKEN_SEPS` or end-of-stem. Ties on
    position are broken by longer match, so a registry containing both
    ``"Phase"`` and ``"Phase PH3"`` prefers the more specific name.

    The leading-``_`` and trailing-separator constraints prevent false hits
    on channel-shaped substrings inside project/sample names (e.g. the
    ``Cy5`` inside ``sCy5DA-01`` is not preceded by ``_``, so it never
    matches).

    Falls back to legacy strict-suffix stripping (``stem.endswith(_<phase>)``)
    when no known channel hits, so naming like ``site01_Phase.tif`` /
    ``site01_HADA.tif`` continues to work even if ``HADA`` isn't in the
    registry. Returns ``(None, stem)`` if nothing matches.
    """
    matches: list[tuple[int, int, str]] = []
    for channel in known_channels:
        if not channel:
            continue
        needle = f"_{channel}"
        start = 0
        while True:
            idx = stem.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            if end == len(stem) or stem[end] in _CHANNEL_TOKEN_SEPS:
                matches.append((idx, len(needle), channel))
            start = idx + 1
    if matches:
        # Rightmost wins (closest to the extension); longer breaks ties.
        matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
        idx, _, channel = matches[0]
        return channel, stem[:idx]
    if phase_channel:
        suffix = f"_{phase_channel}"
        if stem.endswith(suffix):
            return phase_channel, stem[: -len(suffix)]
    return None, stem


def discover_sibling_channels(
    source: Path,
    *,
    phase_channel: str,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
    listing: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Path]:
    """Return ``{channel: path}`` for sibling TIFFs sharing a base with ``source``.

    Identifies the channel + base of ``source`` (via :func:`parse_channel_token`),
    then scans either ``listing`` (pre-built ``(stem, resolved_path)`` pairs)
    or ``source.parent`` for TIFFs whose parsed base matches. Excludes the
    source itself, masks, and the phase channel.

    A sibling that doesn't parse as a known channel but whose stem starts
    with ``<base>_`` is treated as ``stem[len(base)+1:]`` — this preserves
    legacy naming (``site01_HADA.tif`` next to ``site01_Phase.tif``) without
    requiring every custom channel to be added to the registry.
    """
    known = tuple(known_channels)
    _, base = parse_channel_token(
        source.stem, known_channels=known, phase_channel=phase_channel,
    )
    if listing is None:
        try:
            entries = [
                (c.stem, str(c.resolve()))
                for c in source.parent.iterdir()
                if c.is_file() and c.suffix.lower() in _TIFF_SUFFIXES
            ]
        except OSError:
            return {}
    else:
        entries = list(listing)
    source_resolved = str(source.resolve())
    out: dict[str, Path] = {}
    for stem, path in entries:
        if path == source_resolved:
            continue
        ch, b = parse_channel_token(
            stem, known_channels=known, phase_channel=phase_channel,
        )
        # Legacy fallback: if no registry hit but stem starts with our base,
        # treat the trailing token as the channel name.
        if ch is None and base:
            prefix = f"{base}_"
            if stem.startswith(prefix):
                ch = stem[len(prefix):]
                b = base
        if ch is None:
            continue
        if not _is_valid_channel(ch, phase_channel):
            continue
        if b != base:
            continue
        out.setdefault(ch, Path(path))
    return out


def discover_channels_in_dir(
    image_dir: Path,
    *,
    phase_channel: str,
    pattern: str = "*Phase*.tif",
    recursive: bool = False,
    sample_limit: int = 50,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
) -> tuple[str, ...]:
    """Return channels present alongside Phase TIFFs in ``image_dir``.

    Walks Phase files matched by ``pattern`` and, for each, identifies
    sibling channel files via :func:`discover_sibling_channels`. The
    returned tuple always starts with ``phase_channel`` followed by detected
    fluorescence channels sorted alphabetically for determinism.

    Returns ``(phase_channel,)`` if no Phase files or sibling channels are
    found — the extractor handles that gracefully.

    ``sample_limit`` caps how many Phase files are probed; channel naming
    is consistent enough across a run that 50 is plenty.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return (phase_channel,)

    if recursive:
        phase_files = list(image_dir.rglob(pattern))
    else:
        phase_files = list(image_dir.glob(pattern))
    if not phase_files:
        return (phase_channel,)

    known = tuple(known_channels)
    found: set[str] = set()
    for source in phase_files[:sample_limit]:
        siblings = discover_sibling_channels(
            source, phase_channel=phase_channel, known_channels=known,
        )
        found.update(siblings.keys())

    return (phase_channel, *sorted(found))


def _is_valid_channel(name: str, phase_channel: str) -> bool:
    if not name:
        return False
    if name == phase_channel:
        return False
    low = name.lower()
    if low == "mask" or low.endswith("_mask"):
        return False
    return True
