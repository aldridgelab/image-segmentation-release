"""
Configuration dataclass for fluorescence feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FluorescenceConfig:
    """Parameters controlling fluorescence feature extraction.

    Attributes:
        n_profile_points: Number of points to resample midline profiles to.
        strip_width: Width (in pixels) of the perpendicular strip for midline
            intensity sampling.
        strip_mode: Aggregation mode across the strip width.  ``"mean"`` is
            the default; ``"max"`` favours shell signal (useful for HADA).

        shell_erosion_radius: Structuring-element radius used to erode the mask
            when splitting it into shell (cell wall) and core (interior).
        bg_ring_dilation: Dilation radius for the outer edge of the
            background ring.
        bg_ring_erosion: Erosion radius for the inner edge of the background
            ring (should be <= bg_ring_dilation to leave a ring).

        pole_window: Fraction of the midline profile assigned to each pole
            (e.g. 0.10 means the first and last 10 %).
        center_window: Start and end fractions defining the cell centre
            window along the normalised midline.

        puncta_min_distance: Minimum pixel distance between detected puncta.
        puncta_threshold_rel: Relative intensity threshold passed to the
            LoG peak finder.
        puncta_gaussian_sigma: Gaussian smoothing sigma (pixels) used before
            LoG peak detection.
        puncta_log_sigma: Sigma (pixels) of the Laplacian-of-Gaussian filter.
        puncta_threshold_abs: Optional absolute threshold on the LoG response.
            ``None`` disables absolute thresholding.
        puncta_exclude_border: Whether to suppress peaks near image borders.
        puncta_use_core: If ``True``, restrict puncta search to the eroded
            core mask (avoids boundary artefacts).

        channel_aliases: Mapping from raw channel names to biologically
            readable aliases used in output column prefixes.
    """

    # --- Profile ---
    n_profile_points: int = 100
    strip_width: int = 3
    strip_mode: str = "mean"

    # --- Shell / core ---
    shell_erosion_radius: int = 2
    bg_ring_dilation: int = 5
    bg_ring_erosion: int = 3

    # --- HADA polar windows ---
    pole_window: float = 0.10
    center_window: tuple[float, float] = (0.45, 0.55)

    # --- Puncta detection ---
    puncta_min_distance: int = 1
    puncta_threshold_rel: float = 0.1
    puncta_gaussian_sigma: float = 0.5
    puncta_log_sigma: float = 1.0
    puncta_threshold_abs: float | None = None
    puncta_exclude_border: bool = True
    puncta_use_core: bool = True

    # --- Channel aliases ---
    channel_aliases: dict[str, str] = field(
        default_factory=lambda: {"AF405": "HADA", "Bod493": "BOD"}
    )
