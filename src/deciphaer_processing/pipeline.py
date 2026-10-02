"""
Main pipeline for single-cell feature extraction.

Orchestrates feature extraction from manual masks or U-Net predictions,
with filtering, hashing, and export to deciphaer format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
from loguru import logger
from scipy import ndimage
from tqdm import tqdm

from deciphaer_processing.features.regionprops import extract_regionprops, apply_morphology_filters
from deciphaer_processing.features.centerline import extract_centerline_features, compute_midline
from deciphaer_processing.features.intensity import extract_intensity_features
from deciphaer_processing.hashing.cell_hash import generate_cell_id, parse_filename_metadata
from deciphaer_processing.hashing.lookup import CellLookupTable
from deciphaer_processing.export.deciphaer_format import export_deciphaer_format, export_features_polars
from deciphaer_processing.session_output import (
    apply_cell_annotations,
    load_annotation_maps,
    normalize_review_status,
    resolve_central_label,
    resolve_channel_names,
    resolve_image_path,
    resolve_mask_path,
)
from fluorescence import FluorescenceConfig
from fluorescence.background import (
    compute_background_ring, background_stats,
    compute_shell_core, shell_core_stats,
)
from fluorescence.profiles import compute_midline_profile, profile_scalar_summaries, reorient_profile
from fluorescence.hada import hada_polar_features, hada_state_classify
from fluorescence.puncta import detect_puncta, puncta_scalar_summary, puncta_axial_radial
from fluorescence.sidecar import write_profiles_parquet, write_puncta_parquet


@dataclass
class FilterParams:
    """Parameters for cell quality filtering."""

    # Area filters (in pixels)
    area_min: float | None = 100.0
    area_max: float | None = 500.0

    # Shape filters
    eccentricity_min: float | None = None
    eccentricity_max: float | None = None
    solidity_min: float | None = None
    solidity_max: float | None = None
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None

    # Edge exclusion
    exclude_edge: bool = True
    edge_distance: int = 8


@dataclass
class PipelineConfig:
    """Configuration for the feature extraction pipeline."""

    # Pixel calibration
    pixel_microns: float = 0.10317

    # Channel configuration
    channel_names: list[str] = field(default_factory=lambda: ["Phase", "AF405", "Bod493"])

    # Filter parameters
    filter_params: FilterParams = field(default_factory=FilterParams)

    # Output options
    output_parquet: bool = True
    output_csv: bool = True

    # Processing options
    extract_centerline: bool = True
    extract_intensity: bool = True
    extract_profiles: bool = True

    # Fluorescence feature extraction
    extract_fluorescence: bool = True
    fluor_config: FluorescenceConfig = field(default_factory=FluorescenceConfig)

    # U-Net thresholds
    unet_cell_prob_min: float = 0.6
    unet_cell_margin: float = 0.1

    # Mask export options
    save_masks: bool = False
    masks_output_dir: str | None = None

    # Data hygiene
    drop_nan_cells: bool = True


class SingleCellPipeline:
    """
    Main pipeline for single-cell feature extraction.

    Supports two modes:
    1. Manual masks: Process pre-computed masks with metadata JSON
    2. U-Net predictions: Run U-Net inference and extract features
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        output_dir: str | Path = "processing_output",
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
            output_dir: Output directory for results
        """
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lookup table
        self.lookup_table = CellLookupTable()

        # Mask export directory
        if self.config.save_masks:
            masks_dir = Path(self.config.masks_output_dir) if self.config.masks_output_dir else self.output_dir / "masks"
            self.masks_dir = masks_dir
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Mask export enabled, saving to {self.masks_dir}")
        else:
            self.masks_dir = None

        # Results storage
        self.features_list: list[dict[str, Any]] = []
        self.cell_ids: list[str] = []
        self.metadata_list: list[dict[str, Any]] = []
        self.filtered_cells: list[dict[str, Any]] = []
        self.mask_manifest: list[dict[str, Any]] = []
        self.all_profile_records: list[dict[str, Any]] = []
        self.all_puncta_records: list[dict[str, Any]] = []

    def process_manual_masks(
        self,
        masks_dir: str | Path,
        metadata_dir: str | Path,
        images_dir: str | Path,
        progress: bool = True,
    ) -> dict[str, Any]:
        """
        Process manually curated masks with existing metadata.

        Args:
            masks_dir: Directory containing mask TIFFs
            metadata_dir: Directory containing metadata JSONs
            images_dir: Directory containing source images
            progress: Show progress bar

        Returns:
            Summary statistics dictionary
        """
        masks_dir = Path(masks_dir)
        metadata_dir = Path(metadata_dir)
        images_dir = Path(images_dir)

        # Find all mask files
        mask_files = sorted(masks_dir.glob("*.tif")) + sorted(masks_dir.glob("*.tiff"))
        logger.info(f"Found {len(mask_files)} mask files in {masks_dir}")

        stats = {
            "total_masks": len(mask_files),
            "total_cells": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "errors": 0,
        }

        iterator = tqdm(mask_files, desc="Processing masks") if progress else mask_files

        for mask_path in iterator:
            try:
                # Find corresponding metadata and image
                base_name = mask_path.stem
                if base_name.endswith("_mask"):
                    base_name = base_name[:-5]

                meta_path = metadata_dir / f"{base_name}_meta.json"
                image_path = images_dir / f"{base_name}.tif"

                if not meta_path.exists():
                    logger.warning(f"Metadata not found: {meta_path}")
                    stats["errors"] += 1
                    continue

                if not image_path.exists():
                    # Try alternative extensions
                    for ext in [".tiff", ".TIF", ".TIFF"]:
                        alt_path = images_dir / f"{base_name}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break
                    else:
                        logger.warning(f"Image not found: {image_path}")
                        stats["errors"] += 1
                        continue

                # Load data
                mask = tifffile.imread(mask_path)
                image = tifffile.imread(image_path)
                with open(meta_path) as f:
                    metadata = json.load(f)

                # Process cells from metadata
                cells_info = metadata.get("cells", [])
                for cell_info in cells_info:
                    if not cell_info.get("included", True):
                        stats["failed_filter"] += 1
                        continue

                    label = cell_info["label"]
                    bbox = tuple(cell_info["bbox"])

                    # Extract features
                    result = self._process_single_cell(
                        mask=mask,
                        image=image,
                        label=label,
                        bbox=bbox,
                        source_path=image_path,
                        metadata=metadata,
                    )

                    stats["total_cells"] += 1

                    if result["passed_filter"]:
                        stats["passed_filter"] += 1
                        self.features_list.append(result["features"])
                        self.cell_ids.append(result["cell_id"])
                        self.metadata_list.append(result["metadata"])
                        self.all_profile_records.extend(result.get("profile_records", []))
                        self.all_puncta_records.extend(result.get("puncta_records", []))
                    else:
                        stats["failed_filter"] += 1
                        self.filtered_cells.append({
                            "cell_id": result["cell_id"],
                            "reasons": result["failure_reasons"],
                        })

            except Exception as e:
                logger.error(f"Error processing {mask_path}: {e}")
                stats["errors"] += 1

        logger.info(f"Processed {stats['total_cells']} cells, {stats['passed_filter']} passed filters")

        return stats

    def process_unet_predictions(
        self,
        images_dir: str | Path,
        checkpoint_path: str | Path,
        progress: bool = True,
        image_pattern: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Run U-Net predictions and extract features.

        Args:
            images_dir: Directory containing source images
            checkpoint_path: Path to U-Net checkpoint
            progress: Show progress bar

        Returns:
            Summary statistics dictionary
        """
        from tb_unet.inference import TBUNetPredictor

        images_dir = Path(images_dir)

        # Find all image files
        if image_pattern:
            image_files = sorted(images_dir.glob(image_pattern))
        else:
            image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
        if limit:
            image_files = image_files[:limit]
        logger.info(f"Found {len(image_files)} images in {images_dir}")

        # Load U-Net predictor
        logger.info(f"Loading U-Net model from {checkpoint_path}")
        predictor = TBUNetPredictor(checkpoint_path)

        stats = {
            "total_images": len(image_files),
            "total_cells": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "no_cell_detected": 0,
            "errors": 0,
        }

        iterator = tqdm(image_files, desc="Processing images") if progress else image_files

        for image_path in iterator:
            try:
                # Load image
                image = tifffile.imread(image_path)

                # Run U-Net prediction with 3-class thresholds
                preds, probs = predictor.predict(image, return_probs=True)
                cell_prob = probs[1]
                non_cell_prob = probs[2]
                mask = (
                    (preds == 1)
                    & (cell_prob >= self.config.unet_cell_prob_min)
                    & (cell_prob >= non_cell_prob + self.config.unet_cell_margin)
                ).astype(np.uint16)

                if mask.any():
                    labeled, n_labels = ndimage.label(mask)
                    if n_labels > 1:
                        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
                        largest_label = int(np.argmax(sizes) + 1)
                        mask = (labeled == largest_label).astype(np.uint16)
                    else:
                        mask = (labeled > 0).astype(np.uint16)

                if mask.max() == 0:
                    stats["no_cell_detected"] += 1
                    continue

                # Get bounding box from mask
                rows, cols = np.where(mask > 0)
                if len(rows) == 0:
                    stats["no_cell_detected"] += 1
                    continue

                bbox = (rows.min(), cols.min(), rows.max(), cols.max())

                # Extract features
                result = self._process_single_cell(
                    mask=mask,
                    image=image,
                    label=1,  # U-Net outputs single label
                    bbox=bbox,
                    source_path=image_path,
                    metadata={"source": "unet"},
                )

                stats["total_cells"] += 1

                if result["passed_filter"]:
                    stats["passed_filter"] += 1
                    self.features_list.append(result["features"])
                    self.cell_ids.append(result["cell_id"])
                    self.metadata_list.append(result["metadata"])
                    self.all_profile_records.extend(result.get("profile_records", []))
                    self.all_puncta_records.extend(result.get("puncta_records", []))
                else:
                    stats["failed_filter"] += 1
                    self.filtered_cells.append({
                        "cell_id": result["cell_id"],
                        "reasons": result["failure_reasons"],
                    })

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                stats["errors"] += 1

        logger.info(f"Processed {stats['total_cells']} cells, {stats['passed_filter']} passed filters")

        return stats

    def process_session_output(
        self,
        session_dir: str | Path,
        *,
        metadata_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        annotations_dir: str | Path | None = None,
        images_dir: str | Path | None = None,
        progress: bool = True,
        included_only: bool = True,
        central_cells_only: bool = False,
        apply_annotations: bool = True,
        allowed_statuses: set[str] | None = None,
        reapply_filters: bool = False,
        limit: int | None = None,
        pixel_microns: float | None = None,
    ) -> dict[str, Any]:
        """
        Process an existing ``session_output`` directory.

        Selected cells are measured from the saved multi-label masks and source
        images, then exported using the same ``sc_morph_*`` schema as the U-Net
        workflow.
        """
        session_dir = Path(session_dir)
        metadata_dir = Path(metadata_dir) if metadata_dir else session_dir / "metadata"
        masks_dir = Path(masks_dir) if masks_dir else session_dir / "masks"

        if annotations_dir is None:
            annotations_candidate = session_dir / "annotations"
            resolved_annotations_dir = annotations_candidate if annotations_candidate.exists() else None
        else:
            resolved_annotations_dir = Path(annotations_dir)

        if images_dir is None:
            images_candidate = session_dir / "images"
            resolved_images_dir = images_candidate if images_candidate.exists() else None
        else:
            resolved_images_dir = Path(images_dir)

        meta_files = sorted(metadata_dir.glob("*_meta.json"))
        if limit is not None:
            meta_files = meta_files[:limit]
        logger.info(f"Found {len(meta_files)} metadata files in {metadata_dir}")

        stats = {
            "total_metadata_files": len(meta_files),
            "candidate_images": 0,
            "images_with_selected_cells": 0,
            "total_cells": 0,
            "selected_cells": 0,
            "exported_cells": 0,
            "failed_filter": 0,
            "skipped_status": 0,
            "skipped_cells": 0,
            "errors": 0,
            "annotation_files_loaded": 0,
        }

        iterator = tqdm(meta_files, desc="Processing session_output") if progress else meta_files

        for meta_path in iterator:
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {meta_path}: {e}")
                stats["errors"] += 1
                continue

            image_id = str(
                metadata.get("image_id")
                or metadata.get("id")
                or meta_path.stem.removesuffix("_meta")
            )
            status = normalize_review_status(metadata.get("status"))

            if allowed_statuses is None:
                if status == "discard":
                    stats["skipped_status"] += 1
                    continue
            elif status not in allowed_statuses:
                stats["skipped_status"] += 1
                continue

            stats["candidate_images"] += 1

            note_map: dict[str, str] = {}
            override_map: dict[str, str] = {}
            annotation_path: Path | None = None
            if apply_annotations and resolved_annotations_dir is not None:
                annotation_path = resolved_annotations_dir / f"{image_id}_annotations.json"
                if annotation_path.exists():
                    note_map, override_map = load_annotation_maps(annotation_path)
                    stats["annotation_files_loaded"] += 1

            raw_cells = metadata.get("cells", []) or []
            central_label = (
                resolve_central_label(metadata, raw_cells)
                if central_cells_only
                else None
            )
            selected_cells: list[dict[str, Any]] = []
            for cell in raw_cells:
                stats["total_cells"] += 1
                effective_cell = apply_cell_annotations(
                    cell,
                    note_map=note_map,
                    override_map=override_map,
                )

                should_select = True
                if central_cells_only:
                    label = effective_cell.get("label")
                    try:
                        label = int(label)
                    except (TypeError, ValueError):
                        label = None
                    should_select = (
                        central_label is not None
                        and label == central_label
                        and effective_cell.get("override") != "exclude"
                    )
                elif included_only and not effective_cell.get("included", False):
                    should_select = False

                if not should_select:
                    stats["skipped_cells"] += 1
                    continue
                selected_cells.append(effective_cell)

            if not selected_cells:
                continue

            try:
                mask_path = resolve_mask_path(masks_dir, image_id)
                image_path = resolve_image_path(
                    metadata,
                    image_id,
                    images_dir=resolved_images_dir,
                )
                mask = tifffile.imread(mask_path)
                image = tifffile.imread(image_path)
                channel_count = image.shape[0] if image.ndim == 3 else 1
                channel_names = resolve_channel_names(image_path, channel_count)
            except Exception as e:
                logger.error(f"Error resolving assets for {image_id}: {e}")
                stats["errors"] += 1
                continue

            stats["images_with_selected_cells"] += 1

            for cell_info in selected_cells:
                label = int(cell_info.get("label", 0))
                bbox_value = cell_info.get("bbox")
                if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
                    bbox = tuple(int(v) for v in bbox_value)
                else:
                    bbox = self._bbox_for_label(mask, label)
                    if bbox is None:
                        logger.warning(f"Label {label} not found in mask for {image_id}")
                        stats["errors"] += 1
                        continue

                if not np.any(mask == label):
                    logger.warning(f"Label {label} not present in saved mask for {image_id}")
                    stats["errors"] += 1
                    continue

                session_metadata = {
                    "session_image_id": image_id,
                    "session_status": status,
                    "session_mask_path": str(mask_path),
                    "session_annotation_path": str(annotation_path) if annotation_path else "",
                    "session_cell_id": str(cell_info.get("cell_id", f"{image_id}_{label}")),
                    "session_cell_included": bool(cell_info.get("included", False)),
                    "session_cell_override": cell_info.get("override"),
                    "session_cell_note": cell_info.get("note"),
                    "session_failure_reasons": json.dumps(cell_info.get("failure_reasons", [])),
                    "session_channel_names": "|".join(channel_names),
                }

                try:
                    effective_pixel_microns = (
                        float(pixel_microns)
                        if pixel_microns is not None
                        else float(metadata.get("pixel_microns", self.config.pixel_microns))
                    )
                    result = self._process_single_cell(
                        mask=mask,
                        image=image,
                        label=label,
                        bbox=bbox,
                        source_path=image_path,
                        metadata=session_metadata,
                        channel_names=channel_names,
                        skip_filters=not reapply_filters,
                        pixel_microns=effective_pixel_microns,
                    )
                except Exception as e:
                    logger.error(f"Error processing cell {image_id}:{label}: {e}")
                    stats["errors"] += 1
                    continue

                stats["selected_cells"] += 1

                if result["passed_filter"]:
                    stats["exported_cells"] += 1
                    self.features_list.append(result["features"])
                    self.cell_ids.append(result["cell_id"])
                    self.metadata_list.append(result["metadata"])
                    self.all_profile_records.extend(result.get("profile_records", []))
                    self.all_puncta_records.extend(result.get("puncta_records", []))
                else:
                    stats["failed_filter"] += 1
                    self.filtered_cells.append({
                        "cell_id": result["cell_id"],
                        "reasons": result["failure_reasons"],
                    })

        logger.info(
            "Processed session_output cells: "
            f"{stats['selected_cells']} selected, {stats['exported_cells']} exported"
        )
        return stats

    def process_single_image(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Process a single image (for CLI single-image mode).

        Args:
            image_path: Path to source image
            mask_path: Optional path to pre-computed mask
            checkpoint_path: Optional path to U-Net checkpoint (used if no mask)

        Returns:
            Feature dictionary if successful, None otherwise
        """
        image_path = Path(image_path)
        image = tifffile.imread(image_path)

        # Get mask
        if mask_path:
            mask = tifffile.imread(mask_path)
        elif checkpoint_path:
            from tb_unet.inference import TBUNetPredictor
            predictor = TBUNetPredictor(checkpoint_path)
            mask = predictor.predict_cell_mask(image)
        else:
            raise ValueError("Either mask_path or checkpoint_path must be provided")

        if mask.max() == 0:
            logger.warning(f"No cell detected in {image_path}")
            return None

        # Get bounding box
        rows, cols = np.where(mask > 0)
        bbox = (rows.min(), cols.min(), rows.max(), cols.max())

        # Extract features
        result = self._process_single_cell(
            mask=mask,
            image=image,
            label=1,
            bbox=bbox,
            source_path=image_path,
            metadata={"source": "single"},
        )

        if result["passed_filter"]:
            self.features_list.append(result["features"])
            self.cell_ids.append(result["cell_id"])
            self.metadata_list.append(result["metadata"])
            self.all_profile_records.extend(result.get("profile_records", []))
            self.all_puncta_records.extend(result.get("puncta_records", []))
            return result["features"]
        else:
            logger.warning(f"Cell filtered: {result['failure_reasons']}")
            return None

    def _process_single_cell(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        label: int,
        bbox: tuple[int, int, int, int],
        source_path: Path,
        metadata: dict[str, Any],
        channel_names: list[str] | None = None,
        skip_filters: bool = False,
        pixel_microns: float | None = None,
    ) -> dict[str, Any]:
        """
        Process a single cell and extract all features.

        Args:
            mask: Mask array (may contain multiple labels)
            image: Image array (C, H, W) or (H, W)
            label: Label to extract in mask
            bbox: Bounding box (min_row, min_col, max_row, max_col)
            source_path: Path to source image
            metadata: Additional metadata

        Returns:
            Dictionary with features, cell_id, metadata, passed_filter, failure_reasons
        """
        # Generate cell ID
        cell_id = generate_cell_id(source_path, label, bbox)
        effective_channel_names = channel_names or self.config.channel_names
        effective_pixel_microns = (
            float(pixel_microns) if pixel_microns is not None else self.config.pixel_microns
        )

        # Extract regionprops
        rp_features = extract_regionprops(
            mask,
            intensity_image=image if image.ndim == 2 else image[0],
            pixel_microns=effective_pixel_microns,
            label=label,
        )

        if skip_filters:
            passed = True
            failure_reasons: list[str] = []
        else:
            # Apply morphology filters first
            filter_params = self.config.filter_params
            passed, failure_reasons = apply_morphology_filters(
                rp_features,
                area_min=filter_params.area_min,
                area_max=filter_params.area_max,
                eccentricity_min=filter_params.eccentricity_min,
                eccentricity_max=filter_params.eccentricity_max,
                solidity_min=filter_params.solidity_min,
                solidity_max=filter_params.solidity_max,
                aspect_ratio_min=filter_params.aspect_ratio_min,
                aspect_ratio_max=filter_params.aspect_ratio_max,
            )

            # Check edge exclusion
            if filter_params.exclude_edge:
                h, w = mask.shape
                edge_dist = filter_params.edge_distance
                if (bbox[0] < edge_dist or bbox[1] < edge_dist or
                        bbox[2] > h - edge_dist or bbox[3] > w - edge_dist):
                    passed = False
                    failure_reasons.append("touching_edge")

        features = dict(rp_features)

        # Sidecar accumulators
        profile_records: list[dict[str, Any]] = []
        puncta_records: list[dict[str, Any]] = []

        # Only extract additional features if passed filter
        midline = None
        if passed:
            # Centerline features
            if self.config.extract_centerline:
                cl_features = extract_centerline_features(
                    mask,
                    pixel_microns=effective_pixel_microns,
                    label=label,
                )
                features.update(cl_features)

                # Also compute raw midline array for intensity/fluorescence
                midline = compute_midline(
                    mask,
                    pixel_microns=effective_pixel_microns,
                    label=label,
                )

            # Intensity features
            if self.config.extract_intensity and image.ndim == 3:
                int_features = extract_intensity_features(
                    mask,
                    image,
                    channel_names=effective_channel_names,
                    midline=midline,
                    label=label,
                )
                features.update(int_features)

            # Fluorescence features
            if self.config.extract_fluorescence and image.ndim == 3:
                fluor_feats, prof_recs, punc_recs = self._extract_fluorescence(
                    mask, image, label, midline, cell_id, effective_channel_names,
                )
                features.update(fluor_feats)
                profile_records.extend(prof_recs)
                puncta_records.extend(punc_recs)

        # Build metadata
        filename_meta = parse_filename_metadata(source_path.name)
        cell_metadata = {
            "cell_id": cell_id,
            "original_cell_id": f"{source_path.stem}_{label}",
            "source_path": str(source_path),
            "source_filename": source_path.name,
            "mask_label": label,
            "drug": "UNK",
            "GroupLabels_Strict": "UNK",
            "pixel_microns": effective_pixel_microns,
            "bbox_min_row": bbox[0],
            "bbox_min_col": bbox[1],
            "bbox_max_row": bbox[2],
            "bbox_max_col": bbox[3],
            **filename_meta,
        }
        for key, value in metadata.items():
            if key not in cell_metadata:
                cell_metadata[key] = value

        # Add to lookup table
        self.lookup_table.add_cell(source_path, label, bbox)

        # Save cropped binary mask if enabled and cell passed filters
        if passed and self.masks_dir is not None:
            self._save_cell_mask(mask, label, bbox, cell_id, cell_metadata)

        return {
            "features": features,
            "cell_id": cell_id,
            "metadata": cell_metadata,
            "passed_filter": passed,
            "failure_reasons": failure_reasons,
            "profile_records": profile_records,
            "puncta_records": puncta_records,
        }

    def _save_cell_mask(
        self,
        mask: np.ndarray,
        label: int,
        bbox: tuple[int, int, int, int],
        cell_id: str,
        cell_metadata: dict[str, Any],
    ) -> None:
        """Save a full-size binary mask for a single cell to disk.

        The output mask has the same dimensions as the input so it stays
        spatially registered with the source image.
        """
        cell_mask = (mask == label).astype(np.uint8) * 255

        mask_filename = f"{cell_id}.tif"
        tifffile.imwrite(self.masks_dir / mask_filename, cell_mask)

        self.mask_manifest.append({
            "cell_id": cell_id,
            "mask_filename": mask_filename,
            "source_image": cell_metadata.get("source_path", ""),
            "plate": cell_metadata.get("plate", ""),
            "scene": cell_metadata.get("scene", ""),
            "position": cell_metadata.get("position", ""),
            "drug": cell_metadata.get("drug", "UNK"),
            "GroupLabels_Strict": cell_metadata.get("GroupLabels_Strict", "UNK"),
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        })

    def _extract_fluorescence(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        label: int,
        midline: np.ndarray | None,
        cell_id: str,
        channel_names: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract fluorescence features for a single cell.

        Returns:
            Tuple of (scalar_features, profile_records, puncta_records).
        """
        cfg = self.config.fluor_config
        features: dict[str, Any] = {}
        profile_records: list[dict[str, Any]] = []
        puncta_records: list[dict[str, Any]] = []

        binary_mask = (mask == label).astype(np.uint8) if label else (mask > 0).astype(np.uint8)

        shell, core = compute_shell_core(binary_mask, cfg.shell_erosion_radius)
        bg_ring = compute_background_ring(
            binary_mask, dilation_r=cfg.bg_ring_dilation, erosion_r=cfg.bg_ring_erosion,
        )

        for ch_idx, ch in enumerate(channel_names):
            if ch == "Phase":
                continue
            if ch_idx >= image.shape[0]:
                continue

            ch_img = image[ch_idx].astype(np.float64)
            prefix_ch = ch

            # Background
            bg = background_stats(ch_img, bg_ring)
            features[f"{prefix_ch}_bg_mean"] = bg["bg_mean"]
            features[f"{prefix_ch}_bg_median"] = bg["bg_median"]
            features[f"{prefix_ch}_bg_std"] = bg["bg_std"]

            bg_val = bg["bg_mean"] if not np.isnan(bg["bg_mean"]) else 0.0
            cell_pixels = ch_img[binary_mask.astype(bool)]
            features[f"{prefix_ch}_mean_bgsub"] = (
                float(np.mean(cell_pixels)) - bg_val if cell_pixels.size > 0 else np.nan
            )

            # Shell/core
            sc = shell_core_stats(ch_img, shell, core)
            features[f"{prefix_ch}_shell_mean"] = sc["shell_mean"]
            features[f"{prefix_ch}_core_mean"] = sc["core_mean"]
            features[f"{prefix_ch}_shell_core_ratio"] = sc["shell_core_ratio"]

            # Midline profile
            profile = compute_midline_profile(
                midline, ch_img,
                width=cfg.strip_width, mode=cfg.strip_mode,
                n_points=cfg.n_profile_points,
            )
            if profile is not None:
                profile, _ = reorient_profile(profile)

            scalars = profile_scalar_summaries(profile)
            for k, v in scalars.items():
                features[f"{prefix_ch}_{k}"] = v

            if profile is not None:
                profile_records.append({
                    "cell_id": cell_id,
                    "channel": ch,
                    "profile_kind": f"midline_strip_{cfg.strip_mode}",
                    "n_points": cfg.n_profile_points,
                    "profile": profile.tolist(),
                })

            # HADA
            if ch == "AF405" and profile is not None:
                bg_sub_profile = profile - bg_val
                hada_feats = hada_polar_features(
                    bg_sub_profile,
                    pole_window=cfg.pole_window,
                    center_window=cfg.center_window,
                )
                for k, v in hada_feats.items():
                    features[f"AF405_{k}"] = v
                state_code, _ = hada_state_classify(
                    hada_feats["polar_asym"], hada_feats["center_enrichment"],
                )
                features["AF405_state_code"] = state_code

            # Bod493 puncta
            if ch == "Bod493":
                core_mask = core if cfg.puncta_use_core else None
                puncta = detect_puncta(
                    ch_img, binary_mask, core_mask=core_mask,
                    min_distance=cfg.puncta_min_distance,
                    threshold_rel=cfg.puncta_threshold_rel,
                    gaussian_sigma=cfg.puncta_gaussian_sigma,
                    log_sigma=cfg.puncta_log_sigma,
                    threshold_abs=cfg.puncta_threshold_abs,
                    exclude_border=cfg.puncta_exclude_border,
                    bg_value=bg_val,
                )
                cell_total = float(np.sum(cell_pixels)) if cell_pixels.size > 0 else 0.0
                punc_summary = puncta_scalar_summary(puncta, cell_total)
                for k, v in punc_summary.items():
                    features[f"Bod493_{k}"] = v

                if puncta.size > 0 and midline is not None:
                    axrad = puncta_axial_radial(puncta, midline)
                    for j in range(len(puncta)):
                        puncta_records.append({
                            "cell_id": cell_id,
                            "channel": "Bod493",
                            "punctum_id": j,
                            "y_px": float(puncta[j, 0]),
                            "x_px": float(puncta[j, 1]),
                            "sigma_px": float(abs(puncta[j, 3])),
                            "peak_intensity": float(puncta[j, 2]),
                            "axial_s": float(axrad[j, 0]) if j < len(axrad) else np.nan,
                            "radial_r": float(axrad[j, 1]) if j < len(axrad) else np.nan,
                        })

        return features, profile_records, puncta_records

    @staticmethod
    def _bbox_for_label(
        mask: np.ndarray,
        label: int,
    ) -> tuple[int, int, int, int] | None:
        """Compute a skimage-style bbox for a label when metadata is missing one."""
        rows, cols = np.where(mask == label)
        if len(rows) == 0 or len(cols) == 0:
            return None
        return (
            int(rows.min()),
            int(cols.min()),
            int(rows.max()) + 1,
            int(cols.max()) + 1,
        )

    def export(self, prefix: str = "sc_morph") -> dict[str, Path]:
        """
        Export all results to files.

        Args:
            prefix: Prefix for output files

        Returns:
            Dictionary of output type -> path
        """
        if self.config.drop_nan_cells:
            self._drop_cells_with_nonfinite_features()

        outputs = {}

        # Export deciphaer format CSVs
        if self.config.output_csv:
            data_path, meta_path, mapping_path = export_deciphaer_format(
                self.features_list,
                self.cell_ids,
                self.metadata_list,
                self.output_dir,
                prefix=prefix,
            )
            outputs["data_csv"] = data_path
            outputs["meta_csv"] = meta_path
            outputs["mapping_csv"] = mapping_path

        # Export Parquet
        if self.config.output_parquet:
            parquet_path = export_features_polars(
                self.features_list,
                self.cell_ids,
                self.output_dir / f"{prefix}_features.parquet",
            )
            outputs["features_parquet"] = parquet_path

        # Save lookup table
        lookup_path = self.output_dir / "cell_lookup.parquet"
        self.lookup_table.save(lookup_path)
        outputs["lookup_parquet"] = lookup_path

        # Save mask manifest
        if self.mask_manifest:
            import polars as pl
            manifest_df = pl.DataFrame(self.mask_manifest)
            manifest_parquet = self.output_dir / f"{prefix}_mask_manifest.parquet"
            manifest_csv = self.output_dir / f"{prefix}_mask_manifest.csv"
            manifest_df.write_parquet(manifest_parquet)
            manifest_df.write_csv(manifest_csv)
            outputs["mask_manifest_parquet"] = manifest_parquet
            outputs["mask_manifest_csv"] = manifest_csv
            logger.info(f"Saved mask manifest ({len(self.mask_manifest)} entries) to {manifest_parquet}")

        # Save filtered cells log
        if self.filtered_cells:
            import json
            filtered_path = self.output_dir / "filtered_cells.json"
            with open(filtered_path, "w") as f:
                json.dump(self.filtered_cells, f, indent=2)
            outputs["filtered_log"] = filtered_path

        # Write sidecar parquet files
        if self.all_profile_records:
            profiles_path = self.output_dir / f"{prefix}_profiles.parquet"
            write_profiles_parquet(self.all_profile_records, profiles_path)
            outputs["profiles_parquet"] = profiles_path
        if self.all_puncta_records:
            puncta_path = self.output_dir / f"{prefix}_puncta.parquet"
            write_puncta_parquet(self.all_puncta_records, puncta_path)
            outputs["puncta_parquet"] = puncta_path

        logger.info(f"Exported {len(self.cell_ids)} cells to {self.output_dir}")

        return outputs

    def _drop_cells_with_nonfinite_features(self) -> None:
        """Drop any cells that contain NaN/inf values in their feature vector.

        This prevents NaNs from propagating into downstream preprocessing.
        """
        if not self.features_list:
            return

        import pandas as pd

        feature_df = pd.DataFrame(self.features_list, index=self.cell_ids)
        # Coerce any unexpected non-numeric values to NaN so they get filtered.
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

        is_finite = np.isfinite(feature_df.to_numpy(dtype=float))
        bad_mask = ~is_finite.all(axis=1)
        if not bad_mask.any():
            return

        bad_cell_ids = feature_df.index[bad_mask].tolist()
        bad_set = set(bad_cell_ids)

        logger.warning(
            f"Dropping {len(bad_cell_ids)} cells with non-finite feature values (NaN/inf)"
        )

        # Log as filtered for provenance/debugging.
        self.filtered_cells.extend(
            {"cell_id": cell_id, "reasons": ["nan_in_features"]}
            for cell_id in bad_cell_ids
        )

        # Filter core outputs (features/meta/cell_ids)
        keep_mask = [cell_id not in bad_set for cell_id in self.cell_ids]
        self.features_list = [
            features for features, keep in zip(self.features_list, keep_mask) if keep
        ]
        self.metadata_list = [
            meta for meta, keep in zip(self.metadata_list, keep_mask) if keep
        ]
        self.cell_ids = [cell_id for cell_id, keep in zip(self.cell_ids, keep_mask) if keep]

        kept_ids = set(self.cell_ids)

        # Filter sidecar records
        if self.all_profile_records:
            self.all_profile_records = [
                rec for rec in self.all_profile_records if rec.get("cell_id") in kept_ids
            ]
        if self.all_puncta_records:
            self.all_puncta_records = [
                rec for rec in self.all_puncta_records if rec.get("cell_id") in kept_ids
            ]

        # Filter mask manifest (and delete any orphaned mask TIFFs)
        if self.mask_manifest:
            self.mask_manifest = [
                rec for rec in self.mask_manifest if rec.get("cell_id") in kept_ids
            ]

        if self.masks_dir is not None:
            n_deleted = 0
            for cell_id in bad_cell_ids:
                mask_path = self.masks_dir / f"{cell_id}.tif"
                if mask_path.exists():
                    mask_path.unlink()
                    n_deleted += 1
            if n_deleted:
                logger.info(f"Deleted {n_deleted} mask files for NaN-filtered cells")

    def get_summary(self) -> dict[str, Any]:
        """Get summary of processed data."""
        return {
            "n_cells": len(self.cell_ids),
            "n_features": len(self.features_list[0]) if self.features_list else 0,
            "n_filtered": len(self.filtered_cells),
            "output_dir": str(self.output_dir),
        }

    def reset(self) -> None:
        """Reset pipeline state for a new run."""
        self.features_list = []
        self.cell_ids = []
        self.metadata_list = []
        self.filtered_cells = []
        self.mask_manifest = []
        self.lookup_table = CellLookupTable()
        self.all_profile_records = []
        self.all_puncta_records = []
