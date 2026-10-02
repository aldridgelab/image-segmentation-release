# Pipeline Guide

## Pipeline Overview

```text
Input TIFFs
  -> symlink batch folders
  -> MOMIA2 loose segmentation per batch
  -> compile masks, metadata, run summaries, and cell tables
  -> optional JSON post-filter
  -> optional calibrated crop generation (minimum target size)
  -> optional TB U-Net classification
  -> final deciphaer-style sc_morph outputs
```

## Stage Contract

The top-level command is:

```bash
uv run pipeline.py --config config.yaml
```

All dependencies — the MOMIA2 segmentation core, the batch runner, the TB U-Net, and the deciphaer feature extractors — are vendored under `src/` and resolved through `pyproject.toml`. There is no external repo dependency:

```bash
uv sync
```

For Slurm, the runner first creates batch symlinks locally and then submits one MOMIA job per batch. The compile job depends on all MOMIA jobs. The filter job depends on compile when enabled. The crop job depends on filter or compile. U-Net shard jobs depend on crop, and final export depends on all U-Net shards.

Each stage can also be run directly:

```bash
uv run pipeline.py --config config.yaml --stage setup
uv run pipeline.py --config config.yaml --stage momia --batch-name batch1
uv run pipeline.py --config config.yaml --stage compile
uv run pipeline.py --config config.yaml --stage filter
uv run pipeline.py --config config.yaml --stage crop
uv run pipeline.py --config config.yaml --stage unet --shard 0/16
uv run pipeline.py --config config.yaml --stage final
```

## Config Notes

`momia.settings` is resolved as described under [Spatial calibration](#spatial-calibration), then passed to the vendored `momia_seg.batch` runner. The orchestrator fills in per-batch `io.input_dir`, `io.output_dir`, channel names, pixel size, and runtime controls. `inputs.pixel_microns` is authoritative; an embedded `momia.settings.image.pixel_microns` does not override it.

`filter.config_path` is a JSON object. The preferred shape is:

```json
{
  "rules": {
    "area_px": {"min": 100, "max": 500},
    "touching_edge": false
  }
}
```

The filter also accepts shorthand keys such as `area_min`, `area_max`, `length_min`, and `width_max`.

`unet.save.masks` defines requested crop stack channels. `Mask` is always appended if omitted and is written last because the U-Net loader treats the final plane as the ROI mask.

`unet.save.include_available_channels` defaults to `true`. When enabled, the crop and composite writers discover sibling TIFF channels next to the Phase source image using the `<source_base>_<channel>.tif[f]` convention. For example, a cell from `img_s01_m01_Phase.tiff` will include `img_s01_m01_HADA.tiff` automatically if it exists, even when `unet.save.masks` is only `[Phase, Mask]`. The stored channel order is Phase, requested/auto-discovered fluorescence channels, then Mask.

Crops are named with deterministic `sc_<hash>` IDs derived from the segmentation run hash, resolved source image path, mask label, and bbox. The crop manifest includes `Cell_ID`, `cell_hash` (`sha256:<hash>`), `segmentation_run_hash`, `source_image`, `source_path`, channel labels, and JSON channel metadata. Each crop TIFF also stores ImageJ labels plus a JSON `Info` payload with the same cell/source/channel provenance.

## Spatial calibration

`inputs.pixel_microns` describes the actual input sampling in µm/pixel. It must be finite and positive. The pipeline assumes square pixels and one calibration for the entire run. Split mixed-calibration inputs into separate runs; no TIFF calibration is inferred automatically.

To transfer a validated parameter set, set `spatial_calibration.reference_pixel_microns` to the sampling at which its pixel-unit settings were chosen. The example configuration uses the repository reference, `0.10317`. This value describes the parameter set; it is not inferred from checkpoint training metadata. Verify the appropriate reference for a different parameter set or model.

When a reference is supplied, conversion uses:

```text
s = reference_pixel_microns / inputs.pixel_microns
input-pixel length = reference-pixel length × s
input-pixel area   = reference-pixel area × s²
physical length   = reference-pixel length × reference_pixel_microns
physical area     = reference-pixel area × reference_pixel_microns²
```

Omitting the block or setting the reference to `null` disables scaling. Existing configurations retain their spatial behavior. The example opts in but has identical effective settings at its original sampling. Keep all supported pixel-unit values at the same reference calibration; do not combine automatic scaling with manually converted values. To specify a desired physical crop width, divide that width in µm by the reference sampling and set the resulting integer `unet.crop_size`.

### Converted settings

| Setting | Units before conversion | Rule |
| --- | --- | --- |
| `unet.crop_size`, `unet.min_bbox_pad` | reference px | × s, integer |
| `momia.settings.image.max_drift` | reference px | × s, float |
| `momia.settings.segmentation.window_size` | reference px | × s, odd integer ≥ 3 |
| `momia.settings.filtering.edge_distance` | reference px | × s, integer |
| `momia.settings.centrality.center_window_px` | reference px | × s, integer |
| `momia.settings.segmentation.min_particle_size`, `min_hole_size` | reference px² | × s², integer |
| `momia.settings.filtering.area_min`, `area_max` | reference px² | × s², float; null stays null |
| Post-filter `area_px` min/max, including `area` and `area_min`/`area_max` aliases | reference px² | × s², float; null stays null |

MOMIA conversion applies **after merging defaults and overrides**, so omitted settings scale too. Integer conversion uses nearest rounding with ties upward. Crop size is at least 1. Other ordinary integer settings may round to zero. Threshold windows use the nearest odd integer, with ties upward and a minimum of 3. Floating-point area bounds retain fractional values. Pixel-area equality rules are rejected when scaling would be required; use min/max bounds instead.

Already-physical length/width thresholds, `area_um2` post-filter rules, dimensionless shape rules, classifier probabilities, preview sizes, and custom filter columns are **not** scaled. The standalone `momia_seg` interface is unchanged; this option belongs to the top-level pipeline. Fluorescence feature settings and internal pixel-scale operations outside the table are not converted. This is not a claim of complete scale invariance.

### Crop geometry and model input

At reference sampling `0.10317`, a 96-pixel crop target covers `9.90432 µm`. At input sampling `0.0645`, it resolves to 154 pixels (`9.933 µm`). The small difference is integer rounding. Using `0.103` as the reference instead gives 153 pixels: use the actual calibration rather than a rounded label. The reference padding of 4 pixels resolves to 6 pixels at `0.0645`.

These are minimum crop dimensions, not a forced square size. Each axis can expand to accommodate the bounding box plus twice the resolved padding. Edge crops retain the existing reflect/constant padding behavior. Crop manifests record actual output dimensions.

Images, intensities, coordinates, and label masks remain at native sampling. This change does not resize cells to a model's training scale. The predictor separately pads input arrays to its required multiple and removes that padding from its output. Neither crop-field matching nor that architectural padding compensates for optical resolution, focus, contrast, or domain shift. Evaluate checkpoint suitability independently.

### Provenance and reruns

- `momia/batches/batch*/effective_momia_settings.json` records resolved MOMIA values.
- `filter/filter_summary.json` includes the resolved rules and scale factor.
- `unet/crop_settings*.json` records requested and resolved crop settings and physical sizes. Sharded runs write a separate file per crop shard.
- `run_metadata.json` includes requested configuration and `config.resolved_spatial_settings` for enabled MOMIA/filter stages and crop geometry. Resolved settings also enter the run hash.

Archive the YAML and filter JSON with these records. Set `run.keep_intermediate: true` when retaining masks and rejected candidates for validation. **Choose a new output directory after changing calibration, reference values, or spatial parameters.** Resume/skip-existing logic is not a configuration-aware cache and can reuse stale results. Changing only the crop stage cannot recover candidates rejected by earlier segmentation/filtering.

### Verification

```bash
python -m pytest -q tests/test_spatial_calibration.py tests/test_config_and_plan.py
```

The regression tests check conversion, legacy behavior, filter decisions, native crop coverage, label preservation, and shared local/Slurm stage behavior using synthetic arrays. They do not establish biological accuracy or checkpoint performance on a new image collection. Review annotated full fields as well as accepted and rejected crops, and report stage-specific counts, precision, and recall before claiming improved transfer.

## Output Strategy

The compiled MOMIA mask and metadata folders use symlinks to avoid duplicating large batch outputs. U-Net composites are disabled by default because the expected high-throughput path usually only needs single-cell crops with a configured minimum size.

Final `sc_morph_meta.csv` and `sc_morph_mapping.csv` preserve crop provenance when available, including `cell_hash`, `source_image`, `source_path`, `crop_path`, and `segmentation_run_hash`.
