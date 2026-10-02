# aldridge-image-segmentation-release

A data-free release-testing snapshot of `deciphaer-image-segmentation`.
This repository contains code, user documentation, generic configuration, and
synthetic tests. Input images, measurement tables, populated padmaps, model
checkpoints, manuscript files, and internal planning documents are not bundled.
Supply inputs and checkpoints from external paths and keep generated results
outside the repository.

A self-contained, config-driven pipeline that takes raw Phase microscopy TIFFs through MOMIA2 segmentation, optional rod-shape filtering, optional TB U-Net cell classification, and deciphaer-format feature extraction — locally or on Slurm.

Everything the pipeline needs (the MOMIA2 segmentation core, the batch runner, the TB U-Net, the deciphaer feature extractors, and the fluorescence analysis modules) is vendored under `src/`. The only external runtime requirements are the Python dependencies declared in `pyproject.toml`.

## What It Does

Replaces previous Ilastik-based workflows. The pipeline runs MOMIA2 as an object detector on full-size Phase images, optionally filters detections to rod-shaped cells, optionally classifies single-cell crops with a TB U-Net, and emits deciphaer-style per-cell feature tables.

It can auto-restart failed Slurm batches and post webhook notifications on completion or failure.

## Layout

```
aldridge-image-segmentation-release/
├── pipeline.py                          # CLI entrypoint
├── pyproject.toml
├── configs/
│   ├── example_pipeline.yaml            # template run config
│   └── rod_filter.json                  # rod-shape filter rules
├── scripts/                             # status watcher + helper scripts
├── tests/
├── docs/
└── src/
    ├── deciphaer_image_segmentation/    # orchestration: batching, runner, slurm, CLI
    ├── momia_seg/                       # batch MOMIA2 runner
    ├── momia2/                          # MOMIA2 segmentation core
    ├── tb_unet/                         # TB U-Net model + inference
    ├── deciphaer_processing/            # feature extractors, cell hashing, padmap registry
    └── fluorescence/                    # fluorescence profiles, puncta, HADA features
```

## Quickstart

```bash
# Environment setup (uv recommended; pip install -e . also works)
uv sync --extra dev

# Inspect the Slurm DAG without submitting jobs
uv run pipeline.py --config configs/example_pipeline.yaml --plan

# Run from the YAML
uv run pipeline.py --config configs/example_pipeline.yaml

# View the status of a run
uv run pipeline.py --config configs/example_pipeline.yaml --status
```

## Configuration

Main example: `configs/example_pipeline.yaml`.

Important keys:

- `run.output_dir`: root for all batch, MOMIA, filter, U-Net, final, and metadata outputs.
- `inputs.image_dir`, `inputs.pattern`, `inputs.channels`: image discovery and channel contract.
- `batching.n_batches`: number of symlink folders and MOMIA Slurm jobs.
- `inputs.pixel_microns`: actual image sampling in µm/pixel; use one calibration per run.
- `spatial_calibration.reference_pixel_microns`: optional reference sampling for automatic conversion of supported pixel-unit settings. The example enables it; existing configurations without it retain legacy pixel behavior.
- `momia.settings`: MOMIA2 segmentation/filtering settings (resolved at the input calibration when scaling is enabled).
- `filter.enabled`, `filter.config_path`: optional post-MOMIA filtering, for example rod-shape enforcement from `configs/rod_filter.json`.
- `unet.enabled`, `unet.model_path`, `unet.padmap`: optional TB U-Net classification and padmap annotation. `unet.model_path` points at your trained TB U-Net checkpoint (`.pt`).
- `unet.crop_size`: minimum crop width/height, default `96` pixels (reference pixels when calibration scaling is enabled). Larger bounding boxes plus padding expand the crop.
- `unet.save.masks`: requested crop stack channels, with `Mask` written last for U-Net ROI use.
- `unet.save.include_available_channels`: when `true` (default), crops and composites also include sibling fluorescence TIFFs named like the source Phase TIFF, for example `img_s01_m01_HADA.tiff` next to `img_s01_m01_Phase.tiff`.
- `slurm.submit`: submit instead of local execution.

### Different microscope pixel sizes

Keep the parameters at their validated reference values and set the actual image sampling:

```yaml
inputs:
  image_dir: /path/to/images
  pixel_microns: 0.0645
spatial_calibration:
  reference_pixel_microns: 0.10317
unet:
  crop_size: 96
  min_bbox_pad: 4
```

This resolves the crop target to **154 pixels** (about **9.93 µm**, versus 9.90 µm at the reference sampling). It also scales supported MOMIA lengths, pixel areas, and post-filter area bounds. Do not manually scale these values as well. Omit `spatial_calibration` or set its reference to `null` to use legacy input-pixel values.

**Use a new `run.output_dir` after changing calibration or spatial settings.** Existing masks, crops, and classifications are not automatically invalidated by configuration changes.

This preserves approximate physical crop coverage, not the pixel size of objects presented to the model. Images and masks are **not resampled**. Pixel sampling is not optical resolution, and calibration alone does not establish checkpoint transfer or improved recall. Review both missed objects in full fields and accepted/rejected candidates.

See [spatial calibration details](docs/PIPELINE.md#spatial-calibration) for supported settings, rounding, provenance, and limitations.

## Common Workflows

Filter-only MOMIA run:

```yaml
filter:
  enabled: true
unet:
  enabled: false
```

MOMIA straight into U-Net, skipping the intermediate filter:

```yaml
filter:
  enabled: false
unet:
  enabled: true
  model_path: /path/to/tb_unet/best.pt
  save:
    crops: true
    composites: false
    include_available_channels: true
    masks: [Phase, Mask]
```

## Outputs

Outputs land under `run.output_dir`:

- `batches/batch*/`: symlinks to source TIFFs.
- `momia/batches/batch*/`: raw per-batch MOMIA2 outputs.
- `momia/compiled/cell_measurements.csv`: compiled MOMIA cell table.
- `momia/compiled/masks/` and `momia/compiled/metadata/`: linked compiled masks and metadata.
- `filter/cell_measurements.csv`: optional post-filtered table with `filter_passed` and `filter_reasons`.
- `unet/cell_crops_<size>/`: optional hashed single-cell crop TIFFs plus `crops_manifest.csv`. Crop TIFFs store channels as `Phase`, any requested or auto-discovered fluorescence channels, then `Mask`; ImageJ metadata includes channel labels, cell hash, source image, and segmentation run hash.
- `unet/unet_classifications.csv`: optional TB U-Net calls.
- `final/sc_morph_data.csv`, `final/sc_morph_meta.csv`, `final/sc_morph_mapping.csv`: deciphaer-style final outputs. Metadata/mapping files include `cell_hash`, `source_image`, and `segmentation_run_hash`.
- `unet/crop_settings*.json`: requested and resolved crop geometry, reference/input sampling, and physical dimensions (one file per crop shard when sharded).
- `run_metadata.json`: config snapshot (including `resolved_spatial_settings`), inputs, package versions, seed, segmentation run hash, and output artifacts.
