# aldridge lab image segmentation - public release

this repo's focus is to turn phase-contrast microscopy TIFFs into per-cell morphology and fluorescence features. the pipeline uses MOMIA2 for segmentation, with optional rule-based filtering and U-Net classification, and writes tables in a format that allows for simple analysis.

the pipeline code is included under `src/`. download the U-Net checkpoint separately if desired.

## quickstart

install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. the project requires Python 3.11 or newer; uv can install Python if needed.

```bash
git clone https://github.com/aldridgelab/image-segmentation-release.git
cd image-segmentation-release
uv sync --locked

mkdir -p configs/local
cp configs/example_pipeline.yaml configs/local/run.yaml
```

edit `configs/local/run.yaml`. update these settings in the copied file; keep the other settings to start with:

```yaml
run:
  output_dir: /absolute/path/to/results
inputs:
  image_dir: /absolute/path/to/images
  pattern: "*Phase*.tif"  # use "*Phase*.tiff" for .tiff files
  pixel_microns: 0.10317  # replace with your image sampling in µm/pixel
filter:
  config_path: ../rod_filter.json
```

paths are relative to the YAML file.

run locally:

```bash
uv run pipeline.py --config configs/local/run.yaml
```

the example enables the rod-shape filter, leaves U-Net off, and keeps intermediate outputs. check the masks on a few images before processing a whole dataset. adjust `momia.settings` and the filter rules for your images; set `filter.enabled: false` to skip the post-MOMIA filter. our university provides very generous usage of GPUs. if you are in a similar scenario, turning the filter on may (counter-intuitively) slow down the pipeline. 

on Linux, uv installs PyTorch from the CUDA 12.8 index; on macOS, it uses PyPI. U-Net inference automatically uses CUDA when available, otherwise CPU. a compatible NVIDIA GPU and driver are needed for CUDA acceleration. this has only been tested using our university's slurm setup, there may be some extra environment tweaking based on the resources you have at your disposal. 

## use the U-Net

download the `.pt` asset from the **[official U-Net checkpoint release](https://github.com/aldridgelab/image-segmentation-release/releases/tag/unet-checkpoint-v1)** (~800mb). then update the `unet` section in your YAML:

```yaml
unet:
  enabled: true
  model_path: /absolute/path/to/unet_hyperparameter_sweep_ch0_att0_bc96_d4_lr0.0005_fg2_focaldice_attributed.pt
```

the pipeline crops MOMIA detections and classifies them with the checkpoint. it does not replace the initial full-image segmentation. this u-net was trained on 10,000 manually annotated cells from our own work. the model has been independently tested on m. abscessus, and clinical isolates of m. tuberculosis. (unofficially), the model did well with these external tests. the optional to create your own u-net is always there, and once you've accumulated a decent repository of cell/no-cell examples, it's definitely worth considering. 

## images and calibration

for the example setup, use single-channel Phase TIFFs. put fluorescence channels alongside them with matching names, such as `img_001_Phase.tif` and `img_001_Fluorescence.tif`. feature extraction auto-discovers sibling channels for `.tif` inputs. for `.tiff` inputs, set `export.extract_channels` explicitly, for example `[Phase, Fluorescence]`.

set `inputs.pixel_microns` to the actual sampling, ours is `0.10317`. the example's `spatial_calibration.reference_pixel_microns` is `0.10317`; supported pixel-unit settings are scaled from that reference. do not scale them manually. images and masks are not resampled.

use a **new output directory** after changing calibration or processing settings to avoid reusing old results. see the [calibration guide](docs/PIPELINE.md#spatial-calibration) for details.

## run on Slurm

install uv and make the checkout, config, inputs, and checkpoint accessible on the compute nodes. set the `slurm` resources for your cluster, then preview or submit:

```bash
# preview the job dependencies without submitting
uv run pipeline.py --config configs/local/run.yaml --plan

# submit jobs
uv run pipeline.py --config configs/local/run.yaml --submit

# check progress
uv run pipeline.py --config configs/local/run.yaml --status

# a more elegant progress dashboard
sh scripts/watch.sh configs/local/run.yaml 
```


rerun `--submit` to resume. completed stages are skipped, running jobs are preserved, and old pending jobs are cancelled and replaced. failed images may be added to `skip.txt` so successful batches can continue; check it and the run summaries for omissions.

## outputs

under `run.output_dir`, the main files are:

| file | contents |
| --- | --- |
| `final/sc_morph_data.csv` | feature matrix: features in rows, cells in columns |
| `final/sc_morph_meta.csv` | cell IDs and group labels; unmatched labels are `UNK` |
| `final/sc_morph_mapping.csv` | source paths, mask labels, crop paths, and cell/run identifiers |
| `final/sc_morph_diagnostics.csv` | inclusion flags and U-Net scores, when available |
| `final/extract_summary.json` | counts, channels, and output paths |
| `run_metadata.json` | configuration, software versions/revision, and run provenance |

`export.prefix` changes the `sc_morph` filename prefix. the final folder also contains profile and cell-lookup Parquet files, plus puncta results when detected.

with `run.keep_intermediate: true` (as in the example), batch links, MOMIA masks/tables, and U-Net crops are retained. with it set to `false`, those directories are removed after final export. U-Net calls remain in `unet/unet_classifications.csv` when enabled.

## repository layout

- `pipeline.py`: CLI entrypoint.
- `configs/`: example YAML and rod-filter rules.
- `src/`: pipeline, MOMIA2, U-Net, and feature-extraction code.
- `scripts/`: Slurm stage wrapper and helpers.
- `docs/PIPELINE.md`: stage commands and detailed configuration notes.
