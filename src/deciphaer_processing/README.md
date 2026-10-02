# Feature extraction support

`deciphaer_processing` contains the vendored feature extraction, cell lookup,
padmap, and export utilities used by the image segmentation pipeline.

Use the supported repository-level workflow:

```bash
uv run pipeline.py --config /absolute/path/to/run.yaml
```

See the [pipeline documentation](../../docs/PIPELINE.md) and
[example configuration](../../configs/example_pipeline.yaml) for configuration
and output details.

Images, segmentation outputs, populated padmaps, and trained checkpoints are
user-supplied inputs; none are bundled here. An optional padmap CSV must contain
`id` and `drug` columns and use identifiers matching the configured extraction
pattern. Keep the file outside this repository and provide its path in the run
configuration.
