from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from deciphaer_image_segmentation.channels import KNOWN_CHANNELS
from deciphaer_image_segmentation.spatial import scaled_integer, spatial_scale

RunMode = Literal["local", "slurm"]
LinkMode = Literal["symlink", "hardlink", "copy"]


@dataclass(frozen=True)
class RunConfig:
    name: str = "segmentation_run"
    output_dir: Path = Path("output/segmentation_run")
    mode: RunMode = "local"
    dry_run: bool = False
    seed: int = 0
    log_level: str = "INFO"
    keep_intermediate: bool = False


@dataclass(frozen=True)
class InputConfig:
    image_dir: Path
    pattern: str = "*Phase*.tif"
    recursive: bool = False
    phase_channel: str = "Phase"
    channels: tuple[str, ...] = ("Phase",)
    pixel_microns: float = 0.10317
    channel_axis: str = "auto"
    # Channel-name registry used to identify channels in filenames. Anything
    # added here is treated as a recognized channel token (matched as
    # ``_<name>`` followed by a separator), which lets the pipeline cope with
    # quirky exports like ``..._Cy5_ORG.tif`` without renaming files.
    known_channels: tuple[str, ...] = KNOWN_CHANNELS


@dataclass(frozen=True)
class SpatialCalibrationConfig:
    # None preserves legacy input-pixel settings. Otherwise pixel-unit settings
    # describe the reference sampling and are resolved at inputs.pixel_microns.
    reference_pixel_microns: float | None = None


@dataclass(frozen=True)
class BatchConfig:
    n_batches: int = 1
    batch_dir: Path | None = None
    overwrite_links: bool = True


@dataclass(frozen=True)
class MomiaConfig:
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    workers_per_batch: int = 1
    force: bool = False
    skip_existing: bool = True
    fail_fast: bool = False


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    config_path: Path | None = None
    require_momia_included: bool = True
    output_subdir: str = "filter"


@dataclass(frozen=True)
class UnetSaveConfig:
    crops: bool = True
    composites: bool = False
    passed_crops: bool = False
    include_available_channels: bool = True
    masks: tuple[str, ...] = ("Phase", "Mask")


@dataclass(frozen=True)
class UnetConfig:
    enabled: bool = False
    model_path: Path | None = None
    padmap: Path | None = None
    padmap_join: Literal["left", "inner"] = "left"
    padmap_id_pattern: str = r"((?:\d{4}_s\d+)|(?:t\d+_rep\d+_s\d+))"
    padmap_id_source: str = "source_path"
    device: str | None = None
    probability_threshold: float = 0.5
    keep_class: Literal["cell", "any"] = "cell"
    use_mask_channel: bool = True
    crop_size: int = 96
    min_bbox_pad: int = 4
    edge_pad_mode: Literal["reflect", "constant"] = "reflect"
    crop_workers: int = 1
    crop_shards: int = 1
    shards: int = 1
    save: UnetSaveConfig = field(default_factory=UnetSaveConfig)


@dataclass(frozen=True)
class SlurmConfig:
    partition: str | None = "preempt"
    time: str = "08:00:00"
    mem: str = "16g"
    cpus_per_task: int = 4
    job_prefix: str = "segv2"
    submit: bool = False
    account: str | None = None
    log_dir: Path | None = None
    gpu_partition: str | None = None
    gpu_gres: str | None = None
    gpu_time: str | None = None
    gpu_mem: str | None = None
    gpu_cpus_per_task: int | None = None
    gpu_requeue: bool = False
    # Sidecar monitor: a tiny slurm job that polls running MOMIA batches and
    # posts a webhook when one stalls (no new mask in ``monitor_stall_minutes``
    # while slurm still reports the job RUNNING). Disable by setting
    # ``monitor_enabled: false``.
    monitor_enabled: bool = True
    monitor_stall_minutes: float = 5.0
    monitor_poll_seconds: float = 60.0
    monitor_time: str | None = None  # defaults to slurm.time if unset
    monitor_mem: str = "2g"
    monitor_cpus_per_task: int = 1


@dataclass(frozen=True)
class ExportConfig:
    """Output / extraction configuration.

    feature_columns:
        Empty tuple means "derive automatically from the resolved channel
        set" — :mod:`channels.feature_columns_for_channels` produces the
        full schema (morphology + intensity/profile/fluor-common per
        channel + plugin extras). Pass a non-empty tuple to project a
        custom schema.

    extract_channels:
        Empty tuple, or ``("auto",)``, triggers auto-detection by scanning
        sibling TIFFs next to the first available Phase image. Pass an
        explicit tuple (e.g. ``("Phase", "AF405", "Bod493")``) to pin the
        schema for reproducible runs.
    """

    prefix: str = "sc_morph"
    include_unet_probabilities: bool = True
    feature_columns: tuple[str, ...] = ()
    extract_channels: tuple[str, ...] = ("auto",)
    workers: int = 1
    fluorescence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationsConfig:
    webhook_url: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    run: RunConfig
    inputs: InputConfig
    batching: BatchConfig = field(default_factory=BatchConfig)
    momia: MomiaConfig = field(default_factory=MomiaConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    unet: UnetConfig = field(default_factory=UnetConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    spatial_calibration: SpatialCalibrationConfig = field(default_factory=SpatialCalibrationConfig)

    @property
    def spatial_scale(self) -> float:
        return spatial_scale(self.inputs.pixel_microns, self.spatial_calibration.reference_pixel_microns)

    @property
    def effective_crop_size(self) -> int:
        return scaled_integer(self.unet.crop_size, self.spatial_scale, minimum=1)

    @property
    def effective_min_bbox_pad(self) -> int:
        return scaled_integer(self.unet.min_bbox_pad, self.spatial_scale)

    @property
    def output_dir(self) -> Path:
        return self.run.output_dir

    @property
    def batch_dir(self) -> Path:
        return self.batching.batch_dir or self.output_dir / "batches"

    @property
    def momia_batches_dir(self) -> Path:
        return self.output_dir / "momia" / "batches"

    @property
    def momia_compiled_dir(self) -> Path:
        return self.output_dir / "momia" / "compiled"

    @property
    def filter_dir(self) -> Path:
        return self.output_dir / self.filter.output_subdir

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "unet" / f"cell_crops_{self.effective_crop_size}"

    @property
    def unet_dir(self) -> Path:
        return self.output_dir / "unet"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"

    @property
    def batch_names(self) -> list[str]:
        return [f"batch{i}" for i in range(1, self.batching.n_batches + 1)]


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = _config_from_mapping(raw, config_path=config_path)
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if config.batching.n_batches < 1:
        raise ValueError("batching.n_batches must be at least 1")
    spatial_scale(config.inputs.pixel_microns, config.spatial_calibration.reference_pixel_microns)
    if config.unet.crop_size < 1:
        raise ValueError("unet.crop_size must be >= 1")
    if config.unet.min_bbox_pad < 0:
        raise ValueError("unet.min_bbox_pad must be >= 0")
    if config.momia.workers_per_batch < 1:
        raise ValueError("momia.workers_per_batch must be at least 1")
    if config.filter.enabled and config.filter.config_path is None:
        raise ValueError("filter.enabled requires filter.config_path")
    if config.filter.enabled and config.filter.config_path is not None and not config.filter.config_path.exists():
        raise FileNotFoundError(f"filter.config_path does not exist: {config.filter.config_path}")
    if config.unet.enabled:
        if config.unet.model_path is None:
            raise ValueError("unet.enabled requires unet.model_path")
        if config.unet.shards < 1:
            raise ValueError("unet.shards must be >= 1")
        if config.unet.crop_shards < 1:
            raise ValueError("unet.crop_shards must be >= 1")
    if config.slurm.cpus_per_task < 1:
        raise ValueError("slurm.cpus_per_task must be >= 1")


def _config_from_mapping(raw: dict[str, Any], *, config_path: Path) -> PipelineConfig:
    base_dir = config_path.parent
    run = _load_run(raw.get("run", {}), base_dir)
    inputs = _load_inputs(raw.get("inputs", {}), base_dir)
    batching = _load_batching(raw.get("batching", {}), base_dir)
    momia = _load_momia(raw.get("momia", {}), base_dir)
    filter_cfg = _load_filter(raw.get("filter", {}), base_dir)
    unet = _load_unet(raw.get("unet", {}), base_dir)
    slurm = _load_slurm(raw.get("slurm", {}), base_dir)
    export = _load_export(raw.get("export", {}))
    notifications = _load_notifications(raw.get("notifications", {}))
    spatial = _as_mapping(raw.get("spatial_calibration", {}), "spatial_calibration")
    unknown = spatial.keys() - {"reference_pixel_microns"}
    if unknown:
        raise ValueError(f"Unknown spatial_calibration settings: {sorted(unknown)}")
    reference = spatial.get("reference_pixel_microns")
    return PipelineConfig(
        config_path=config_path,
        run=run,
        inputs=inputs,
        batching=batching,
        momia=momia,
        filter=filter_cfg,
        unet=unet,
        slurm=slurm,
        export=export,
        notifications=notifications,
        spatial_calibration=SpatialCalibrationConfig(
            reference_pixel_microns=float(reference) if reference is not None else None,
        ),
    )


def _load_run(raw: Any, base_dir: Path) -> RunConfig:
    data = _as_mapping(raw, "run")
    return RunConfig(
        name=str(data.get("name", "segmentation_run")),
        output_dir=_resolve_path(data.get("output_dir", "output/segmentation_run"), base_dir),
        mode=str(data.get("mode", "local")),  # type: ignore[arg-type]
        dry_run=bool(data.get("dry_run", False)),
        seed=int(data.get("seed", 0)),
        log_level=str(data.get("log_level", "INFO")),
        keep_intermediate=bool(data.get("keep_intermediate", False)),
    )


def _load_inputs(raw: Any, base_dir: Path) -> InputConfig:
    data = _as_mapping(raw, "inputs")
    image_dir_raw = data.get("image_dir")
    if not image_dir_raw:
        raise ValueError("inputs.image_dir is required")
    channels = tuple(str(v) for v in data.get("channels", ["Phase"]))
    # `inputs.known_channels` is optional. When omitted we fall back to the
    # built-in KNOWN_CHANNELS registry; when provided we *extend* the registry
    # so users adding a new fluorophore don't accidentally lose Phase/AF405/etc.
    user_known = data.get("known_channels")
    if user_known is None:
        known = KNOWN_CHANNELS
    else:
        if not isinstance(user_known, list):
            raise ValueError("inputs.known_channels must be a list of strings")
        extras = tuple(str(v) for v in user_known if str(v))
        known = tuple(dict.fromkeys((*KNOWN_CHANNELS, *extras)))
    return InputConfig(
        image_dir=_resolve_path(image_dir_raw, base_dir),
        pattern=str(data.get("pattern", "*Phase*.tif")),
        recursive=bool(data.get("recursive", False)),
        phase_channel=str(data.get("phase_channel", "Phase")),
        channels=channels,
        pixel_microns=float(data.get("pixel_microns", 0.10317)),
        channel_axis=str(data.get("channel_axis", "auto")),
        known_channels=known,
    )


def _load_batching(raw: Any, base_dir: Path) -> BatchConfig:
    data = _as_mapping(raw, "batching")
    return BatchConfig(
        n_batches=int(data.get("n_batches", 1)),
        batch_dir=_resolve_optional_path(data.get("batch_dir"), base_dir),
        overwrite_links=bool(data.get("overwrite_links", True)),
    )


def _load_momia(raw: Any, base_dir: Path) -> MomiaConfig:
    data = _as_mapping(raw, "momia")
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("momia.settings must be a mapping")
    return MomiaConfig(
        enabled=bool(data.get("enabled", True)),
        settings=dict(settings),
        workers_per_batch=int(data.get("workers_per_batch", 1)),
        force=bool(data.get("force", False)),
        skip_existing=bool(data.get("skip_existing", True)),
        fail_fast=bool(data.get("fail_fast", False)),
    )


def _load_filter(raw: Any, base_dir: Path) -> FilterConfig:
    data = _as_mapping(raw, "filter")
    return FilterConfig(
        enabled=bool(data.get("enabled", False)),
        config_path=_resolve_optional_path(data.get("config_path") or data.get("filter_config"), base_dir),
        require_momia_included=bool(data.get("require_momia_included", True)),
        output_subdir=str(data.get("output_subdir", "filter")),
    )


def _load_unet(raw: Any, base_dir: Path) -> UnetConfig:
    data = _as_mapping(raw, "unet")
    save = _load_unet_save(data.get("save", {}))
    return UnetConfig(
        enabled=bool(data.get("enabled", False)),
        model_path=_resolve_optional_path(data.get("model_path") or data.get("checkpoint"), base_dir),
        padmap=_resolve_optional_path(data.get("padmap"), base_dir),
        padmap_join=str(data.get("padmap_join", "left")),  # type: ignore[arg-type]
        padmap_id_pattern=str(data.get("padmap_id_pattern", r"((?:\d{4}_s\d+)|(?:t\d+_rep\d+_s\d+))")),
        padmap_id_source=str(data.get("padmap_id_source", "source_path")),
        device=data.get("device"),
        probability_threshold=float(data.get("probability_threshold", data.get("cell_prob_threshold", 0.5))),
        keep_class=str(data.get("keep_class", "cell")),  # type: ignore[arg-type]
        use_mask_channel=bool(data.get("use_mask_channel", True)),
        crop_size=int(data.get("crop_size", 96)),
        min_bbox_pad=int(data.get("min_bbox_pad", 4)),
        edge_pad_mode=str(data.get("edge_pad_mode", "reflect")),  # type: ignore[arg-type]
        crop_workers=int(data.get("crop_workers", 1)),
        crop_shards=int(data.get("crop_shards", 1)),
        shards=int(data.get("shards", 1)),
        save=save,
    )


def _load_unet_save(raw: Any) -> UnetSaveConfig:
    data = _as_mapping(raw, "unet.save")
    masks_raw = data.get("masks", ["Phase", "Mask"])
    masks: list[str] = []
    for value in masks_raw:
        if isinstance(value, list):
            masks.extend(str(item) for item in value)
        else:
            masks.append(str(value))
    return UnetSaveConfig(
        crops=bool(data.get("crops", True)),
        composites=bool(data.get("composites", False)),
        passed_crops=bool(data.get("passed_crops", False)),
        include_available_channels=bool(data.get("include_available_channels", True)),
        masks=tuple(masks),
    )


def _load_slurm(raw: Any, base_dir: Path) -> SlurmConfig:
    data = _as_mapping(raw, "slurm")
    return SlurmConfig(
        partition=data.get("partition", "preempt"),
        time=str(data.get("time", "08:00:00")),
        mem=str(data.get("mem", "16g")),
        cpus_per_task=int(data.get("cpus_per_task", data.get("cpus", 4))),
        job_prefix=str(data.get("job_prefix", "segv2")),
        submit=bool(data.get("submit", False)),
        account=data.get("account"),
        log_dir=_resolve_optional_path(data.get("log_dir"), base_dir),
        gpu_partition=data.get("gpu_partition"),
        gpu_gres=data.get("gpu_gres"),
        gpu_time=str(data["gpu_time"]) if data.get("gpu_time") is not None else None,
        gpu_mem=str(data["gpu_mem"]) if data.get("gpu_mem") is not None else None,
        gpu_cpus_per_task=int(data["gpu_cpus_per_task"]) if data.get("gpu_cpus_per_task") is not None else None,
        gpu_requeue=bool(data.get("gpu_requeue", False)),
        monitor_enabled=bool(data.get("monitor_enabled", True)),
        monitor_stall_minutes=float(data.get("monitor_stall_minutes", 5.0)),
        monitor_poll_seconds=float(data.get("monitor_poll_seconds", 60.0)),
        monitor_time=str(data["monitor_time"]) if data.get("monitor_time") is not None else None,
        monitor_mem=str(data.get("monitor_mem", "2g")),
        monitor_cpus_per_task=int(data.get("monitor_cpus_per_task", 1)),
    )


def _load_export(raw: Any) -> ExportConfig:
    data = _as_mapping(raw, "export")
    # ``feature_columns: []`` (or omitted) → derive the schema from the
    # resolved channel set in :func:`extract.run_extract`. Provide a
    # non-empty list to lock a specific schema.
    raw_features = data.get("feature_columns") or []
    feature_columns = tuple(str(v) for v in raw_features)
    # ``extract_channels: []`` or ``["auto"]`` triggers auto-detection.
    raw_channels = data.get("extract_channels")
    if raw_channels is None or (isinstance(raw_channels, list) and not raw_channels):
        extract_channels: tuple[str, ...] = ("auto",)
    else:
        extract_channels = tuple(str(v) for v in raw_channels)
    fluor = data.get("fluorescence", {}) or {}
    if not isinstance(fluor, dict):
        raise ValueError("export.fluorescence must be a mapping")
    return ExportConfig(
        prefix=str(data.get("prefix", "sc_morph")),
        include_unet_probabilities=bool(data.get("include_unet_probabilities", True)),
        feature_columns=feature_columns,
        extract_channels=extract_channels,
        workers=int(data.get("workers", 1)),
        fluorescence=dict(fluor),
    )


def _load_notifications(raw: Any) -> NotificationsConfig:
    data = _as_mapping(raw, "notifications")
    url = data.get("webhook_url")
    return NotificationsConfig(webhook_url=str(url) if url else None)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or value == "":
        return None
    return _resolve_path(value, base_dir)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value
