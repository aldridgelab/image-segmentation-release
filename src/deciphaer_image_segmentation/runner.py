from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time as _time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from deciphaer_image_segmentation import notify, skiplist
from deciphaer_image_segmentation.batching import setup_batch_folders
from deciphaer_image_segmentation.status import squeue_states
from deciphaer_image_segmentation.config import PipelineConfig
from deciphaer_image_segmentation.crops import (
    crop_cells_from_table,
    merge_crop_manifests,
    write_filtered_composites,
)
from deciphaer_image_segmentation.export import finalize_outputs
from deciphaer_image_segmentation.extract import run_extract
from deciphaer_image_segmentation.filtering import load_filter_config, resolve_filter_rules, run_filter_stage
from deciphaer_image_segmentation.momia import compile_momia_batches, effective_momia_settings, run_momia_batch
from deciphaer_image_segmentation.monitor import run_monitor
from deciphaer_image_segmentation.provenance import seed_everything, write_run_metadata
from deciphaer_image_segmentation.slurm import JobSpec, SlurmSubmitter
from deciphaer_image_segmentation.unet import classify_crops, merge_classification_shards, parse_shard

_log = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # __file__ → src/deciphaer_image_segmentation/runner.py → parents[2] is the repo root.
        self.repo_root = Path(__file__).resolve().parents[2]
        self.stage_script = self.repo_root / "scripts" / "slurm" / "run_stage.sh"
        self.log_dir = self._resolve_log_dir()

    @property
    def _log_dir_pointer(self) -> Path:
        """File under output_dir that records which subdir holds this run's logs."""
        return self.config.output_dir / ".slurm_log_dir"

    def _log_dir_base(self) -> Path:
        """Base of the log tree (config.slurm.log_dir or <repo>/logs)."""
        return self.config.slurm.log_dir or (self.repo_root / "logs")

    def _resolve_log_dir(self) -> Path:
        """Pick (or recall) the per-submission log subdir.

        Layout: ``<base>/<config-stem>-<short-hash>``. ``<base>`` is the global
        ``slurm.log_dir`` (or ``<repo>/logs``). The hash is derived from the
        config path + the timestamp of the first submission, so each fresh
        submission gets its own dir; resumes (and ``--status``) recall the
        same dir via a pointer file in ``output_dir``.
        """
        pointer = self._log_dir_pointer
        if pointer.exists():
            try:
                stored = pointer.read_text(encoding="utf-8").strip()
            except OSError:
                stored = ""
            if stored:
                return Path(stored)
        return self._fresh_log_dir()

    def _fresh_log_dir(self) -> Path:
        """Compute a brand-new log subdir for this submission."""
        stem = self.config.config_path.stem.replace("_", "-")
        seed = f"{self.config.config_path}|{_time.time_ns()}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
        return self._log_dir_base() / f"{stem}-{digest}"

    def _persist_log_dir(self) -> None:
        """Persist the chosen log dir so resume/--status can find it."""
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir_pointer.write_text(str(self.log_dir), encoding="utf-8")
        except OSError as exc:
            _log.warning("Could not persist log dir pointer (%s): %s", self._log_dir_pointer, exc)

    def _rotate_log_dir(self) -> None:
        """Discard the recalled log dir and pick a fresh one (e.g. --no-resume)."""
        self.log_dir = self._fresh_log_dir()

    def run_local(self) -> dict[str, Path]:
        seed_everything(self.config.run.seed)
        self.setup_batches()
        if self.config.momia.enabled:
            for batch_name in self.config.batch_names:
                run_momia_batch(self.config, batch_name)
            compile_momia_batches(
                self.config, log_dir=self.log_dir, skip_path=self.skip_path,
            )
        if self.config.filter.enabled:
            self.run_filter()
        if self.config.unet.enabled:
            self.run_crop()
            self.run_unet(shard=None)
        outputs = self.run_final()
        self.write_metadata(outputs)
        return outputs

    def setup_batches(self) -> Path:
        skip_tokens = skiplist.load(self.skip_path)
        _log.info(
            "setup: pattern=%s, image_dir=%s, n_batches=%d, skip=%d",
            self.config.inputs.pattern,
            self.config.inputs.image_dir,
            self.config.batching.n_batches,
            len(skip_tokens),
        )
        records = setup_batch_folders(
            image_dir=self.config.inputs.image_dir,
            batch_dir=self.config.batch_dir,
            n_batches=self.config.batching.n_batches,
            pattern=self.config.inputs.pattern,
            recursive=self.config.inputs.recursive,
            overwrite_links=self.config.batching.overwrite_links,
            skip_tokens=skip_tokens,
        )
        summary = {
            "n_images": len(records),
            "n_batches": self.config.batching.n_batches,
            "batch_dir": str(self.config.batch_dir),
            "n_skipped": len(skip_tokens),
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "batch_setup_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log.info("setup: linked %d images into %s", len(records), self.config.batch_dir)
        return path

    @property
    def skip_path(self) -> Path:
        return self.config.output_dir / "skip.txt"

    def run_filter(self) -> Path:
        if not self.config.filter.enabled or self.config.filter.config_path is None:
            return self.selected_cell_table_before_unet()
        result = run_filter_stage(
            cell_table_path=self.config.momia_compiled_dir / "cell_measurements.csv",
            filter_config_path=self.config.filter.config_path,
            output_dir=self.config.filter_dir,
            require_momia_included=self.config.filter.require_momia_included,
            spatial_scale=self.config.spatial_scale,
        )
        return result.table_path

    def run_crop(self, shard: str | None = None) -> Path:
        selected_channels = self._selected_channels_for_unet()
        parsed = parse_shard(shard)
        self.config.unet_dir.mkdir(parents=True, exist_ok=True)
        # Separate per-shard files avoid concurrent writes to shared provenance.
        suffix = "" if parsed is None else f"_shard_{parsed[0]}_of_{parsed[1]}"
        (self.config.unet_dir / f"crop_settings{suffix}.json").write_text(
            json.dumps(self._crop_settings(), indent=2), encoding="utf-8",
        )
        manifest = crop_cells_from_table(
            cell_table_path=self.selected_cell_table_before_unet(),
            masks_dir=self.config.momia_compiled_dir / "masks",
            crops_dir=self.config.crops_dir,
            selected_channels=selected_channels,
            phase_channel=self.config.inputs.phase_channel,
            crop_size=self.config.effective_crop_size,
            min_bbox_pad=self.config.effective_min_bbox_pad,
            edge_pad_mode=self.config.unet.edge_pad_mode,
            workers=self.config.unet.crop_workers,
            include_available_channels=self.config.unet.save.include_available_channels,
            segmentation_run_hash=self.segmentation_run_hash(),
            shard=parsed,
            known_channels=self.config.inputs.known_channels,
        )
        # Composites are an unsharded artifact today (single composites_manifest.csv);
        # only emit them when this isn't a per-shard crop job.
        if self.config.unet.save.composites and parsed is None:
            write_filtered_composites(
                cell_table_path=self.selected_cell_table_before_unet(),
                masks_dir=self.config.momia_compiled_dir / "masks",
                composites_dir=self.config.unet_dir / "composites",
                selected_channels=selected_channels,
                phase_channel=self.config.inputs.phase_channel,
                include_available_channels=self.config.unet.save.include_available_channels,
                segmentation_run_hash=self.segmentation_run_hash(),
                known_channels=self.config.inputs.known_channels,
            )
        return manifest

    def run_unet(self, shard: str | None) -> Path:
        if not self.config.unet.enabled or self.config.unet.model_path is None:
            raise ValueError("U-Net stage requires unet.enabled and unet.model_path")
        parsed = parse_shard(shard)
        if parsed is None:
            out_csv = self.config.unet_dir / "unet_classifications.csv"
        else:
            out_csv = self.config.unet_dir / f"unet_classifications_shard_{parsed[0]}_of_{parsed[1]}.csv"
        return classify_crops(
            crops_dir=self.config.crops_dir,
            output_csv=out_csv,
            model_path=self.config.unet.model_path,
            device=self.config.unet.device,
            probability_threshold=self.config.unet.probability_threshold,
            use_mask_channel=self.config.unet.use_mask_channel,
            shard=parsed,
        )

    def run_final(self) -> dict[str, Path]:
        unet_classifications = None
        crop_manifest = None
        if self.config.unet.enabled:
            crop_manifest = self.config.crops_dir / "crops_manifest.csv"
            if self.config.unet.crop_shards > 1 and not crop_manifest.exists():
                crop_manifest = merge_crop_manifests(
                    self.config.crops_dir, n_shards=self.config.unet.crop_shards
                )
            unet_classifications = self.config.unet_dir / "unet_classifications.csv"
            if self.config.unet.shards > 1 and not unet_classifications.exists():
                unet_classifications = merge_classification_shards(self.config.unet_dir, n_shards=self.config.unet.shards)
        outputs = run_extract(
            masks_dir=self.config.momia_compiled_dir / "masks",
            metadata_dir=self.config.momia_compiled_dir / "metadata",
            cell_table_path=self.selected_cell_table_before_unet(),
            crop_manifest_path=crop_manifest,
            unet_classifications_path=unet_classifications,
            output_dir=self.config.final_dir,
            pixel_microns=self.config.inputs.pixel_microns,
            channel_names=self.config.export.extract_channels,
            phase_channel=self.config.inputs.phase_channel,
            feature_columns=self.config.export.feature_columns,
            include_unet_filter=self.config.unet.enabled,
            probability_threshold=self.config.unet.probability_threshold,
            keep_class=self.config.unet.keep_class,
            require_momia_included=True,
            padmap_path=self.config.unet.padmap,
            padmap_id_pattern=self.config.unet.padmap_id_pattern,
            padmap_id_source=self.config.unet.padmap_id_source,
            padmap_join=self.config.unet.padmap_join,
            workers=self.config.export.workers,
            prefix=self.config.export.prefix,
            segmentation_run_hash=self.segmentation_run_hash(),
            fluorescence_config=self.config.export.fluorescence,
            known_channels=self.config.inputs.known_channels,
        )
        if self.config.unet.save.passed_crops and crop_manifest and crop_manifest.exists():
            self._link_passed_crops(outputs)
        if not self.config.run.keep_intermediate:
            self._cleanup_intermediate()
        self._notify_run_complete(outputs)
        return outputs

    def _link_passed_crops(self, outputs: dict[str, Path]) -> None:
        """Hardlink the crop TIFFs for cells that passed into final/passed_crops/.

        Replicates the behaviour of the legacy ``finalize_outputs`` flow: after
        extract decides which Cell_IDs survived the U-Net + filter, copy/link
        their crops into ``final/passed_crops`` so they're preserved when
        ``unet/cell_crops_*`` is cleaned up.  Uses the diagnostics CSV to know
        which cells passed.
        """
        import os
        import shutil as _shutil

        diagnostics_path = outputs.get("diagnostics")
        mapping_path = outputs.get("mapping")
        if diagnostics_path is None or not Path(diagnostics_path).exists():
            return
        if mapping_path is None or not Path(mapping_path).exists():
            return
        import pandas as pd
        diag = pd.read_csv(diagnostics_path)
        mapping = pd.read_csv(mapping_path)
        if "filter_passed" not in diag.columns:
            return
        passed_ids = set(diag.loc[diag["filter_passed"].astype(str).str.lower().isin({"true", "1", "yes"}), "Cell_ID"])
        if not passed_ids:
            return
        dst_dir = self.config.final_dir / "passed_crops"
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for _, row in mapping.iterrows():
            if row.get("Cell_ID") not in passed_ids:
                continue
            crop_path = str(row.get("crop_path") or "")
            if not crop_path:
                continue
            src = Path(crop_path)
            if not src.exists():
                continue
            dst = dst_dir / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                _shutil.copy2(src, dst)
            n += 1
        _log.info("final: linked %d passed crops -> %s", n, dst_dir)

    def _notify_run_complete(self, outputs: dict[str, Path]) -> None:
        url = self.config.notifications.webhook_url
        if not url:
            return
        summary_path = outputs.get("summary")
        n_cells: int | None = None
        if summary_path and Path(summary_path).exists():
            try:
                payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                # extract.run_extract writes ``n_cells_passed`` (post-filter
                # surviving count); fall back to legacy ``n_cells`` for older
                # runs.
                value = payload.get("n_cells_passed")
                if value is None:
                    value = payload.get("n_cells", 0)
                n_cells = int(value)
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                n_cells = None
        cell_str = f"{n_cells} cells" if n_cells is not None else "unknown cell count"
        notify.post(
            url,
            f":white_check_mark: Pipeline complete: *{self.config.run.name}* — {cell_str} written to `{self.config.final_dir}`",
        )

    def _cleanup_intermediate(self) -> None:
        """Remove intermediate batch/MOMIA/crop dirs once final outputs are written.

        Skipped when `run.keep_intermediate: true`. Final exports and (if requested)
        passed_crops live under final/ and are never touched.
        """
        targets: list[Path] = [
            self.config.batch_dir,
            self.config.output_dir / "momia",
            self.config.crops_dir,
        ]
        for target in targets:
            if not target.exists():
                continue
            try:
                shutil.rmtree(target)
                _log.info("Removed intermediate dir: %s", target)
            except OSError as exc:
                _log.warning("Failed to remove %s: %s", target, exc)

    def run_stage(self, stage: str, *, batch_name: str | None = None, shard: str | None = None) -> Path | dict[str, Path]:
        if stage == "setup":
            return self.setup_batches()
        if stage == "momia":
            if not batch_name:
                raise ValueError("stage=momia requires --batch-name")
            return run_momia_batch(self.config, batch_name)
        if stage == "compile":
            return compile_momia_batches(
                self.config, log_dir=self.log_dir, skip_path=self.skip_path,
            )
        if stage == "filter":
            return self.run_filter()
        if stage == "crop":
            return self.run_crop(shard=shard)
        if stage == "unet":
            return self.run_unet(shard)
        if stage == "final":
            outputs = self.run_final()
            self.write_metadata(outputs)
            return outputs
        if stage == "monitor":
            slurm_cfg = self.config.slurm
            return run_monitor(
                self.config,
                stall_minutes=slurm_cfg.monitor_stall_minutes,
                poll_seconds=slurm_cfg.monitor_poll_seconds,
            )
        raise ValueError(f"Unknown stage: {stage}")

    def selected_cell_table_before_unet(self) -> Path:
        filtered = self.config.filter_dir / "cell_measurements.csv"
        if self.config.filter.enabled and filtered.exists():
            return filtered
        return self.config.momia_compiled_dir / "cell_measurements.csv"

    def plan(self) -> list[JobSpec]:
        specs: list[JobSpec] = []
        # Sidecar stall monitor — runs in parallel with everything else,
        # tiny resources, no deps. Polls running MOMIA batches and posts a
        # webhook when one stops producing masks. Pure notifier; doesn't kill
        # or skip anything (the watchdog at compile time still does that).
        if self.config.slurm.monitor_enabled:
            monitor_name = f"{self.config.slurm.job_prefix}_monitor"
            specs.append(self._job(
                name=monitor_name,
                stage="monitor",
                dependencies=(),
                is_monitor=True,
            ))
        momia_names: list[str] = []
        for batch_name in self.config.batch_names:
            name = f"{self.config.slurm.job_prefix}_{batch_name}_momia"
            momia_names.append(name)
            specs.append(self._job(name=name, stage="momia", batch_name=batch_name, dependencies=()))
        compile_name = f"{self.config.slurm.job_prefix}_compile"
        # Compile uses `afterany` so a single failed/OOM batch can't wedge the
        # whole DAG in `DependencyNeverSatisfied`. compile_momia_batches itself
        # acts as a watchdog: detects which batches failed, posts one webhook
        # listing them with their diagnosed reasons, flags the affected images
        # into skip.txt, and proceeds with whatever batches did succeed.
        specs.append(self._job(
            name=compile_name,
            stage="compile",
            dependencies=tuple(momia_names),
            dependency_type="afterany",
        ))
        previous = compile_name
        if self.config.filter.enabled:
            filter_name = f"{self.config.slurm.job_prefix}_filter"
            specs.append(self._job(name=filter_name, stage="filter", dependencies=(previous,)))
            previous = filter_name
        if self.config.unet.enabled:
            crop_shards = self.config.unet.crop_shards
            if crop_shards <= 1:
                crop_name = f"{self.config.slurm.job_prefix}_crop"
                specs.append(self._job(name=crop_name, stage="crop", dependencies=(previous,)))
                crop_deps: tuple[str, ...] = (crop_name,)
            else:
                crop_names: list[str] = []
                for k in range(crop_shards):
                    name = f"{self.config.slurm.job_prefix}_crop_{k}_of_{crop_shards}"
                    crop_names.append(name)
                    specs.append(
                        self._job(
                            name=name,
                            stage="crop",
                            dependencies=(previous,),
                            shard=f"{k}/{crop_shards}",
                        )
                    )
                crop_deps = tuple(crop_names)
            unet_names = []
            for shard_idx in range(self.config.unet.shards):
                name = f"{self.config.slurm.job_prefix}_unet_{shard_idx}_of_{self.config.unet.shards}"
                unet_names.append(name)
                specs.append(
                    self._job(
                        name=name,
                        stage="unet",
                        dependencies=crop_deps,
                        shard=f"{shard_idx}/{self.config.unet.shards}",
                        use_gpu=True,
                    )
                )
            previous_deps = tuple(unet_names)
        else:
            previous_deps = (previous,)
        specs.append(self._job(name=f"{self.config.slurm.job_prefix}_final", stage="final", dependencies=previous_deps))
        return specs

    def submit(self, *, resume: bool = True) -> dict[str, str]:
        # On a clean (non-resume) submit, rotate to a fresh log subdir so the
        # new attempt's logs don't get mixed with the old run's.
        if not resume:
            self._rotate_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._persist_log_dir()
        _log.info("Logs for this submission: %s", self.log_dir)
        manifest = self.config.output_dir / "slurm_submission_manifest.json"
        old_manifest: dict[str, str] = {}
        if resume and manifest.exists():
            try:
                old_manifest = json.loads(manifest.read_text(encoding="utf-8")) or {}
            except json.JSONDecodeError:
                old_manifest = {}
        if resume:
            self._auto_flag_failed_images(manifest)
            cancelled = self._cancel_stale_jobs(manifest)
            if cancelled:
                _log.info("Cancelled %d stale pending job(s) before resubmit: %s", len(cancelled), ", ".join(cancelled))
        self.setup_batches()

        old_states = squeue_states(list(old_manifest.values())) if old_manifest else {}

        submitter = SlurmSubmitter()
        submitted: dict[str, str] = {}
        skipped_done: list[str] = []
        kept_running: list[str] = []
        for spec in self.plan():
            if resume and self._stage_done(spec):
                skipped_done.append(spec.name)
                continue
            old_job = old_manifest.get(spec.name)
            old_state = old_states.get(old_job, "") if old_job else ""
            # If a prior job for this stage is still running, do NOT resubmit:
            # both writers would race on the same outputs (e.g. run_summary.json).
            # Inherit its job ID so downstream stages can chain to it via afterok.
            if resume and old_state.startswith("RUNNING"):
                submitted[spec.name] = old_job
                kept_running.append(f"{spec.name}={old_job}")
                continue
            deps = tuple(submitted[name] for name in spec.dependencies if name in submitted)
            resolved = replace(spec, dependencies=deps)
            job_id = submitter.submit(resolved)
            submitted[spec.name] = job_id
        if skipped_done:
            _log.info("Resume: skipped %d already-completed job(s)", len(skipped_done))
        if kept_running:
            _log.info("Resume: kept %d still-running job(s): %s", len(kept_running), ", ".join(kept_running))
        manifest.write_text(json.dumps(submitted, indent=2), encoding="utf-8")
        return submitted

    def _auto_flag_failed_images(self, manifest_path: Path) -> None:
        """Resume-time wrapper around the watchdog helper.

        Same detection logic as the compile-time watchdog; defers to
        :func:`deciphaer_image_segmentation.watchdog.flag_failed_batches`.
        """
        from deciphaer_image_segmentation.watchdog import flag_failed_batches

        flag_failed_batches(
            self.config,
            log_dir=self.log_dir,
            skip_path=self.skip_path,
            webhook_url=self.config.notifications.webhook_url,
            manifest_path=manifest_path,
            context="resume",
        )

    def _cancel_stale_jobs(self, manifest_path: Path) -> list[str]:
        """Cancel any pending jobs from a prior submission so we can replace their DAG.

        A previously-submitted DAG often gets stuck in DependencyNeverSatisfied if any
        parent failed (e.g. OOM). Re-running the same command should yield a clean
        chain — that means cancelling the dead pending children first, since they'd
        otherwise sit forever and confuse `--status` for the new run.
        """
        import shutil as _shutil
        import subprocess

        if not manifest_path.exists():
            return []
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            return []
        if not old or _shutil.which("squeue") is None or _shutil.which("scancel") is None:
            return []
        ids = [str(v) for v in old.values() if v]
        if not ids:
            return []
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        states: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        to_cancel = [job_id for job_id in ids if states.get(job_id) == "PENDING"]
        if not to_cancel:
            return []
        try:
            subprocess.run(["scancel", *to_cancel], check=False, capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return []
        return to_cancel

    def _stage_done(self, spec: JobSpec) -> bool:
        if spec.stage == "momia":
            batch_name = spec.args[3] if len(spec.args) > 3 else ""
            if not batch_name:
                return False
            summary = self.config.momia_batches_dir / batch_name / "run_summary.json"
            if not summary.exists():
                return False
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return False
            return int(data.get("failed", 0)) == 0
        if spec.stage == "compile":
            return (self.config.momia_compiled_dir / "cell_measurements.csv").exists()
        if spec.stage == "filter":
            return (self.config.filter_dir / "cell_measurements.csv").exists()
        if spec.stage == "crop":
            shard_arg = spec.args[4] if len(spec.args) > 4 else ""
            if shard_arg:
                k, n = shard_arg.split("/", 1)
                shard_csv = self.config.crops_dir / f"crops_manifest_shard_{k}_of_{n}.csv"
                if shard_csv.exists():
                    return True
            return (self.config.crops_dir / "crops_manifest.csv").exists()
        if spec.stage == "unet":
            shard_arg = spec.args[4] if len(spec.args) > 4 else ""
            if shard_arg:
                k, n = shard_arg.split("/", 1)
                shard_csv = self.config.unet_dir / f"unet_classifications_shard_{k}_of_{n}.csv"
                if shard_csv.exists():
                    return True
            return (self.config.unet_dir / "unet_classifications.csv").exists()
        if spec.stage == "final":
            return (self.config.final_dir / "extract_summary.json").exists()
        return False

    def write_metadata(self, artifacts: dict[str, Path]) -> Path:
        input_paths = [str(self.config.inputs.image_dir)]
        artifact_strings = {key: str(value) for key, value in artifacts.items()}
        return write_run_metadata(
            output_dir=self.config.output_dir,
            config_snapshot=self._config_snapshot(),
            input_paths=input_paths,
            seed=self.config.run.seed,
            artifacts=artifact_strings,
            run_hash=self.segmentation_run_hash(),
        )

    def _job(
        self,
        *,
        name: str,
        stage: str,
        dependencies: tuple[str, ...],
        batch_name: str | None = None,
        shard: str | None = None,
        use_gpu: bool = False,
        is_monitor: bool = False,
        dependency_type: str = "afterok",
    ) -> JobSpec:
        args = [
            str(self.repo_root),
            str(self.config.config_path),
            stage,
            batch_name or "",
            shard or "",
        ]
        slurm = self.config.slurm
        partition = slurm.partition
        time_limit = slurm.time
        mem = slurm.mem
        cpus = slurm.cpus_per_task
        gres: str | None = None
        requeue = False
        if use_gpu and slurm.gpu_gres:
            gres = slurm.gpu_gres
            partition = slurm.gpu_partition or slurm.partition
            time_limit = slurm.gpu_time or slurm.time
            mem = slurm.gpu_mem or slurm.mem
            cpus = slurm.gpu_cpus_per_task or slurm.cpus_per_task
            requeue = slurm.gpu_requeue
        if is_monitor:
            # Monitor is a tiny polling loop — small CPU, small mem, but it
            # needs to outlive every MOMIA batch so it can warn on stalls.
            # Default time matches the rest of the DAG (so it's alive long
            # enough to see an OOM/hang); user can override via slurm.monitor_time.
            time_limit = slurm.monitor_time or slurm.time
            mem = slurm.monitor_mem
            cpus = slurm.monitor_cpus_per_task
        return JobSpec(
            name=name,
            stage=stage,
            script=self.stage_script,
            args=tuple(args),
            dependencies=dependencies,
            partition=partition,
            account=slurm.account,
            time=time_limit,
            mem=mem,
            cpus_per_task=cpus,
            output_path=str(self.log_dir / f"{name}-%j.log"),
            error_path=str(self.log_dir / f"{name}-%j.err"),
            gres=gres,
            requeue=requeue,
            dependency_type=dependency_type,
        )

    def _selected_channels_for_unet(self) -> tuple[str, ...]:
        channels = [channel for channel in self.config.unet.save.masks if channel != "Mask"]
        if self.config.inputs.phase_channel not in channels:
            channels.insert(0, self.config.inputs.phase_channel)
        channels.append("Mask")
        return tuple(dict.fromkeys(channels))

    def _crop_settings(self) -> dict[str, Any]:
        return {
            "pixel_microns": self.config.inputs.pixel_microns,
            "reference_pixel_microns": self.config.spatial_calibration.reference_pixel_microns,
            "scale_factor": self.config.spatial_scale,
            "requested_crop_size": self.config.unet.crop_size,
            "requested_min_bbox_pad": self.config.unet.min_bbox_pad,
            "crop_size_px": self.config.effective_crop_size,
            "min_bbox_pad_px": self.config.effective_min_bbox_pad,
            "crop_size_um": self.config.effective_crop_size * self.config.inputs.pixel_microns,
            "min_bbox_pad_um": self.config.effective_min_bbox_pad * self.config.inputs.pixel_microns,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        snapshot = _stringify_paths(asdict(self.config))
        resolved: dict[str, Any] = {"unet": self._crop_settings()}
        if self.config.momia.enabled:
            resolved["momia"] = effective_momia_settings(self.config)
        if self.config.filter.enabled and self.config.filter.config_path is not None:
            resolved["filter_rules"] = resolve_filter_rules(
                load_filter_config(self.config.filter.config_path), spatial_scale=self.config.spatial_scale,
            )
        snapshot["resolved_spatial_settings"] = resolved
        return snapshot

    def segmentation_run_hash(self) -> str:
        payload = json.dumps(self._config_snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
