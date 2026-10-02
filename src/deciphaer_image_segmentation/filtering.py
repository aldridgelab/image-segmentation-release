from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    table_path: Path
    summary_path: Path
    n_input: int
    n_passed: int


def run_filter_stage(
    *,
    cell_table_path: Path,
    filter_config_path: Path,
    output_dir: Path,
    require_momia_included: bool,
    spatial_scale: float = 1.0,
) -> FilterResult:
    rules = resolve_filter_rules(load_filter_config(filter_config_path), spatial_scale=spatial_scale)
    df = pd.read_csv(cell_table_path)
    n_rules = len(_normalize_rules(rules))
    _log.info(
        "filter: loaded %d cells from %s; applying %d rule(s) from %s (require_momia_included=%s)",
        len(df),
        cell_table_path,
        n_rules,
        filter_config_path,
        require_momia_included,
    )
    passed, reasons = evaluate_rows(df, rules=rules, require_momia_included=require_momia_included)
    filtered = df.copy()
    filtered["filter_passed"] = passed
    filtered["filter_reasons"] = reasons
    filtered["included"] = passed

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "cell_measurements.csv"
    filtered.to_csv(table_path, index=False)
    n_passed = int(np.asarray(passed, dtype=bool).sum())
    summary = {
        "filter_config": str(filter_config_path),
        "spatial_scale": spatial_scale,
        "effective_rules": rules,
        "n_input": int(len(df)),
        "n_passed": n_passed,
        "n_failed": int(len(df) - n_passed),
    }
    summary_path = output_dir / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log.info(
        "filter: %d/%d passed (%d filtered out) — wrote %s",
        n_passed,
        len(df),
        len(df) - n_passed,
        table_path,
    )
    return FilterResult(
        table_path=table_path,
        summary_path=summary_path,
        n_input=summary["n_input"],
        n_passed=summary["n_passed"],
    )


def load_filter_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Filter config must be a JSON object: {path}")
    return payload


def resolve_filter_rules(payload: dict[str, Any], *, spatial_scale: float = 1.0) -> dict[str, Any]:
    """Resolve known pixel-area bounds; physical and custom columns stay unchanged."""
    rules = _normalize_rules(payload)
    if spatial_scale != 1 and "area_px" in rules:
        area = rules["area_px"]
        if "equals" in area:
            raise ValueError("Calibration-aware area_px rules require min/max bounds, not equals")
        for bound in ("min", "max"):
            if area.get(bound) is not None:
                area[bound] = float(area[bound]) * spatial_scale * spatial_scale
    return rules


def evaluate_rows(
    df: pd.DataFrame,
    *,
    rules: dict[str, Any],
    require_momia_included: bool,
) -> tuple[list[bool], list[str]]:
    normalized = _normalize_rules(rules)
    passed: list[bool] = []
    reasons: list[str] = []
    for _, row in df.iterrows():
        row_passed = True
        row_reasons: list[str] = []
        if require_momia_included and "included" in row and not _as_bool(row["included"]):
            row_passed = False
            row_reasons.append("momia_excluded")
        for column, rule in normalized.items():
            ok, reason = _evaluate_rule(row, column, rule)
            if not ok:
                row_passed = False
                row_reasons.append(reason)
        passed.append(row_passed)
        reasons.append(";".join(row_reasons))
    return passed, reasons


def _normalize_rules(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rules = payload.get("rules", payload)
    if not isinstance(raw_rules, dict):
        raise ValueError("Filter rules must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw_rules.items():
        if key.endswith("_min"):
            column = key.removesuffix("_min")
            normalized.setdefault(_canonical_column(column), {})["min"] = value
        elif key.endswith("_max"):
            column = key.removesuffix("_max")
            normalized.setdefault(_canonical_column(column), {})["max"] = value
        elif isinstance(value, dict):
            normalized[_canonical_column(key)] = dict(value)
        else:
            normalized[_canonical_column(key)] = {"equals": value}
    return normalized


def _canonical_column(name: str) -> str:
    return {
        "area": "area_px",
        "length": "length_um",
        "width": "width_um",
    }.get(name, name)


def _evaluate_rule(row: pd.Series, column: str, rule: dict[str, Any]) -> tuple[bool, str]:
    if column not in row.index:
        return False, f"{column}:missing"
    value = row[column]
    if "equals" in rule:
        expected = rule["equals"]
        if isinstance(expected, bool):
            ok = _as_bool(value) is expected
        else:
            ok = str(value) == str(expected)
        return ok, "" if ok else f"{column}:expected_{expected}"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False, f"{column}:not_numeric"
    if not np.isfinite(numeric):
        return False, f"{column}:not_finite"
    if rule.get("min") is not None and numeric < float(rule["min"]):
        return False, f"{column}:below_min"
    if rule.get("max") is not None and numeric > float(rule["max"]):
        return False, f"{column}:above_max"
    return True, ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
