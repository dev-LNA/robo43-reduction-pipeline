"""Preflight validation of resolved frame metadata, and override-file handling.

:class:`OverrideSet` parses ``--set``/``--overrides`` values into the
``(value, FieldOrigin)`` pairs :func:`robo43.io.headers.read_meta` layers on
top of header/alias resolution.  :func:`validate` then checks a batch of
already-resolved :class:`~robo43.io.meta.FrameMeta` against what each
pipeline stage requires, producing a :class:`ValidationReport` that can be
rendered as a table or used to emit a fill-in-the-blanks overrides template.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robo43.exceptions import ConfigError, ValidationError
from robo43.io.meta import FieldOrigin, FrameMeta, Source

logger = logging.getLogger(__name__)

TIER1: tuple[str, ...] = ("naxis1", "naxis2", "frame_type", "exptime", "date_obs")

STAGE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "bias": (),
    "dark": ("ccd_temp",),
    "flat": ("filter",),
    "reduce": ("filter",),
    "astrometry": ("ra_deg", "dec_deg", "pixel_scale"),
    "photometry": ("filter", "gain", "saturation"),
    "zeropoint": ("airmass",),
}

#: Override-file/CLI key spelling -> FrameMeta attribute name.
_OVERRIDE_KEY_TO_FIELD: dict[str, str] = {
    "imagetyp": "frame_type",
    "frame_type": "frame_type",
    "ra": "ra_deg",
    "ra_deg": "ra_deg",
    "dec": "dec_deg",
    "dec_deg": "dec_deg",
    "focal_length": "focal_length_mm",
    "focal_length_mm": "focal_length_mm",
}

#: Reverse of the non-identity entries above, for template emission.
_FIELD_TO_OVERRIDE_KEY: dict[str, str] = {
    "frame_type": "imagetyp",
    "ra_deg": "ra",
    "dec_deg": "dec",
    "focal_length_mm": "focal_length",
}

#: FrameMeta field name -> the type its override value should be coerced to.
_FIELD_TYPES: dict[str, type] = {
    "naxis1": int, "naxis2": int,
    "frame_type": str,
    "exptime": float,
    "filter": str,
    "ra_deg": float, "dec_deg": float,
    "pixel_scale": float,
    "ccd_temp": float,
    "gain": float,
    "read_noise": float,
    "saturation": float,
    "airmass": float,
    "object": str, "telescope": str, "instrument": str, "observer": str,
    "binning": int,
    "focal_length_mm": float,
    "site_lat_deg": float, "site_lon_deg": float, "site_elevation_m": float,
    "mjd": float,
}


def _coerce_value(key: str, value: Any) -> Any:
    """Coerce ``value`` (from TOML or a ``--set`` string) for override key ``key``.

    ``key`` is the ORIGINAL, untranslated override-file/CLI key. The target
    type is looked up by translating ``key`` to its ``FrameMeta`` field name
    first; an unrecognised key is passed through unchanged. Raises
    :class:`ValidationError` naming the original key on a bad coercion.
    """
    field_name = _OVERRIDE_KEY_TO_FIELD.get(key, key)
    type_fn = _FIELD_TYPES.get(field_name)
    if type_fn is None:
        return value
    try:
        coerced = type_fn(value)
    except (TypeError, ValueError) as exc:
        msg = f"cannot interpret override '{key}={value!r}' as {type_fn.__name__}"
        raise ValidationError(msg) from exc
    if field_name == "frame_type":
        coerced = str(coerced).lower()
    return coerced


def _has_wildcard(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _specificity_key(pattern: str) -> tuple[int, int, int]:
    """Sort key for override patterns, least specific first.

    Ordering rule: more path separators first, then longer pattern, and
    exact paths beat globs. Concretely: ``(has_wildcard, separator_count,
    length)`` ascending, so wildcard patterns sort before exact paths, and
    within a tier more separators / a longer string sort later (more
    specific).
    """
    return (0 if _has_wildcard(pattern) else 1, pattern.count("/"), len(pattern))


def _matches(pattern: str, path: Path, root: Path | None) -> bool:
    if root is not None and path.is_relative_to(root):
        candidate_str = path.relative_to(root).as_posix()
    else:
        candidate_str = path.as_posix()
    matched = fnmatch.fnmatch(candidate_str, pattern)
    if not matched and "/" not in pattern:
        matched = fnmatch.fnmatch(path.name, pattern)
    return matched


def _coerce_table(table: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(key, value) for key, value in table.items()}


@dataclass(frozen=True, slots=True)
class OverrideSet:
    """A set of field overrides from a TOML file and/or ``--set`` arguments.

    ``defaults``, ``per_pattern``, and ``cli_set`` are stored keyed by the
    ORIGINAL (untranslated) override-file/CLI key spelling, already
    type-coerced. ``per_pattern`` is ordered least-specific to most-specific
    per :func:`_specificity_key`. Key translation into ``FrameMeta`` field
    names happens only in :meth:`for_path`.
    """

    defaults: dict[str, Any]
    per_pattern: tuple[tuple[str, dict[str, Any]], ...]
    cli_set: dict[str, Any]

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> OverrideSet:
        path = Path(path)
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except OSError as exc:
            msg = f"could not read overrides file {path}: {exc}"
            raise ConfigError(msg) from exc
        except tomllib.TOMLDecodeError as exc:
            msg = f"could not parse overrides file {path} as TOML: {exc}"
            raise ConfigError(msg) from exc

        raw_defaults = raw.pop("defaults", {})
        if not isinstance(raw_defaults, dict):
            msg = f"{path}: [defaults] must be a table, got {type(raw_defaults).__name__}"
            raise ConfigError(msg)
        defaults = _coerce_table(raw_defaults)

        per_pattern: list[tuple[str, dict[str, Any]]] = []
        for key, table in raw.items():
            if not isinstance(table, dict):
                msg = f"{path}: section [{key!r}] must be a table, got {type(table).__name__}"
                raise ConfigError(msg)
            per_pattern.append((key, _coerce_table(table)))

        per_pattern.sort(key=lambda item: _specificity_key(item[0]))
        return cls(defaults=defaults, per_pattern=tuple(per_pattern), cli_set={})

    @classmethod
    def from_cli(cls, set_args: Sequence[str]) -> OverrideSet:
        cli_set: dict[str, Any] = {}
        for arg in set_args:
            key, sep, value = arg.partition("=")
            if not sep:
                msg = f"malformed --set argument {arg!r}, expected KEY=VALUE"
                raise ValidationError(msg)
            cli_set[key] = _coerce_value(key, value)
        return cls(defaults={}, per_pattern=(), cli_set=cli_set)

    def merge(self, other: OverrideSet) -> OverrideSet:
        defaults = {**self.defaults, **other.defaults}
        combined = self.per_pattern + other.per_pattern
        per_pattern = tuple(sorted(combined, key=lambda item: _specificity_key(item[0])))
        cli_set = {**self.cli_set, **other.cli_set}
        return OverrideSet(defaults=defaults, per_pattern=per_pattern, cli_set=cli_set)

    def for_path(self, path: Path, root: Path | None = None) -> dict[str, tuple[Any, FieldOrigin]]:
        result: dict[str, tuple[Any, FieldOrigin]] = {}
        for key, value in self.defaults.items():
            field_name = _OVERRIDE_KEY_TO_FIELD.get(key, key)
            result[field_name] = (value, FieldOrigin(Source.OVERRIDE, "defaults"))
        for pattern, values in self.per_pattern:
            if _matches(pattern, path, root):
                for key, value in values.items():
                    field_name = _OVERRIDE_KEY_TO_FIELD.get(key, key)
                    result[field_name] = (value, FieldOrigin(Source.OVERRIDE, pattern))
        for key, value in self.cli_set.items():
            field_name = _OVERRIDE_KEY_TO_FIELD.get(key, key)
            result[field_name] = (value, FieldOrigin(Source.OVERRIDE, "--set"))
        return result


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: Path
    field: str
    stage: str
    reason: str


_TABLE_FIELDS: tuple[str, ...] = (
    "naxis1", "naxis2", "frame_type", "exptime", "date_obs",
    "filter", "ra_deg", "dec_deg", "pixel_scale", "ccd_temp",
    "gain", "read_noise", "saturation", "airmass",
    "object", "telescope", "instrument", "observer", "binning",
    "focal_length_mm", "site_lat_deg", "site_lon_deg", "site_elevation_m", "mjd",
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    metas: tuple[FrameMeta, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def blocking(self) -> tuple[ValidationIssue, ...]:
        return self.issues

    def as_table(self) -> pd.DataFrame:
        groups: dict[tuple[Any, Any, Any], list[FrameMeta]] = {}
        for meta in self.metas:
            # instrument label falls back: instrument -> telescope -> profile_name.
            instrument = meta.instrument or meta.telescope or meta.profile_name
            key = (instrument, meta.filter, meta.exptime)
            groups.setdefault(key, []).append(meta)

        rows: dict[str, list[Any]] = {
            "instrument": [], "filter": [], "exptime": [], "n_frames": [],
            "field": [], "resolution": [],
        }
        for (instrument, filt, exptime), metas in groups.items():
            for field_name in _TABLE_FIELDS:
                values_present = [m for m in metas if m.get(field_name) is not None]
                if not values_present:
                    resolution = "MISSING"
                else:
                    sources: set[Source] = set()
                    for m in values_present:
                        origin = m.origin(field_name)
                        sources.add(origin.source if origin is not None else Source.HEADER)
                    resolution = sources.pop().value if len(sources) == 1 else "MIXED"
                rows["instrument"].append(instrument)
                rows["filter"].append(filt)
                rows["exptime"].append(exptime)
                rows["n_frames"].append(len(metas))
                rows["field"].append(field_name)
                rows["resolution"].append(resolution)

        table = pd.DataFrame(rows)
        return table.sort_values(
            by=["instrument", "filter", "exptime", "field"]
        ).reset_index(drop=True)

    def emit_overrides_template(self, path: str | os.PathLike[str]) -> None:
        missing_by_field: dict[str, list[Path]] = {}
        for issue in self.issues:
            if issue.reason != "required field is missing":
                continue
            missing_by_field.setdefault(issue.field, []).append(issue.path)

        lines: list[str] = [
            "# Auto-generated overrides template -- fill in the blank values below."
        ]

        default_fields = [
            field_name
            for field_name, paths in missing_by_field.items()
            if len(paths) == len(self.metas)
        ]
        if default_fields:
            lines.append("")
            lines.append("[defaults]")
            for field_name in default_fields:
                key = _FIELD_TO_OVERRIDE_KEY.get(field_name, field_name)
                lines.append(f'{key} = ""')

        # per-path sections for fields missing on only some metas.
        per_path_fields: dict[Path, list[str]] = {}
        for field_name, paths in missing_by_field.items():
            if field_name in default_fields:
                continue
            for p in paths:
                per_path_fields.setdefault(p, []).append(field_name)

        for p, field_names in per_path_fields.items():
            escaped = str(p).replace("\\", "\\\\").replace('"', '\\"')
            lines.append("")
            lines.append(f'["{escaped}"]')
            for field_name in field_names:
                key = _FIELD_TO_OVERRIDE_KEY.get(field_name, field_name)
                lines.append(f'{key} = ""')

        text = "\n".join(lines) + "\n"
        Path(path).write_text(text, encoding="utf-8")


def validate(
    metas: Iterable[FrameMeta], *, stages: Sequence[str], strict: bool = False
) -> ValidationReport:
    unknown_stages = [stage for stage in stages if stage not in STAGE_REQUIREMENTS]
    if unknown_stages:
        valid = ", ".join(sorted(STAGE_REQUIREMENTS))
        msg = f"unknown stage(s) {unknown_stages!r}; valid stages are: {valid}"
        raise ValueError(msg)

    metas = tuple(metas)
    issues: list[ValidationIssue] = []
    for meta in metas:
        for stage in stages:
            required = STAGE_REQUIREMENTS[stage]
            for field_name in meta.missing(required):
                issues.append(
                    ValidationIssue(meta.path, field_name, stage, "required field is missing")
                )
            if strict:
                for field_name in required:
                    origin = meta.origin(field_name)
                    if origin is not None and origin.source is Source.DERIVED:
                        issues.append(
                            ValidationIssue(
                                meta.path,
                                field_name,
                                stage,
                                f"value was derived ({origin.detail}); --strict refuses this",
                            )
                        )
    return ValidationReport(metas=metas, issues=tuple(issues))
