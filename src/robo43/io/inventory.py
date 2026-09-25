"""Batch frame inventory: one row per file, header read exactly once when
profile is auto-resolved."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.time import Time

from robo43.config import Settings
from robo43.instruments.keywords import CANONICAL_IMAGETYP
from robo43.instruments.profile import InstrumentProfile, resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header
from robo43.io.headers import read_meta
from robo43.io.meta import FieldOrigin, FrameMeta, Source
from robo43.io.validate import OverrideSet

logger = logging.getLogger(__name__)

#: FrameMeta's scalar fields verbatim (path first, "origins" replaced by
#: "origins_json"), plus a derived "imagetyp" column (canonical ZERO/DARK/
#: OBJECT/FLAT/FOCUS vocabulary, computed from frame_type) and "read_error".
INVENTORY_COLUMNS: tuple[str, ...] = (
    "path", "naxis1", "naxis2", "frame_type", "imagetyp", "exptime", "date_obs",
    "filter", "ra_deg", "dec_deg", "pixel_scale", "ccd_temp", "gain_setting",
    "read_noise", "saturation", "airmass", "object", "telescope", "instrument",
    "observer", "binning", "readout_mode", "focal_length_mm", "site_lat_deg",
    "site_lon_deg", "site_elevation_m", "mjd", "night", "is_valid", "profile_name",
    "origins_json", "read_error",
)

_STRING_COLUMNS: tuple[str, ...] = (
    "path", "frame_type", "imagetyp", "date_obs", "filter", "object", "telescope",
    "instrument", "observer", "readout_mode", "night", "profile_name",
    "origins_json", "read_error",
)


@dataclass(frozen=True, slots=True)
class _InventoryTask:
    path: str
    profile: InstrumentProfile | None
    overrides: OverrideSet | None
    settings: Settings | None


def _build_row(task: _InventoryTask) -> dict[str, Any]:
    path = Path(task.path)
    row: dict[str, Any] = dict.fromkeys(INVENTORY_COLUMNS)
    row["path"] = str(path)
    try:
        if task.profile is not None:
            profile = task.profile
            header = read_header(path, profile)
        else:
            # once-only-read guarantee: exactly one fits.open per file.
            header = read_header(path, profile=None)
            profile = resolve_profile(header=header)
        meta = read_meta(
            path,
            profile,
            overrides=task.overrides,
            settings=task.settings,
            header=header,
        )
    except Exception as exc:
        row["read_error"] = str(exc)
        logger.warning("%s: could not build inventory row: %s", path, exc)
        return row

    for field_name in INVENTORY_COLUMNS:
        if field_name in ("path", "imagetyp", "origins_json", "read_error"):
            continue
        value = meta.get(field_name)
        if field_name == "date_obs":
            value = value.isot if value is not None else None
        elif field_name == "night":
            value = value.isoformat() if value is not None else None
        row[field_name] = value
    row["imagetyp"] = CANONICAL_IMAGETYP.get(meta.frame_type) if meta.frame_type else None
    row["origins_json"] = json.dumps(
        {f: {"source": o.source.value, "detail": o.detail} for f, o in meta.origins.items()}
    )
    row["read_error"] = None
    return row


def build_inventory(
    paths: Any,
    *,
    profile: InstrumentProfile | None = None,
    overrides: OverrideSet | None = None,
    settings: Settings | None = None,
    nprocs: int = 1,
) -> pd.DataFrame:
    """One row per FITS file under `paths` (anything iter_fits_files accepts).

    When profile is None, each file's profile is auto-resolved from its own
    primary header, reading the file exactly once; this cannot correctly
    resolve a multi-extension instrument's layout in auto mode, so such a
    file lands as a read_error row rather than raising.
    """
    file_paths = list(iter_fits_files(paths))
    tasks = [_InventoryTask(str(p), profile, overrides, settings) for p in file_paths]

    if nprocs > 1 and tasks:
        with multiprocessing.get_context("spawn").Pool(processes=nprocs) as pool:
            rows = list(pool.imap(_build_row, tasks))
    else:
        rows = [_build_row(t) for t in tasks]

    df = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))
    for col in _STRING_COLUMNS:
        df[col] = df[col].astype(object)
    return df


def metas_from_inventory(df: pd.DataFrame) -> list[FrameMeta]:
    """Reconstruct FrameMeta objects. Rows with a non-null read_error are
    skipped (logged, never raised). origins is rebuilt from the FULL
    origins_json mapping (not FrameMeta.provenance_json's filtered subset),
    so provenance_records() on a reloaded row matches the original exactly."""
    metas: list[FrameMeta] = []
    for _, row in df.iterrows():
        if row.get("read_error"):
            logger.warning("%s: skipping row with read_error: %s", row["path"], row["read_error"])
            continue

        def _v(name: str) -> Any:
            val = row[name]  # noqa: B023
            return None if pd.isna(val) else val

        origins_raw = json.loads(row["origins_json"]) if row["origins_json"] else {}
        origins = {f: FieldOrigin(Source(o["source"]), o["detail"]) for f, o in origins_raw.items()}
        date_obs = Time(_v("date_obs"), format="isot", scale="utc") if _v("date_obs") else None
        import datetime as _dt
        night = _dt.date.fromisoformat(_v("night")) if _v("night") else None
        metas.append(
            FrameMeta(
                path=Path(row["path"]), naxis1=int(_v("naxis1")), naxis2=int(_v("naxis2")),
                frame_type=_v("frame_type"), exptime=_v("exptime"), date_obs=date_obs,
                filter=_v("filter"), ra_deg=_v("ra_deg"), dec_deg=_v("dec_deg"),
                pixel_scale=_v("pixel_scale"), ccd_temp=_v("ccd_temp"),
                gain_setting=_v("gain_setting"), read_noise=_v("read_noise"),
                saturation=_v("saturation"), airmass=_v("airmass"), object=_v("object"),
                telescope=_v("telescope"), instrument=_v("instrument"), observer=_v("observer"),
                binning=_v("binning"), readout_mode=_v("readout_mode"),
                focal_length_mm=_v("focal_length_mm"), site_lat_deg=_v("site_lat_deg"),
                site_lon_deg=_v("site_lon_deg"), site_elevation_m=_v("site_elevation_m"),
                mjd=_v("mjd"), night=night, is_valid=bool(_v("is_valid")),
                profile_name=_v("profile_name"), origins=origins,
            )
        )
    return metas


def write_inventory(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Dispatch on Path(path).suffix: '.csv'->to_csv, '.parquet'->to_parquet, else ValueError."""
    p = Path(path)
    if p.suffix == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        msg = f"unsupported inventory file extension {p.suffix!r} (expected .csv or .parquet)"
        raise ValueError(msg)


def filter_inventory(
    df: pd.DataFrame,
    *,
    frame_type: str | None = None,
    filter: str | None = None,
    object: str | None = None,
    night: Any = None,
    exptime: float | None = None,
    ccd_temp: float | None = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    """`frame_type` matches the canonical `imagetyp` column (ZERO/DARK/OBJECT/
    FLAT/FOCUS, case-insensitive) -- not the internal frame_type column --
    per the pipeline's binding IMAGETYP vocabulary. `filter` never matches a
    ZERO or DARK row regardless of that row's FILTER header, since FILTER is
    meaningless on bias/dark frames."""
    import datetime as _dt

    out = df
    if valid_only:
        out = out[(~out["is_valid"].eq(False)) | out["read_error"].notna()]
    if frame_type is not None:
        out = out[out["imagetyp"].str.upper() == frame_type.upper()]
    if filter is not None:
        out = out[(~out["imagetyp"].isin(("ZERO", "DARK"))) & (out["filter"] == filter)]
    if object is not None:
        out = out[out["object"] == object]
    if night is not None:
        night_str = night.isoformat() if isinstance(night, _dt.date) else str(night)
        out = out[out["night"] == night_str]
    if exptime is not None:
        out = out[np.isclose(out["exptime"].astype(float), exptime)]
    if ccd_temp is not None:
        out = out[np.isclose(out["ccd_temp"].astype(float), ccd_temp)]
    return out
