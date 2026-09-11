"""Per-target, per-filter averaging of tidy photometry into light-curve epochs.

This module consumes a "tidy" long-format DataFrame -- one row per
(target, filter, epoch) measurement -- with the required columns
``target``, ``filter``, ``jd``, ``mag``, ``mag_err``. Producing that tidy
CSV format is the job of ``robo43.photometry.aperture``; this module only
consumes it.

Science note: averaging magnitudes is not the same as averaging in flux
space. Because the magnitude scale is a nonlinear (logarithmic) transform
of flux, the magnitude-space mean of noisy points is statistically biased
relative to the flux-space inverse-variance-weighted mean (a
Jensen's-inequality-style effect). For that reason ``"weighted_mean"`` and
``"sigma_clipped_mean"`` combine measurements in flux space via
:func:`combine_in_flux_space`, and the naive magnitude-space
``"median"`` method is offered only for backwards comparison with the
legacy ``calculate_average_mags.py`` script, not recommended for new use.

This module replaces the legacy ``calculate_average_mags.py`` script and
fixes three bugs present there:

- Comparing an individual row's magnitude to a median-derived magnitude
  with ``==`` (dead code, since floats essentially never compare equal).
- A ``jd`` local variable leaking across loop iterations when a filter
  had no data, silently reusing a previous filter's value.
- Building the output DataFrame from a dict of independently-appended-to
  lists, which produced ragged (differently-sized) columns when a
  filter's data was skipped partway through the loop.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

from robo43.exceptions import PhotometryError

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("target", "filter", "jd", "mag", "mag_err")

_METHODS = ("median", "weighted_mean", "sigma_clipped_mean")

_OUTPUT_COLUMNS = ("target", "filter", "jd", "mag", "mag_err", "n_points", "method")


@dataclasses.dataclass(frozen=True, slots=True)
class EpochMagnitude:
    """One averaged (target, filter) epoch."""

    target: str
    filter: str
    jd: float
    mag: float
    mag_err: float
    n_points: int
    method: str


def combine_in_flux_space(mags: ArrayLike, mag_errs: ArrayLike) -> tuple[float, float]:
    """Inverse-variance-weighted combination of magnitudes in flux space.

    Averaging in magnitude space is not equivalent to averaging in flux
    space: the log transform is nonlinear, so the magnitude-space mean of
    noisy points is biased relative to the flux-space weighted mean. This
    function converts to relative flux, propagates errors, computes the
    inverse-variance-weighted flux mean, and converts back to magnitude.
    """
    mags_arr = np.asarray(mags, dtype=float)
    mag_errs_arr = np.asarray(mag_errs, dtype=float)

    flux = 10 ** (-0.4 * mags_arr)
    flux_err = flux * np.log(10) / 2.5 * mag_errs_arr

    weights = 1.0 / flux_err**2
    flux_mean = np.sum(weights * flux) / np.sum(weights)
    flux_mean_err = np.sqrt(1.0 / np.sum(weights))

    mag_mean = -2.5 * np.log10(flux_mean)
    mag_mean_err = 2.5 / np.log(10) * flux_mean_err / flux_mean

    return float(mag_mean), float(mag_mean_err)


def average_magnitudes(
    frame: pd.DataFrame,
    *,
    method: str = "sigma_clipped_mean",
    sigma: float = 3.0,
    min_points: int = 1,
) -> pd.DataFrame:
    """Average a tidy photometry frame into one row per (target, filter).

    Parameters
    ----------
    frame:
        Tidy long-format DataFrame with at least ``REQUIRED_COLUMNS``.
    method:
        One of ``"median"``, ``"weighted_mean"``, ``"sigma_clipped_mean"``.
    sigma:
        Sigma-clipping threshold, used only for ``"sigma_clipped_mean"``.
    min_points:
        Minimum number of surviving rows required to emit a row for a
        (target, filter) group; groups with fewer are skipped (with a
        warning) rather than producing a ragged/partial row.
    """
    if method not in _METHODS:
        msg = f"Unknown averaging method {method!r}; expected one of {_METHODS}"
        raise PhotometryError(msg)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"Frame is missing required columns: {missing}"
        raise PhotometryError(msg)

    results: list[EpochMagnitude] = []

    for (target, filt), group in frame.groupby(["target", "filter"], sort=False):
        surviving = group.dropna(subset=["mag", "mag_err", "jd"])

        if len(surviving) < min_points:
            logger.warning(
                "Skipping target=%r filter=%r: only %d valid points (need >= %d)",
                target,
                filt,
                len(surviving),
                min_points,
            )
            continue

        if method == "weighted_mean":
            mag, mag_err = combine_in_flux_space(
                surviving["mag"].to_numpy(), surviving["mag_err"].to_numpy()
            )
            n_points = len(surviving)
            jd = float(surviving["jd"].mean())

        elif method == "sigma_clipped_mean":
            clip_result = sigma_clip(surviving["mag"].to_numpy(), sigma=sigma, masked=True)
            keep_mask = ~clip_result.mask
            kept = surviving.loc[keep_mask]

            if len(kept) < min_points:
                logger.warning(
                    "Skipping target=%r filter=%r: only %d points survived "
                    "sigma-clipping (need >= %d)",
                    target,
                    filt,
                    len(kept),
                    min_points,
                )
                continue

            mag, mag_err = combine_in_flux_space(
                kept["mag"].to_numpy(), kept["mag_err"].to_numpy()
            )
            n_points = len(kept)
            jd = float(kept["jd"].mean())

        else:  # method == "median"
            mag = float(surviving["mag"].median())
            mag_err = float(surviving["mag_err"].median())
            n_points = len(surviving)
            jd = float(surviving["jd"].mean())

            nearest_idx = (surviving["mag"] - mag).abs().idxmin()
            logger.debug(
                "target=%r filter=%r: observation closest to the group's "
                "median magnitude is jd=%s",
                target,
                filt,
                surviving.loc[nearest_idx, "jd"],
            )

        results.append(
            EpochMagnitude(
                target=target,
                filter=filt,
                jd=jd,
                mag=mag,
                mag_err=mag_err,
                n_points=n_points,
                method=method,
            )
        )

    if not results:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    return pd.DataFrame([dataclasses.asdict(e) for e in results])[list(_OUTPUT_COLUMNS)]


def read_photometry_csvs(
    paths: Iterable[str | os.PathLike[str]], *, missing_ok: bool = True
) -> pd.DataFrame:
    """Read and concatenate several tidy photometry CSV files.

    Each successfully read file gets a ``source_path`` column identifying
    where it came from before being concatenated with the others. Missing
    files, unreadable files, or files lacking ``REQUIRED_COLUMNS`` are
    skipped with a warning when ``missing_ok`` is True, or raise
    ``PhotometryError`` otherwise.
    """
    frames: list[pd.DataFrame] = []

    for path in paths:
        if not Path(path).exists():
            msg = f"Photometry file not found: {path}"
            if missing_ok:
                logger.warning(msg)
                continue
            raise PhotometryError(msg)

        try:
            data = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            msg = f"Could not read photometry file {path}: {exc}"
            if missing_ok:
                logger.warning(msg)
                continue
            raise PhotometryError(msg) from exc

        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            msg = f"Photometry file {path} is missing required columns: {missing}"
            if missing_ok:
                logger.warning(msg)
                continue
            raise PhotometryError(msg)

        data = data.copy()
        data["source_path"] = str(path)
        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "source_path"])

    return pd.concat(frames, ignore_index=True)
