"""Static per-night bad-pixel / hot-pixel DQ mask builder (plan section 5.1).

Builds an ``int32`` ``DQ`` plane from calibration products: a normalised
dark-current map, a master flat, and/or a bias-stack variance map. See
``robo43.products.dq`` for the bit encoding this module writes into.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std

from robo43.calib.combine import CombineResult
from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError
from robo43.products.dq import DQ, DQ_DTYPE, dq_bintable, dq_summary

logger = logging.getLogger(__name__)

__all__ = [
    "MaskSet",
    "build_mask",
    "detect_bad_columns",
    "minimal_mask",
    "nonlinearity_mask",
    "saturation_mask",
]


@dataclass(frozen=True, slots=True)
class MaskSet:
    """A static ``DQ`` plane plus its per-class fraction summary."""

    dq: np.ndarray  # int32, 2-D
    summary: dict[str, float]

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """FITS header cards summarising this mask, per-class and thresholds."""
        cards: list[tuple[str, Any, str]] = [("MASKVER", "1", "DQ mask format version")]
        if "hot" in self.summary:
            cards.append(("FRHOT", self.summary["hot"], "Fraction of pixels flagged HOT"))
        if "dead" in self.summary:
            cards.append(("FRDEAD", self.summary["dead"], "Fraction of pixels flagged DEAD"))
        if "unstable" in self.summary:
            cards.append(
                ("FRUNSTBL", self.summary["unstable"], "Fraction of pixels flagged UNSTABLE")
            )
        if "bad_column" in self.summary:
            cards.append(
                ("FRBADCOL", self.summary["bad_column"], "Fraction of pixels flagged BAD_COLUMN")
            )
        cards.append(("FRANY", self.summary["any"], "Fraction of pixels with any DQ flag set"))
        if "hot_nmad" in self.summary:
            cards.append(
                ("HOTNMAD", self.summary["hot_nmad"], "N*MAD threshold used for HOT detection")
            )
        if "dead_frac" in self.summary:
            cards.append(
                (
                    "DEADFRAC",
                    self.summary["dead_frac"],
                    "Fractional threshold used for DEAD detection",
                )
            )
        if "unstable_nmad" in self.summary:
            cards.append(
                (
                    "UNSTNMAD",
                    self.summary["unstable_nmad"],
                    "N*MAD threshold for UNSTABLE/BAD_COL",
                )
            )
        return cards

    def text_summary(self) -> str:
        """Human-readable multi-line summary of the flagged-pixel fractions."""
        threshold_keys = {"hot_nmad", "dead_frac", "unstable_nmad"}
        lines = ["DQ mask summary:"]
        other_lines = []
        any_line = None
        for name, fraction in self.summary.items():
            if name in threshold_keys or name == "fatal":
                continue
            count = round(fraction * self.dq.size)
            line = f"  {name}: {fraction * 100:.4f}% ({count} px)"
            if name == "any":
                any_line = line
            else:
                other_lines.append(line)
        lines.extend(other_lines)
        if any_line is not None:
            lines.append(any_line)
        return "\n".join(lines)

    def write(self, path: str | Path, *, cards: list[tuple[str, Any, str]] | None = None,
              overwrite: bool = False) -> Path:
        """Write this mask to a FITS file with ``DQ`` and ``CATFLAGS`` HDUs."""
        # TODO(io): route through robo43.io.fits_io.write_fits once available
        path = Path(path)
        if not overwrite and path.exists():
            msg = f"{path} exists and overwrite=False"
            raise CalibrationError(msg)

        primary = fits.PrimaryHDU()
        dq_hdu = fits.ImageHDU(data=self.dq.astype(DQ_DTYPE), name="DQ")
        all_cards = self.header_cards() + list(cards or [])
        for name, value, comment in all_cards:
            dq_hdu.header[name] = (value, comment)
        catflags_hdu = dq_bintable()
        hdulist = fits.HDUList([primary, dq_hdu, catflags_hdu])

        tmp_path = Path(str(path) + ".tmp")
        hdulist.writeto(tmp_path, overwrite=True)
        with tmp_path.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_path).replace(path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> MaskSet:
        """Read a mask previously written by :meth:`write`."""
        with fits.open(path) as hdul:
            dq = np.asarray(hdul["DQ"].data, dtype=DQ_DTYPE)
            header = hdul["DQ"].header

        full = dq_summary(dq)
        summary: dict[str, float] = {"any": full["any"], "fatal": full["fatal"]}
        for card, key in (
            ("FRHOT", "hot"),
            ("FRDEAD", "dead"),
            ("FRUNSTBL", "unstable"),
            ("FRBADCOL", "bad_column"),
        ):
            if card in header:
                summary[key] = full[key]
        for card, key in (
            ("HOTNMAD", "hot_nmad"),
            ("DEADFRAC", "dead_frac"),
            ("UNSTNMAD", "unstable_nmad"),
        ):
            if card in header:
                summary[key] = float(header[card])
        return cls(dq=dq, summary=summary)


def saturation_mask(data: np.ndarray, saturation: float) -> np.ndarray:
    """Pixels at or above ``saturation`` (inclusive)."""
    return data >= saturation


def nonlinearity_mask(data: np.ndarray, nonlinearity: float) -> np.ndarray:
    """Pixels at or above the non-linearity threshold (inclusive)."""
    return data >= nonlinearity


#: Half-width of the neighbourhood a line is compared against.  Small enough
#: that a smooth gradient contributes almost nothing over the window, wide
#: enough to give a stable median.
_NEIGHBOURHOOD = 4


def _leave_one_out_baseline(line_medians: np.ndarray) -> np.ndarray:
    """Median of each line's neighbours, excluding the line itself.

    This is the right comparison for finding a bad column, and it is chosen
    over the two obvious alternatives for concrete reasons:

    * A single global median is dominated by real large-scale structure --
      vignetting in a flat, a thermal gradient in a dark -- so the scatter it
      implies is far larger than the noise and a dead column stops standing
      out.  That is the same structure-versus-outlier confusion the plan
      identifies in the stack-combination statistics.
    * A running median filter, or a polynomial fit, has to say something about
      the region beyond the array ends.  Every edge mode gets that wrong here:
      a flattened baseline leaves the outermost lines with large structured
      residuals, which both hides a real defect a few lines in and invents
      false ones at the very edge.  Both failures were observed.

    Comparing a line only against neighbours that actually exist needs no
    extrapolation, so the edges are handled correctly by construction: at index
    0 the window is simply one-sided.  A genuine bad line is discrepant with
    its immediate neighbours; smooth structure is not.
    """
    n = line_medians.size
    baseline = np.empty_like(line_medians, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - _NEIGHBOURHOOD)
        hi = min(n, i + _NEIGHBOURHOOD + 1)
        neighbours = np.concatenate((line_medians[lo:i], line_medians[i + 1 : hi]))
        baseline[i] = np.median(neighbours) if neighbours.size else line_medians[i]
    return baseline


def detect_bad_columns(image: np.ndarray, settings: CalibSettings, *, axis: int = 0) -> np.ndarray:
    """Flag whole columns (``axis=0``) or rows (``axis=1``) deviating in the median.

    A numpy image is indexed ``[row, col]`` = ``[y, x]``, so a *column* is
    ``image[:, x]`` and its median is ``np.median(image, axis=0)``.

    The deviation is measured against a **running** median of the line medians
    rather than a single global one.  Real detectors carry smooth large-scale
    structure -- vignetting in a flat, a thermal gradient in a dark -- and that
    structure dominates the scatter of the line medians, inflating the
    threshold until a genuinely dead column no longer stands out.  Detrending
    first is the same distinction the plan draws in section 4.1 between real
    structure and per-pixel outliers: we are looking for a line that departs
    from its *neighbours*, not from the frame average.
    """
    if axis not in (0, 1):
        msg = f"detect_bad_columns: axis must be 0 or 1, got {axis!r}"
        raise CalibrationError(msg)

    line_medians = np.median(image, axis=axis)
    baseline = _leave_one_out_baseline(line_medians)

    # Judge the departure as a FRACTION of the local level.  An absolute,
    # scatter-based threshold does not work here: the residual against a local
    # baseline is dominated by the frame's own smooth curvature rather than by
    # noise, so its MAD is not a noise estimate at all -- on a low-noise frame
    # it collapses towards zero and the test flags every line with any
    # curvature.  The fractional departure separates the two cleanly.
    scale = np.maximum(np.abs(baseline), np.abs(np.median(line_medians)))
    with np.errstate(divide="ignore", invalid="ignore"):
        fractional = np.abs(line_medians - baseline) / scale
    fractional = np.where(np.isfinite(fractional), fractional, 0.0)

    if not np.any(scale > 0):
        # An identically-zero frame has no level to be a fraction of.
        return np.zeros(image.shape, dtype=bool)

    bad_lines = fractional > settings.bad_column_frac
    out = np.zeros(image.shape, dtype=bool)
    if axis == 0:
        out[:, bad_lines] = True
    else:
        out[bad_lines, :] = True
    return out


def _hot_pixel_mask(dark_current_map: np.ndarray, settings: CalibSettings) -> np.ndarray:
    med = np.median(dark_current_map)
    sigma = mad_std(dark_current_map)
    return dark_current_map > med + settings.hot_pixel_nmad * sigma


def _dead_pixel_mask(master_flat: np.ndarray, settings: CalibSettings) -> np.ndarray:
    norm = master_flat / np.median(master_flat)
    return norm < settings.dead_pixel_frac


def _unstable_pixel_mask(variance: np.ndarray, settings: CalibSettings) -> np.ndarray:
    med = np.median(variance)
    sigma = mad_std(variance)
    return variance > med + settings.unstable_pixel_nmad * sigma


def build_mask(
    *,
    dark_current_map: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
    bias_variance: np.ndarray | CombineResult | None = None,
    settings: CalibSettings,
    profile: Any | None = None,
    shape: tuple[int, int] | None = None,
) -> MaskSet:
    """Assemble a static ``DQ`` mask from whichever inputs are available."""
    if profile is not None:
        logger.debug("build_mask: profile=%r provided but not yet used", profile)

    variance = getattr(bias_variance, "variance", bias_variance)

    if dark_current_map is None and master_flat is None and variance is None and shape is None:
        msg = (
            "build_mask requires at least one of dark_current_map, master_flat, "
            "bias_variance, or shape"
        )
        raise CalibrationError(msg)

    array_shape: tuple[int, int] | None = None
    if dark_current_map is not None:
        array_shape = dark_current_map.shape
    elif master_flat is not None:
        array_shape = master_flat.shape
    elif variance is not None:
        array_shape = variance.shape

    if array_shape is not None and shape is not None and tuple(shape) != tuple(array_shape):
        msg = f"shape {tuple(shape)} does not match input array shape {tuple(array_shape)}"
        raise CalibrationError(msg)

    working_shape = array_shape if array_shape is not None else shape
    assert working_shape is not None

    named_arrays: list[tuple[str, np.ndarray]] = []
    if dark_current_map is not None:
        named_arrays.append(("dark_current_map", dark_current_map))
    if master_flat is not None:
        named_arrays.append(("master_flat", master_flat))
    if variance is not None:
        named_arrays.append(("bias_variance", variance))

    if len(named_arrays) > 1:
        ref_name, ref_arr = named_arrays[0]
        for name, arr in named_arrays[1:]:
            if arr.shape != ref_arr.shape:
                msg = f"{name} shape {arr.shape} does not match {ref_name} shape {ref_arr.shape}"
                raise CalibrationError(msg)

    if dark_current_map is None and master_flat is None and variance is None:
        dq = np.zeros(working_shape, dtype=DQ_DTYPE)
        return MaskSet(dq=dq, summary={"any": 0.0, "fatal": 0.0})

    dq = np.zeros(working_shape, dtype=DQ_DTYPE)

    hot_ran = dead_ran = unstable_ran = bad_column_ran = False

    if dark_current_map is not None:
        hot = _hot_pixel_mask(dark_current_map, settings)
        dq[hot] |= int(DQ.HOT)
        hot_ran = True

    if master_flat is not None:
        dead = _dead_pixel_mask(master_flat, settings)
        dq[dead] |= int(DQ.DEAD)
        dead_ran = True

    if variance is not None:
        unstable = _unstable_pixel_mask(variance, settings)
        dq[unstable] |= int(DQ.UNSTABLE)
        unstable_ran = True

    # Run the column detector over EVERY available source and OR the results.
    # A persistent column can show up as elevated dark current, as depressed
    # flat response, or as excess bias variance, and it need not show up in all
    # three -- so preferring one source and ignoring the others silently misses
    # real defects.
    column_sources = [
        ("dark_current_map", dark_current_map),
        ("master_flat", master_flat),
        ("bias_variance", variance),
    ]
    for name, source in column_sources:
        if source is None:
            continue
        bad_col = detect_bad_columns(source, settings, axis=0)
        n_bad = int(bad_col.any(axis=0).sum())
        if n_bad:
            logger.info("detect_bad_columns: %d column(s) flagged from %s", n_bad, name)
        dq[bad_col] |= int(DQ.BAD_COLUMN)
        bad_column_ran = True

    full = dq_summary(dq)
    summary: dict[str, float] = {"any": full["any"], "fatal": full["fatal"]}
    if hot_ran:
        summary["hot"] = full["hot"]
        summary["hot_nmad"] = settings.hot_pixel_nmad
    if dead_ran:
        summary["dead"] = full["dead"]
        summary["dead_frac"] = settings.dead_pixel_frac
    if unstable_ran:
        summary["unstable"] = full["unstable"]
    if bad_column_ran:
        summary["bad_column"] = full["bad_column"]
    if unstable_ran or bad_column_ran:
        summary["unstable_nmad"] = settings.unstable_pixel_nmad

    return MaskSet(dq=dq, summary=summary)


def minimal_mask(data: np.ndarray, *, saturation: float | None = None) -> MaskSet:
    """Graceful-degradation mask: only ``NO_DATA`` and optionally ``SATURATED``."""
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)

    no_data = ~np.isfinite(data)
    dq[no_data] |= int(DQ.NO_DATA)

    saturated_ran = saturation is not None
    if saturation is not None:
        sat = saturation_mask(data, saturation)
        dq[sat] |= int(DQ.SATURATED)

    full = dq_summary(dq)
    summary: dict[str, float] = {
        "any": full["any"],
        "fatal": full["fatal"],
        "no_data": full["no_data"],
    }
    if saturated_ran:
        summary["saturated"] = full["saturated"]

    return MaskSet(dq=dq, summary=summary)
