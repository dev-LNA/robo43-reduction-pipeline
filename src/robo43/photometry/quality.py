"""Configurable source-rejection chain for SExtractor photometry catalogues.

This module replaces the ad hoc quality filtering in
``src/process_robo43_frames.py:627-634``::

    mask = (sources['FLAGS'] < 4)
    mask &= (sources['FWHM_IMAGE'] > np.percentile(...))
    mask = (sources['FLUX_AUTO'] > np.percentile(sources['FLUX_AUTO'], 80))

That code has two bugs: the third line uses ``=`` instead of ``&=``, silently
discarding the ``FLAGS`` and ``FWHM_IMAGE`` cuts before it, and the surviving
80th-percentile flux cut throws away 80% of every field regardless of the
actual quality of the discarded sources -- a purely relative, data-dependent
threshold rather than a physically motivated one.

This module replaces that with a fixed sequence of absolute, config-driven
cuts. Every cut is recorded as a bit in a single ``Reject`` bitmask column
that is retained on every source -- nothing here deletes rows from the input
table (``reference_sample`` is an explicit, separate convenience filter, not
part of the rejection chain itself).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from robo43.exceptions import PhotometryError

if TYPE_CHECKING:
    from astropy.table import Table
    from astropy.wcs import WCS

    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO", "FLAGS", "FWHM_IMAGE")

# SExtractor/PSFEx convention: SPREAD_MODEL near 0 marks a point source.
_SPREAD_MODEL_STELLAR_LIMIT = 0.002

# Not present in PhotometrySettings; chosen as a reasonable default for a
# Gaia DR3 cross-match. Flagged as a possible missing config field.
_GAIA_MATCH_RADIUS_ARCSEC = 2.0


class Reject(enum.IntFlag):
    NONE = 0
    FLAGS = 1 << 0  # SExtractor FLAGS above the allowed value
    BADPIX = 1 << 1  # IMAFLAGS_ISO != 0 (hot/dead/unstable/CR/saturated in isophote)
    SATURATED = 1 << 2  # FLUX_MAX above saturation_fraction * saturation
    NONLINEAR = 1 << 3  # FLUX_MAX above nonlinearity_fraction * saturation
    EDGE = 1 << 4  # centroid within max(edge_margin_fwhm*FWHM, edge_margin_px)
    FWHM_LOW = 1 << 5  # narrower than fwhm_min_arcsec (cosmic-ray-like)
    FWHM_HIGH = 1 << 6  # wider than fwhm_max_arcsec (blended/extended)
    ELONGATED = 1 << 7  # ellipticity above ellipticity_max (trailed)
    LOW_SNR = 1 << 8  # FLUX_AUTO/FLUXERR_AUTO below snr_min
    NOT_STELLAR = 1 << 9  # CLASS_STAR below class_star_min
    BLENDED = 1 << 10  # neighbour within isolation_radius_arcsec
    NO_GAIA_MATCH = 1 << 11  # no Gaia DR3 counterpart


@dataclass(frozen=True, slots=True)
class QualityReport:
    mask: np.ndarray
    counts: dict[str, int]
    n_input: int
    n_clean: int
    skipped: tuple[str, ...] = ()
    median_fwhm_px: float | None = None
    median_fwhm_arcsec: float | None = None
    median_ellipticity: float | None = None
    spurious_rate: float | None = None

    def clean_index(self, *, allow: Reject = Reject.NONE) -> np.ndarray:
        return np.flatnonzero((self.mask & ~int(allow)) == 0)

    def text_summary(self) -> str:
        """Human-readable multi-line attrition report, e.g. for logs/debugging."""
        lines = [
            f"{self.n_input} sources: {self.n_clean} clean, "
            f"{self.n_input - self.n_clean} rejected"
        ]
        for member in Reject:
            if member is Reject.NONE:
                continue
            name = member.name
            if name is None or name not in self.counts:
                continue
            lines.append(
                f"  {name:<13} {self.counts[name]:5d} tripped "
                f"({self.counts[name + '_new']:5d} newly)"
            )
        seen: set[str] = set()
        for name in self.skipped:
            if name in seen:
                continue
            seen.add(name)
            lines.append(f"  {name:<13} skipped")
        if self.median_fwhm_px is not None:
            fwhm_arcsec = (
                f"{self.median_fwhm_arcsec:.3f} arcsec"
                if self.median_fwhm_arcsec is not None
                else "n/a"
            )
            ellipticity = (
                f"{self.median_ellipticity:.3f}" if self.median_ellipticity is not None else "n/a"
            )
            spurious = (
                f"{self.spurious_rate:.3f}" if self.spurious_rate is not None else "n/a"
            )
            lines.append(
                f"median FWHM: {self.median_fwhm_px:.3f} px ({fwhm_arcsec}), "
                f"median ellipticity: {ellipticity}, spurious rate: {spurious}"
            )
        return "\n".join(lines)

    def metric_cards(self) -> dict[str, int]:
        return {
            "NSOURCES": self.n_input,
            "NCLEAN": self.n_clean,
            "NREJECT": self.n_input - self.n_clean,
            "NSATUR": self.counts.get("SATURATED", 0),
            "NBADPIX": self.counts.get("BADPIX", 0),
        }


def _check_required_columns(table: Table) -> None:
    for name in REQUIRED_COLUMNS:
        if name not in table.colnames:
            msg = f"missing required column: {name!r}"
            raise PhotometryError(msg)


def _saturation_level(
    meta: FrameMeta,
    profile: InstrumentProfile,
    settings: Settings,
    *,
    level_adu: float | None,
) -> float | None:
    """Convert an ADU saturation/nonlinearity level to the data's own units.

    ``level_adu`` is always in ADU (from the header/profile). The
    catalogue's ``FLUX_MAX`` is in electrons when ``settings.calib.apply_gain``
    is True, ADU otherwise. This converts ADU to the data's own units
    exactly once, returning ``None`` when the level -- or, for the
    gain-applied case, the gain itself -- cannot be determined.
    """
    if level_adu is None:
        return None
    if not settings.calib.apply_gain:
        return level_adu
    gain = meta.gain if meta.gain is not None else profile.gain_e_per_adu
    if gain is None:
        return None
    return level_adu * gain


def _sky_columns(table: Table) -> tuple[str, str] | None:
    for ra_name, dec_name in (
        ("ALPHA_J2000", "DELTA_J2000"),
        ("RA", "DEC"),
        ("ra", "dec"),
        ("ra_deg", "dec_deg"),
    ):
        if ra_name in table.colnames and dec_name in table.colnames:
            return ra_name, dec_name
    return None


# Documented addition beyond the written contract: `stellar` distinguishes a
# strict astrometric-reference catalogue (FLAGS==0, NOT_STELLAR active, the
# safer default) from a science catalogue (`stellar=False`), which the
# contract's signature has no way to express.
def assess(
    table: Table,
    *,
    settings: Settings,
    profile: InstrumentProfile,
    meta: FrameMeta,
    dq=None,  # noqa: ARG001 -- reserved for future per-pixel checks
    wcs: WCS | None = None,
    reference: Table | None = None,
    stellar: bool = True,
) -> QualityReport:
    # `dq` and `wcs` are reserved for future per-pixel / per-coordinate
    # checks; `wcs` is used opportunistically by rule 11 below, `dq` is
    # currently unused by any rule.
    _check_required_columns(table)

    ps = settings.photometry
    n_input = len(table)

    if n_input == 0:
        return QualityReport(
            mask=np.array([], dtype=np.int32),
            counts={},
            n_input=0,
            n_clean=0,
        )

    mask = np.zeros(n_input, dtype=np.int32)
    counts: dict[str, int] = {}
    skipped: list[str] = []

    x = np.asarray(table["X_IMAGE"], dtype=float)
    y = np.asarray(table["Y_IMAGE"], dtype=float)
    flux_auto = np.asarray(table["FLUX_AUTO"], dtype=float)
    fluxerr_auto = np.asarray(table["FLUXERR_AUTO"], dtype=float)
    flags = np.asarray(table["FLAGS"])
    fwhm_image = np.asarray(table["FWHM_IMAGE"], dtype=float)

    positive_fwhm = fwhm_image[fwhm_image > 0]
    median_fwhm_px = float(np.median(positive_fwhm)) if len(positive_fwhm) else None
    median_fwhm_arcsec = (
        median_fwhm_px * meta.pixel_scale
        if median_fwhm_px is not None and meta.pixel_scale is not None
        else None
    )
    median_ellipticity: float | None = None
    spurious_rate: float | None = None

    def _apply(name: str, tripped: np.ndarray, bit: Reject) -> None:
        nonlocal mask
        clean_before = mask == Reject.NONE
        newly = int(np.count_nonzero(tripped & clean_before))
        total = int(np.count_nonzero(tripped))
        mask[tripped] |= int(bit)
        counts[name] = total
        counts[f"{name}_new"] = newly
        logger.info("quality cut %s: %d tripped, %d newly rejected", name, total, newly)

    def _skip(*names: str, reason: str) -> None:
        for name in names:
            skipped.append(name)
            logger.info("quality cut %s: skipped: %s", name, reason)

    # 1. FLAGS
    tripped = flags != 0 if stellar else flags > 3
    _apply("FLAGS", tripped, Reject.FLAGS)

    # 2. BADPIX
    if "IMAFLAGS_ISO" in table.colnames:
        imaflags = np.asarray(table["IMAFLAGS_ISO"])
        tripped = imaflags != 0
        _apply("BADPIX", tripped, Reject.BADPIX)
    else:
        _skip("BADPIX", reason="IMAFLAGS_ISO column absent")

    # 3. SATURATED / 4. NONLINEAR
    if "FLUX_MAX" in table.colnames:
        flux_max = np.asarray(table["FLUX_MAX"], dtype=float)

        sat_level_adu = meta.saturation if meta.saturation is not None else profile.saturation_adu
        sat_level = _saturation_level(meta, profile, settings, level_adu=sat_level_adu)
        if sat_level is not None:
            tripped = flux_max > ps.saturation_fraction * sat_level
            _apply("SATURATED", tripped, Reject.SATURATED)
        else:
            _skip("SATURATED", reason="saturation level could not be determined")

        nonlin_level = _saturation_level(
            meta, profile, settings, level_adu=profile.nonlinearity_adu
        )
        if nonlin_level is not None:
            tripped = flux_max > ps.nonlinearity_fraction * nonlin_level
            _apply("NONLINEAR", tripped, Reject.NONLINEAR)
        else:
            _skip("NONLINEAR", reason="nonlinearity level could not be determined")
    else:
        _skip("SATURATED", "NONLINEAR", reason="FLUX_MAX column absent")

    # 5. EDGE
    margin = max(ps.edge_margin_fwhm * (median_fwhm_px or 0.0), ps.edge_margin_px)
    tripped = (
        (x < margin) | (x > meta.naxis1 - margin) | (y < margin) | (y > meta.naxis2 - margin)
    )
    _apply("EDGE", tripped, Reject.EDGE)

    # 6. FWHM_LOW / FWHM_HIGH
    if meta.pixel_scale is not None:
        fwhm_arcsec = fwhm_image * meta.pixel_scale
        tripped_low = fwhm_arcsec < ps.fwhm_min_arcsec
        _apply("FWHM_LOW", tripped_low, Reject.FWHM_LOW)
        tripped_high = fwhm_arcsec > ps.fwhm_max_arcsec
        _apply("FWHM_HIGH", tripped_high, Reject.FWHM_HIGH)
    else:
        _skip("FWHM_LOW", "FWHM_HIGH", reason="meta.pixel_scale is None")

    # 7. ELONGATED
    if "ELLIPTICITY" in table.colnames:
        ellipticity = np.asarray(table["ELLIPTICITY"], dtype=float)
    elif "ELONGATION" in table.colnames:
        elongation = np.asarray(table["ELONGATION"], dtype=float)
        ellipticity = 1 - 1 / elongation
    else:
        ellipticity = None

    if ellipticity is not None:
        median_ellipticity = float(np.median(ellipticity))
        tripped = ellipticity > ps.ellipticity_max
        _apply("ELONGATED", tripped, Reject.ELONGATED)
    else:
        _skip("ELONGATED", reason="ELLIPTICITY/ELONGATION columns absent")

    # 8. LOW_SNR
    bad_err = fluxerr_auto <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(bad_err, 0.0, flux_auto / np.where(bad_err, 1.0, fluxerr_auto))
    tripped = bad_err | (snr < ps.snr_min)
    _apply("LOW_SNR", tripped, Reject.LOW_SNR)

    # 9. NOT_STELLAR
    if stellar:
        if "SPREAD_MODEL" in table.colnames:
            spread_model = np.asarray(table["SPREAD_MODEL"], dtype=float)
            tripped = np.abs(spread_model) > _SPREAD_MODEL_STELLAR_LIMIT
            _apply("NOT_STELLAR", tripped, Reject.NOT_STELLAR)
        elif "CLASS_STAR" in table.colnames:
            class_star = np.asarray(table["CLASS_STAR"], dtype=float)
            tripped = class_star < ps.class_star_min
            _apply("NOT_STELLAR", tripped, Reject.NOT_STELLAR)
        else:
            _skip("NOT_STELLAR", reason="CLASS_STAR/SPREAD_MODEL columns absent")

    # 10. BLENDED
    if meta.pixel_scale is not None:
        radius_px = ps.isolation_radius_arcsec / meta.pixel_scale
        points = np.column_stack([x, y])
        if len(points) > 1:
            tree = cKDTree(points)
            dist, _idx = tree.query(points, k=2)
            nearest = dist[:, 1]
        else:
            nearest = np.full(len(points), np.inf)
        tripped = nearest < radius_px
        _apply("BLENDED", tripped, Reject.BLENDED)
    else:
        _skip("BLENDED", reason="meta.pixel_scale is None")

    # 11. NO_GAIA_MATCH
    if reference is not None:
        src_cols = _sky_columns(table)
        ref_cols = _sky_columns(reference)
        if src_cols is None and wcs is not None:
            src_sky = wcs.pixel_to_world(x, y)
        elif src_cols is not None:
            ra_name, dec_name = src_cols
            src_sky = SkyCoord(
                ra=np.asarray(table[ra_name], dtype=float) * u.deg,
                dec=np.asarray(table[dec_name], dtype=float) * u.deg,
            )
        else:
            src_sky = None

        if src_sky is not None and ref_cols is not None:
            ra_name, dec_name = ref_cols
            ref_sky = SkyCoord(
                ra=np.asarray(reference[ra_name], dtype=float) * u.deg,
                dec=np.asarray(reference[dec_name], dtype=float) * u.deg,
            )
            _idx, sep2d, _dist3d = src_sky.match_to_catalog_sky(ref_sky)
            tripped = sep2d.arcsec > _GAIA_MATCH_RADIUS_ARCSEC
            _apply("NO_GAIA_MATCH", tripped, Reject.NO_GAIA_MATCH)
            spurious_rate = counts.get("NO_GAIA_MATCH", 0) / n_input if n_input else None
        elif ref_cols is None:
            logger.warning("reference table supplied but has no usable RA/Dec columns")
            _skip("NO_GAIA_MATCH", reason="reference table has no usable RA/Dec columns")
        else:
            _skip("NO_GAIA_MATCH", reason="no sky coordinates available for source table")
    else:
        _skip("NO_GAIA_MATCH", reason="reference is None")

    n_clean = int(np.count_nonzero(mask == Reject.NONE))

    return QualityReport(
        mask=mask,
        counts=counts,
        n_input=n_input,
        n_clean=n_clean,
        skipped=tuple(skipped),
        median_fwhm_px=median_fwhm_px,
        median_fwhm_arcsec=median_fwhm_arcsec,
        median_ellipticity=median_ellipticity,
        spurious_rate=spurious_rate,
    )


def reference_sample(table: Table, report: QualityReport, *, settings: Settings) -> Table:
    # `settings` is accepted per the contract but unused here.
    del settings
    return table[(table["FLAGS"] == 0) & (report.mask == int(Reject.NONE))]
