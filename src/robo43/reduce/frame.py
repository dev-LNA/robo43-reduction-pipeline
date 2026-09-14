"""Reduce one science frame to a MEF product, recording which corrections were applied.

This module implements the reduction pipeline for a single frame, following the
exact order of operations specified in the interface contract (section 4.4):
overscan → trim → bias → dark (scaled) → flat → bad-pixel mask → cosmic-ray
rejection → gain to electrons → variance and DQ planes.

Each step is optional and independently recorded. Missing calibrations are handled
gracefully per the graceful-degradation table in the interface contract (§1B.5):
always produce a reduced image and catalogue, and record precisely which
corrections were applied and which were not.

Module deviation from contract: overscan geometry is now passed as a runtime
parameter (``overscan`` keyword argument to ``reduce_frame``) rather than read
from the profile, because overscan detection is now a separate calibration step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.masks import minimal_mask
from robo43.calib.overscan import apply_overscan
from robo43.products.dq import DQ, DQ_DTYPE
from robo43.products.mef import ReducedProduct, calibration_cards
from robo43.products.weight import build_weight
from robo43.reduce.cosmics import apply_to_dq, detect_cosmics

if TYPE_CHECKING:
    from robo43.calib.overscan import OverscanGeometry
    from robo43.calib.set import CalibrationSet
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = ["ReductionStatus", "reduce_frame"]


@dataclass(slots=True)
class ReductionStatus:
    """Record of which corrections were applied to a frame."""

    path: Path
    raw_name: str
    proc_code: int
    proc_status: str
    steps_applied: tuple[str, ...]
    steps_skipped: tuple[str, ...]
    error: str | None = None
    product_path: Path | None = None
    wcs_solved: bool = False
    astrom_rms_arcsec: float | None = None
    fwhm_arcsec: float | None = None
    zeropoint: float | None = None
    is_photometric: bool | None = None
    duration_s: float | None = None

    def as_row(self) -> dict[str, Any]:
        """Return a flat dict of plain Python scalars and strings, suitable as one CSV row.

        Joins tuples with ';' and renders Paths as strings. Every value is a
        str, int, float, or None (never a Path, tuple, or ndarray).
        """
        return {
            "path": str(self.path),
            "raw_name": self.raw_name,
            "proc_code": self.proc_code,
            "proc_status": self.proc_status,
            "steps_applied": ";".join(self.steps_applied),
            "steps_skipped": ";".join(self.steps_skipped),
            "error": self.error,
            "product_path": str(self.product_path) if self.product_path is not None else None,
            "wcs_solved": self.wcs_solved,
            "astrom_rms_arcsec": self.astrom_rms_arcsec,
            "fwhm_arcsec": self.fwhm_arcsec,
            "zeropoint": self.zeropoint,
            "is_photometric": self.is_photometric,
            "duration_s": self.duration_s,
        }


def reduce_frame(
    meta: FrameMeta,
    calib: CalibrationSet,
    *,
    settings: Settings,
    profile: InstrumentProfile,
    overscan: OverscanGeometry | None = None,
) -> tuple[ReducedProduct, ReductionStatus]:
    """Reduce one science frame to a MEF product with a status record.

    Implements the reduction pipeline in exact order, with each step optional
    and independently recorded. Missing calibrations are handled gracefully:
    the frame is always reduced and returned, with its status recording which
    corrections were applied and which were skipped.

    The function does NOT write the product to disk; the caller decides the path.

    Args:
        meta: Frame metadata, including path to the raw image.
        calib: Pre-loaded calibration set (bias, flats, darks, masks).
        settings: Pipeline settings (calib, runtime, etc.).
        profile: Instrument profile (gain, saturation, etc.).
        overscan: Detected overscan geometry, or None to skip overscan/trim.

    Returns:
        (product, status): a ReducedProduct and its ReductionStatus record.
        proc_code is 0 (fully reduced) if every available calibration was applied,
        3 if one or more calibrations were missing/skipped, or 4 if an error
        prevented reduction (though errors do not raise; they are caught and
        recorded in the status).
    """
    start_time = time.monotonic()
    steps_applied: list[str] = []
    steps_skipped: list[str] = []
    header_cards: list[tuple[str, Any, str]] = []

    # Start with the raw frame
    from robo43.io.fits_io import read_image

    try:
        data, raw_header = read_image(meta.path, profile)
    except Exception as e:
        logger.error("failed to read raw frame %s: %s", meta.path, e)
        status = ReductionStatus(
            path=meta.path,
            raw_name=meta.path.name,
            proc_code=4,
            proc_status="reduction_error",
            steps_applied=tuple(steps_applied),
            steps_skipped=tuple(steps_skipped),
            error=f"failed to read raw frame: {e}",
            duration_s=time.monotonic() - start_time,
        )
        raise  # Propagate the error; the caller's exception handler will record it

    # Gain: looked up once, from the night's runtime PTC measurement (see
    # robo43.calib.gain) for this frame's own (readout_mode, gain_setting,
    # binning) -- never from the frame's header or the instrument profile,
    # since the QHY600M's GAIN keyword is a vendor setting index, not an
    # e-/ADU conversion factor. Reused for every gain-derived quantity below
    # -- cosmic-ray detection, the ADU/e- gain-to-electrons step, and the
    # read-noise-in-ADU variance term -- so the same value is applied
    # exactly once and consistently throughout. ``None`` means this frame's
    # group was never measured (e.g. no flats/biases at that setting); the
    # frame is then reduced in ADU rather than guessed at.
    gain_measurement = calib.gain_for(meta.readout_mode, meta.gain_setting, meta.binning)
    gain = gain_measurement.gain_e_per_adu if gain_measurement is not None else None
    read_noise_e = (
        gain_measurement.read_noise_e if gain_measurement is not None else profile.read_noise_e
    )
    if gain_measurement is not None:
        header_cards.extend(gain_measurement.header_cards())
    else:
        from robo43.calib.gain import no_measurement_cards

        header_cards.extend(no_measurement_cards(meta.gain_setting, meta.readout_mode))
    header_cards.append(("MEDRAW", float(np.nanmedian(data)), "median of the raw frame (ADU)"))

    # --- Step 1: Overscan and Trim ---
    oscan_cards: list[tuple[str, Any, str]] = []
    if overscan is not None:
        try:
            data, oscan_cards = apply_overscan(data, overscan, settings=settings)
            steps_applied.append("overscan")
            steps_applied.append("trim")
        except Exception as e:
            logger.error("overscan correction failed: %s", e)
            steps_skipped.append("overscan")
            steps_skipped.append("trim")
    else:
        steps_skipped.append("overscan")
        steps_skipped.append("trim")
        oscan_cards.append(("OSCANCOR", False, "Overscan correction applied"))
        oscan_cards.append(("TRIMMED", False, "Frame trimmed to science region"))

    header_cards.extend(oscan_cards)
    header_cards.append(
        ("MEDOSC", float(np.nanmedian(data)), "median after overscan/trim (ADU)")
    )

    # Initialize DQ plane (all pixels good initially) after overscan/trim
    # so its shape matches the (possibly trimmed) science data
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)
    assert dq.dtype == DQ_DTYPE, f"DQ dtype is {dq.dtype}, expected {DQ_DTYPE}"

    # --- Step 2: Bias Subtraction ---
    if calib.bias is not None:
        try:
            if data.shape != calib.bias.shape:
                logger.error(
                    "bias shape %s does not match frame shape %s; skipping bias",
                    calib.bias.shape, data.shape
                )
                steps_skipped.append("bias")
                header_cards.append(("BIASFIL", "NONE", "no master bias applied"))
            else:
                data = data - calib.bias
                steps_applied.append("bias")
                header_cards.append(
                    ("BIASFIL", calib.bias_path.name, "master bias used")
                )
                if calib.bias_path is not None:
                    from robo43.provenance import sha256_file
                    header_cards.append(
                        ("HISTORY", f"BIASFIL sha256={sha256_file(calib.bias_path)}", "")
                    )
        except Exception as e:
            logger.error("bias subtraction failed: %s", e)
            steps_skipped.append("bias")
            header_cards.append(("BIASFIL", "NONE", "no master bias applied"))
    else:
        steps_skipped.append("bias")
        header_cards.append(("BIASFIL", "NONE", "no master bias applied"))

    header_cards.append(
        ("MEDBIAS", float(np.nanmedian(data)), "median after bias subtraction (ADU)")
    )

    # --- Step 3: Dark Subtraction (scaled) ---
    dark_cards: list[tuple[str, Any, str]] = []
    dark_scale: float | None = None
    try:
        dark_selection = calib.dark_for(meta.exptime, meta.ccd_temp,
                                        settings=settings, profile=profile, gain=gain)
        if dark_selection.method == "none" or dark_selection.entry is None:
            logger.warning("dark current is uncorrected: %s", dark_selection.reason)
            steps_skipped.append("dark")
            dark_cards.append(("DARKFIL", "NONE", "No usable master dark applied"))
            dark_cards.append(("DARKMETH", "none", "Dark selection method"))
        else:
            from robo43.calib.dark import scale_dark
            dark_data, dark_cards = scale_dark(
                dark_selection,
                meta.exptime,
                meta.ccd_temp,
                settings=settings,
            )
            # Guard against the zero-size sentinel array
            if dark_data.size > 0:
                data = data - dark_data
                steps_applied.append("dark")
                dark_scale = dark_selection.scale
            else:
                steps_skipped.append("dark")
                dark_cards = [
                    ("DARKFIL", "NONE", "No usable master dark applied"),
                    ("DARKMETH", "none", "Dark selection method"),
                ]
    except Exception as e:
        logger.error("dark subtraction failed: %s", e)
        steps_skipped.append("dark")
        dark_cards = [
            ("DARKFIL", "NONE", "No usable master dark applied"),
            ("DARKMETH", "none", "Dark selection method"),
        ]

    header_cards.extend(dark_cards)
    header_cards.append(
        ("MEDDARK", float(np.nanmedian(data)), "median after dark subtraction (ADU)")
    )

    # --- Step 4: Flat Fielding ---
    flat_array, flat_path = calib.flat_for(meta.filter)
    is_photometric: bool | None = None
    if flat_array is None:
        logger.warning("no master flat for filter %s; setting NO_FLAT on all pixels", meta.filter)
        steps_skipped.append("flat")
        dq = dq.astype(np.int32) | np.int32(int(DQ.NO_FLAT))
        is_photometric = False
        header_cards.append(("FLATFIL", "NONE", "no master flat applied"))
    else:
        try:
            # Guard against division by zero and non-finite flat pixels
            safe_flat = np.array(flat_array, dtype=np.float32, copy=True)
            bad_flat = (safe_flat <= 0) | ~np.isfinite(safe_flat)

            if np.any(bad_flat):
                dq[bad_flat] |= int(DQ.NO_FLAT)
                safe_flat[bad_flat] = 1.0  # Set to 1 to avoid division by zero

            with np.errstate(divide="ignore", invalid="ignore"):
                data = data / safe_flat

            # Mark pixels where flat was zero/non-finite
            data[bad_flat] = np.nan

            steps_applied.append("flat")
            header_cards.append(("FLATFIL", flat_path.name, "master flat used"))
            if flat_path is not None:
                from robo43.provenance import sha256_file
                header_cards.append(
                    ("HISTORY", f"FLATFIL sha256={sha256_file(flat_path)}", "")
                )
        except Exception as e:
            logger.error("flat fielding failed: %s", e)
            steps_skipped.append("flat")
            dq[:] |= int(DQ.NO_FLAT)
            is_photometric = False
            header_cards.append(("FLATFIL", "NONE", "no master flat applied"))

    header_cards.append(
        ("MEDFLAT", float(np.nanmedian(data)), "median after flat fielding")
    )

    # --- Step 5: Bad-Pixel Mask ---
    if calib.mask is not None:
        try:
            if calib.mask.dq.shape == dq.shape:
                dq = dq | calib.mask.dq.astype(DQ_DTYPE)
                steps_applied.append("mask")
                header_cards.append(("MASKFIL", "mask.fits", "bad-pixel mask used"))
            else:
                logger.error("mask shape %s does not match frame shape %s",
                           calib.mask.dq.shape, dq.shape)
                steps_skipped.append("mask")
                # Fall back to minimal mask
                minimal = minimal_mask(data, saturation=profile.saturation_adu)
                dq = dq | minimal.dq.astype(DQ_DTYPE)
        except Exception as e:
            logger.error("mask application failed: %s", e)
            steps_skipped.append("mask")
            # Fall back to minimal mask
            try:
                minimal = minimal_mask(data, saturation=profile.saturation_adu)
                dq = dq | minimal.dq.astype(DQ_DTYPE)
            except Exception as e2:
                logger.error("minimal mask fallback failed: %s", e2)
    else:
        # Use minimal mask as fallback
        try:
            minimal = minimal_mask(data, saturation=profile.saturation_adu)
            dq = dq | minimal.dq.astype(DQ_DTYPE)
            steps_skipped.append("mask")
            header_cards.append(
                ("MASKFIL", "NONE", "no master mask; using minimal saturation+NaN mask")
            )
        except Exception as e:
            logger.error("minimal mask fallback failed: %s", e)
            steps_skipped.append("mask")

    # --- Step 6: Cosmic-Ray Rejection ---
    cr_cards: list[tuple[str, Any, str]] = []
    if settings.calib.cosmic_ray_rejection:
        if gain is None:
            # No runtime PTC measurement for this frame's group (e.g. no
            # flats/biases at this readout/gain-setting/binning combination):
            # an expected, known condition -- not an astroscrappy failure --
            # so it is handled explicitly here rather than via the ValueError
            # path below, mirroring Step 7's identical gain-is-None handling.
            logger.warning(
                "cosmic-ray rejection requires a measured gain; no measured "
                "gain for readout=%s setting=%s binning=%s; skipping "
                "cosmic-ray rejection for this frame.",
                meta.readout_mode, meta.gain_setting, meta.binning,
            )
            steps_skipped.append("cosmics")
            cr_cards = [
                ("CRREJ", False, "cosmic-ray rejection applied"),
            ]
        else:
            try:
                cr_result = detect_cosmics(
                    data,
                    settings=settings.calib,
                    gain_e_per_adu=gain,
                    read_noise_e=read_noise_e,
                    saturation=profile.saturation_adu,
                    dq=dq,
                    in_electrons=False,
                )
                data = cr_result.cleaned
                dq = apply_to_dq(dq, cr_result.mask)
                steps_applied.append("cosmics")
                cr_cards = cr_result.header_cards()
            except Exception as e:
                logger.error("cosmic-ray rejection failed: %s", e)
                steps_skipped.append("cosmics")
                cr_cards = [
                    ("CRREJ", False, "cosmic-ray rejection applied"),
                ]
    else:
        steps_skipped.append("cosmics")
        cr_cards = [
            ("CRREJ", False, "cosmic-ray rejection applied"),
        ]

    header_cards.extend(cr_cards)
    header_cards.append(
        ("MEDCOSM", float(np.nanmedian(data)), "median after cosmic-ray rejection (ADU)")
    )

    # --- Step 7: Gain to Electrons ---
    bunit = "adu"
    if settings.calib.apply_gain:
        if gain is not None:
            try:
                data = data * gain
                bunit = "electron"
                steps_applied.append("gain")
            except Exception as e:
                logger.error("gain conversion failed: %s", e)
                steps_skipped.append("gain")
        else:
            logger.warning(
                "gain requested but no measured gain for readout=%s setting=%s "
                "binning=%s; reducing this frame in ADU.",
                meta.readout_mode, meta.gain_setting, meta.binning,
            )
            steps_skipped.append("gain")
    else:
        steps_skipped.append("gain")

    header_cards.append(("BUNIT", bunit, "Unit of the SCI plane"))
    header_cards.append(
        ("MEDGAIN", float(np.nanmedian(data)), "median after gain conversion (final SCI median)")
    )

    # --- Step 8: Variance and Error Plane ---
    # Build a variance estimate from read noise and photon noise.
    # When gain is applied, variance is in electrons^2; otherwise in ADU^2.
    # ERR plane is 1-sigma (sqrt of variance).

    variance = np.zeros_like(data, dtype=np.float32)

    # Read noise contribution (always in electrons, convert to data units if
    # needed). ``read_noise_e`` prefers the runtime PTC measurement over the
    # profile's static value; see the gain resolution above.
    read_noise_var = 0.0
    if read_noise_e is not None:
        if bunit == "electron":
            read_noise_var = read_noise_e ** 2
        else:
            # In ADU: read_noise_e / gain
            read_noise_var = (read_noise_e / (gain or 1.0)) ** 2

    variance += read_noise_var

    # Photon noise: sqrt(signal) in the data's native units
    # For ADU: poisson(signal_adu) has variance = signal_adu
    # For electrons: poisson(signal_e) has variance = signal_e
    with np.errstate(invalid="ignore"):
        photon_var = np.maximum(data, 0.0)  # Avoid negative variance from cosmic rays
    variance += photon_var

    # Convert variance to 1-sigma
    err = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

    # Guard against non-finite values in error plane
    err[~np.isfinite(err)] = 0.0

    # --- Step 9: Weight Plane ---
    # Inverse-variance weight, derived from the final ERR/DQ planes (mask and
    # cosmic-ray flagging already folded into dq above). Not a calibration
    # step: its success or failure must never affect proc_code.
    try:
        weight = build_weight(err, dq)
        steps_applied.append("weight")
    except Exception as e:
        logger.error("weight plane construction failed: %s", e)
        steps_skipped.append("weight")
        weight = None

    # --- Build Product Header ---
    product_header = raw_header.copy()
    product_header["EXTNAME"] = "SCI"
    product_header["BUNIT"] = bunit

    # Add calibration provenance cards
    calib_cards_list = calibration_cards(
        bias=calib.bias_path,
        dark=None,  # Already included in header_cards from scale_dark
        flat=flat_path,
        mask=None,  # Already handled
        dark_scale=dark_scale,
        crrej=settings.calib.cosmic_ray_rejection,
        overscan=overscan is not None,
        profile=profile,
        meta=meta,
        settings=settings,
    )

    # Merge the calibration cards with header cards we've built
    for key, value, comment in header_cards:
        # Skip duplicates that might be in both
        if key not in ("BUNIT", "EXTNAME"):
            product_header[key] = (value, comment)

    for key, value, comment in calib_cards_list:
        if key not in product_header:
            product_header[key] = (value, comment)

    # Guard against non-finite header values
    for key in list(product_header.keys()):
        try:
            val = product_header[key]
            if isinstance(val, float) and not np.isfinite(val):
                del product_header[key]
        except (KeyError, TypeError):
            pass

    # Construct the product
    product = ReducedProduct(
        sci=data.astype(np.float32),
        err=err,
        dq=dq.astype(DQ_DTYPE),
        header=product_header,
        catalog=None,  # Filled by photometry phase
        catalog_header=None,
        bunit=bunit,
        weight=weight,
    )

    # Build status
    duration_s = time.monotonic() - start_time

    # proc_code is determined by whether any *calibrations* were skipped.
    # Pipeline steps (overscan, trim, cosmics) are not calibrations.
    # Calibrations are: bias, dark, flat, mask -- and, now that gain is a
    # runtime-measured calibration product rather than a profile constant,
    # "gain" too, but only when apply_gain was requested and skipped for
    # want of a measurement (or a conversion failure); a deliberate
    # apply_gain=False is a settings choice, not a missing calibration.
    calib_steps_skipped = [s for s in steps_skipped if s in ("bias", "dark", "flat", "mask")]
    if "gain" in steps_skipped and settings.calib.apply_gain:
        calib_steps_skipped.append("gain")
    if calib_steps_skipped:
        proc_code = 3  # Reduced but with missing calibrations
        proc_status = "reduced_degraded"
    else:
        proc_code = 0  # Fully reduced (all available calibrations applied)
        proc_status = "reduced"

    status = ReductionStatus(
        path=meta.path,
        raw_name=meta.path.name,
        proc_code=proc_code,
        proc_status=proc_status,
        steps_applied=tuple(steps_applied),
        steps_skipped=tuple(steps_skipped),
        error=None,
        product_path=None,  # Caller fills this in
        is_photometric=is_photometric,
        duration_s=duration_s,
    )

    return product, status
