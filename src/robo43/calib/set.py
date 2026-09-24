"""Load and cache calibration master frames for a reduction run.

The legacy reducer calls ``fits.getdata`` on the masters **for every science
frame**, re-reading them from disk repeatedly. This module loads all masters
once at the start of a run, passes the ``CalibrationSet`` around to workers,
and avoids redundant I/O.

The ``CalibrationSet`` instance is **picklable and cheap to send to a worker**:
it holds only NumPy arrays, Path objects, DarkEntry tuples, and a
GainMeasurementSet (itself plain floats/strs/ints) — never a logger,
file handle, Settings instance, or InstrumentProfile. Missing calibration files
are ``None``, never exceptions, following the graceful-degradation rule in §1B.5.

Note on ``dark_for()``: The interface contract specifies
``def dark_for(self, exptime, ccd_temp, *, settings) -> DarkSelection``, but
the actual implementation requires ``profile`` to delegate to ``select_dark()``.
Since ``CalibrationSet`` cannot store a profile (it must stay picklable),
``dark_for()`` takes ``profile`` as a required keyword-only parameter as well,
deviating from the contract signature. This is documented here and in the method.

``gain_measurements`` holds the night's runtime PTC gain measurements (see
``robo43.calib.gain``), one GainMeasurement per (readout_mode, gain_setting,
binning) group, loaded from ``gain_measurements.json`` in ``calibdir`` if
present. ``gain_for()`` looks up a frame's own group in it, returning
``None`` -- meaning "reduce this frame in ADU" -- when that group was never
measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from robo43.calib.dark import DarkEntry, DarkSelection, select_dark
from robo43.calib.gain import DEFAULT_GAIN_MEASUREMENTS_NAME, GainMeasurement, GainMeasurementSet
from robo43.calib.masks import MaskSet
from robo43.provenance import sha256_file

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

__all__ = ["CalibrationSet"]


def _read_master(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Read a pipeline-written master frame: the image is always HDU 0.

    Masters are this pipeline's own products, so they must never be read
    through the instrument profile's raw ``science_hdu`` (which is 1 for
    fpacked T80S frames, where HDU 1 of a master is its VAR plane).
    """
    with fits.open(path) as hdul:
        return np.array(hdul[0].data, dtype=np.float32), hdul[0].header.copy()


@dataclass(slots=True)
class CalibrationSet:
    """Cached collection of master calibration frames for one reduction run.

    All arrays are float32 and may be ``None`` when the corresponding master
    file is absent. Paths track the location each master was read from,
    enabling checksums and traceability.

    Attributes:
        bias: Master bias frame, or None if absent.
        bias_var: Variance of the master bias, or None if not available.
        bias_path: Path to the bias file, or None.
        flats: Dict mapping filter name to flat array.
        flat_var: Dict mapping filter name to flat variance array.
        flat_paths: Dict mapping filter name to flat file path.
        darks: Tuple of DarkEntry objects; empty tuple if no darks found.
        dark_current_map: Composite dark-current map (e-/s/pixel), or None.
        dark_current_path: Path to the dark-current map file, or None.
        mask: Static bad-pixel mask (MaskSet), or None if absent or unreadable.
        gain_measurements: The night's runtime PTC gain measurements, keyed by
            (readout_mode, gain_setting, binning); empty if none were measured
            or the cache file was absent.
    """

    bias: np.ndarray | None
    bias_var: np.ndarray | None
    bias_path: Path | None
    flats: dict[str, np.ndarray]
    flat_var: dict[str, np.ndarray]
    flat_paths: dict[str, Path]
    darks: tuple[DarkEntry, ...]
    dark_current_map: np.ndarray | None
    dark_current_path: Path | None
    mask: MaskSet | None
    gain_measurements: GainMeasurementSet
    bias_fallback: bool = False
    bias_source_nights: tuple[str, ...] = ()
    flat_fallback: dict[str, bool] = field(default_factory=dict)
    flat_source_nights: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        calibdir: str | Path | None,
        *,
        settings: Settings,  # noqa: ARG003
        profile: InstrumentProfile,
        filters: list[str] | None = None,
    ) -> CalibrationSet:
        """Load all available master calibration frames from a directory.

        Missing files are silently skipped; their corresponding fields are
        ``None`` or empty. Errors reading a file (e.g. corrupt FITS) are
        logged at WARNING and the master is skipped.

        Args:
            calibdir: Directory to search for masters. If None or a
                non-existent path, all fields return None/empty.
            settings: The full Settings tree (needed for profile keyword
                aliases when reading dark headers).
            profile: Instrument profile for keyword aliases and geometry.
            filters: If given, load flats only for these filter names.
                If None, load every flat found.

        Returns:
            A CalibrationSet with all discovered masters. Never raises.
        """
        bias = None
        bias_var = None
        bias_path = None
        flats: dict[str, np.ndarray] = {}
        flat_var: dict[str, np.ndarray] = {}
        flat_paths: dict[str, Path] = {}
        darks: list[DarkEntry] = []
        dark_current_map = None
        dark_current_path = None
        mask = None
        gain_measurements = GainMeasurementSet({})
        bias_fallback = False
        bias_source_nights: tuple[str, ...] = ()
        flat_fallback: dict[str, bool] = {}
        flat_source_nights: dict[str, tuple[str, ...]] = {}

        if calibdir is None:
            calibdir_path = None
        else:
            calibdir_path = Path(calibdir)
            if not calibdir_path.exists():
                calibdir_path = None

        if calibdir_path is not None:
            # Load bias
            bias_file = calibdir_path / "master_bias.fits"
            if bias_file.exists():
                try:
                    bias, header = _read_master(bias_file)
                    bias_path = bias_file
                    bias_fallback = bool(header.get("CALFBACK", False))
                    if bias_fallback:
                        bias_source_nights = tuple(str(header["BIASNGT"]).split(","))
                    # Try to load variance from VAR extension
                    try:
                        with fits.open(bias_file) as hdul:
                            if "VAR" in hdul:
                                bias_var = np.array(
                                    hdul["VAR"].data, dtype=np.float32
                                )
                    except Exception as e:
                        logger.debug(
                            "could not read VAR extension from %s: %s",
                            bias_file, e
                        )
                except Exception as e:
                    logger.warning("failed to read master bias %s: %s", bias_file, e)

            # Load flats
            for flat_file in sorted(calibdir_path.glob("master_flat_*.fits")):
                try:
                    filter_name = _parse_flat_filter_name(flat_file.name)
                    if filters is not None and filter_name not in filters:
                        continue

                    # Skip if we already have a better version (higher precedence value means worse)
                    if filter_name in flat_paths:
                        old_precedence = _flat_precedence(flat_paths[filter_name].name)
                        new_precedence = _flat_precedence(flat_file.name)
                        if new_precedence > old_precedence:
                            # New file has lower priority; skip and log if it's sky vs dome
                            old_name = flat_paths[filter_name].name
                            if "_sky.fits" in flat_file.name and "_dome.fits" in old_name:
                                logger.info(
                                    "skipping sky flat %s; dome flat exists for filter %s",
                                    flat_file.name,
                                    filter_name,
                                )
                            continue

                    flat_data, flat_header = _read_master(flat_file)
                    flats[filter_name] = flat_data
                    flat_paths[filter_name] = flat_file
                    if bool(flat_header.get("CALFBACK", False)):
                        flat_fallback[filter_name] = True
                        # A pooled master flat that combined nights from more
                        # than a handful of nights records FLATN1/FLATN2/
                        # FLATNNGT instead of FLATNGT, to stay within the
                        # FITS string-card length limit (see
                        # robo43.calib.flat._flat_fallback_cards); fall back
                        # to that compact form when the full list is absent,
                        # rather than raising and dropping this flat file.
                        if "FLATNGT" in flat_header:
                            flat_source_nights[filter_name] = tuple(
                                str(flat_header["FLATNGT"]).split(",")
                            )
                        elif "FLATN1" in flat_header and "FLATN2" in flat_header:
                            flat_source_nights[filter_name] = (
                                str(flat_header["FLATN1"]),
                                str(flat_header["FLATN2"]),
                            )
                        else:
                            flat_source_nights[filter_name] = ()
                    else:
                        flat_fallback[filter_name] = False

                    # Try to load variance
                    try:
                        with fits.open(flat_file) as hdul:
                            if "VAR" in hdul:
                                flat_var[filter_name] = np.array(
                                    hdul["VAR"].data, dtype=np.float32
                                )
                    except Exception as e:
                        logger.debug(
                            "could not read VAR extension from %s: %s",
                            flat_file, e
                        )

                except Exception as e:
                    logger.warning("failed to read flat %s: %s", flat_file, e)

            # Load darks
            for dark_file in sorted(calibdir_path.glob("master_dark_*.fits")):
                if dark_file.name.startswith("master_dark_current"):
                    continue
                try:
                    header = fits.getheader(dark_file, 0)
                    exptime = None
                    ccd_temp = None

                    # Read EXPTIME using profile aliases
                    exp_aliases = profile.aliases_for("exptime")
                    for alias in exp_aliases:
                        if alias in header:
                            exptime = float(header[alias])
                            break

                    if exptime is None:
                        logger.warning(
                            "skipping dark %s: no exptime found",
                            dark_file
                        )
                        continue

                    # Read CCD_TEMP using profile aliases
                    temp_aliases = profile.aliases_for("ccd_temp")
                    for alias in temp_aliases:
                        if alias in header:
                            ccd_temp = float(header[alias])
                            break

                    # Read NCOMBINE, default to 1
                    n_combine = int(header.get("NCOMBINE", 1))

                    # Load the data
                    dark_data, _header = _read_master(dark_file)

                    # Try to load variance
                    dark_var = None
                    try:
                        with fits.open(dark_file) as hdul:
                            if "VAR" in hdul:
                                dark_var = np.array(
                                    hdul["VAR"].data, dtype=np.float32
                                )
                    except Exception as e:
                        logger.debug(
                            "could not read VAR extension from %s: %s",
                            dark_file, e
                        )

                    entry = DarkEntry(
                        path=dark_file,
                        exptime=exptime,
                        ccd_temp=ccd_temp,
                        data=dark_data,
                        variance=dark_var,
                        n_combine=n_combine,
                    )
                    darks.append(entry)

                except Exception as e:
                    logger.warning("failed to read dark %s: %s", dark_file, e)

            # Load dark current map
            dc_file = calibdir_path / "dark_current_map.fits"
            if not dc_file.exists():
                dc_file = calibdir_path / "master_dark_current.fits"

            if dc_file.exists():
                try:
                    dark_current_map, _header = _read_master(dc_file)
                    dark_current_path = dc_file
                except Exception as e:
                    logger.warning(
                        "failed to read dark current map %s: %s", dc_file, e
                    )

            # Load mask
            mask_file = calibdir_path / "mask.fits"
            if mask_file.exists():
                try:
                    mask = MaskSet.read(mask_file)
                except Exception as e:
                    logger.warning("failed to read mask %s: %s", mask_file, e)
                    mask = None

            # Load gain measurements
            gain_file = calibdir_path / DEFAULT_GAIN_MEASUREMENTS_NAME
            if gain_file.exists():
                gain_measurements = GainMeasurementSet.read(gain_file)

        return cls(
            bias=bias,
            bias_var=bias_var,
            bias_path=bias_path,
            flats=flats,
            flat_var=flat_var,
            flat_paths=flat_paths,
            darks=tuple(darks),
            dark_current_map=dark_current_map,
            dark_current_path=dark_current_path,
            mask=mask,
            gain_measurements=gain_measurements,
            bias_fallback=bias_fallback,
            bias_source_nights=bias_source_nights,
            flat_fallback=flat_fallback,
            flat_source_nights=flat_source_nights,
        )

    def flat_for(self, filter_name: str) -> tuple[np.ndarray | None, Path | None]:
        """Return the flat array and path for a filter, or (None, None).

        Args:
            filter_name: The canonical filter name (e.g. 'R-Bessell').

        Returns:
            (array, path) if a flat exists for this filter, or (None, None).
        """
        if filter_name in self.flats:
            return (self.flats[filter_name], self.flat_paths[filter_name])
        return (None, None)

    def gain_for(
        self,
        readout_mode: str | None,
        gain_setting: float | None,
        binning: int | None,
    ) -> GainMeasurement | None:
        """Return the measured gain for a frame's own group, or ``None``.

        ``None`` means this ``(readout_mode, gain_setting, binning)`` group
        was never measured (no flats/biases at that setting, or fewer than
        the configured minimum) -- the caller must reduce the frame in ADU
        rather than invent a conversion factor.
        """
        return self.gain_measurements.get(readout_mode, gain_setting, binning)

    def tiles_for(
        self,
        readout_mode: str | None,
        gain_setting: float | None,
        binning: int | None,
    ) -> tuple[GainMeasurement, ...] | None:
        """Per-amplifier-tile measurements for a frame's own group, in
        tile-index order, or ``None`` when this group has no per-tile
        records (never measured, or measured before per-tile storage
        existed). ``None`` means "apply gain_for()'s frame-level summary
        uniformly across the whole frame", never "no gain" -- see
        GainMeasurementSet.tiles_for.
        """
        return self.gain_measurements.tiles_for(readout_mode, gain_setting, binning)

    def dark_for(
        self,
        exptime: float,
        ccd_temp: float | None,
        *,
        settings: Settings,
        profile: InstrumentProfile,
        gain: float | None = None,
    ) -> DarkSelection:
        """Choose and scale the best available dark for a frame.

        Delegates to ``robo43.calib.dark.select_dark()``. The ``profile``
        parameter is required here even though it is not in the interface
        contract, because the contract requires passing ``profile`` to
        ``select_dark``, and this class cannot store a profile (it must
        remain picklable).

        Args:
            exptime: Target exposure time in seconds.
            ccd_temp: Target CCD temperature in Celsius, or None.
            settings: Settings tree (for tolerance parameters).
            profile: Instrument profile (for gain and temperature aliases).
            gain: The reducing frame's own measured e-/ADU gain, already
                looked up by the caller via :meth:`gain_for` (``None`` when
                its group has no measurement), passed through to
                ``select_dark`` for its ``current_map`` branch.

        Returns:
            A DarkSelection with the chosen entry (or None) and scaling
            parameters. Never raises.
        """
        # Determine dark-current reference temperature from the map header
        dark_current_ref_temp = None
        if self.dark_current_map is not None and self.dark_current_path is not None:
            try:
                header = fits.getheader(self.dark_current_path, 0)
                temp_aliases = profile.aliases_for("ccd_temp")
                # Try to get DCTREF or fallback to ccd_temp aliases
                if "DCTREF" in header:
                    dark_current_ref_temp = float(header["DCTREF"])
                else:
                    for alias in temp_aliases:
                        if alias in header:
                            dark_current_ref_temp = float(header[alias])
                            break
            except Exception as e:
                logger.debug(
                    "could not read dark-current reference temp from %s: %s",
                    self.dark_current_path, e
                )

        # Handle empty darks + no dark current map gracefully
        if not self.darks and self.dark_current_map is None:
            return DarkSelection(
                entry=None,
                scale=0.0,
                method="none",
                reason="no darks available; bias-only correction",
            )

        return select_dark(
            self.darks,
            self.dark_current_map,
            dark_current_ref_temp,
            exptime,
            ccd_temp,
            settings=settings,
            profile=profile,
            gain=gain,
        )

    def checksums(self) -> dict[str, str]:
        """Return sha256 checksums for all loaded master files.

        Returns a dict mapping short labels to hex SHA256 digests. Labels are:
        - "bias" for the master bias
        - "flat_<filter>" for each flat
        - "dark_<exptime>" for each dark
        - "dark_current_map"
        - "mask"

        Entries are omitted if the corresponding file was not loaded.

        Returns:
            Dict[label, sha256_hex].
        """
        result: dict[str, str] = {}

        if self.bias_path is not None:
            result["bias"] = sha256_file(self.bias_path)

        for filter_name, path in self.flat_paths.items():
            result[f"flat_{filter_name}"] = sha256_file(path)

        for entry in self.darks:
            label = f"dark_{entry.exptime}"
            result[label] = sha256_file(entry.path)

        if self.dark_current_path is not None:
            result["dark_current_map"] = sha256_file(self.dark_current_path)

        # Note: mask doesn't have a path stored, so we omit it
        # (it's read-only inside mask.fits)
        # If needed, we'd need to store mask_path as well, but the interface
        # doesn't have that field. This is consistent with graceful degradation.

        return result


def _parse_flat_filter_name(filename: str) -> str:
    """Extract the filter name from a flat master filename.

    Handles:
    - master_flat_R-Bessell.fits -> "R-Bessell"
    - master_flat_R-Bessell_dome.fits -> "R-Bessell"
    - master_flat_R-Bessell_sky.fits -> "R-Bessell"
    - master_flat_Halpha_dome.fits -> "Halpha"

    The suffix (_dome or _sky) is separated from the right after stripping
    the master_flat_ prefix and .fits suffix, so a hyphenated filter name
    is preserved.

    Args:
        filename: The flat file basename (e.g. 'master_flat_R-Bessell_dome.fits').

    Returns:
        The canonical filter name (e.g. 'R-Bessell').
    """
    # Strip prefix and suffix
    stripped = filename[len("master_flat_") :]  # "R-Bessell_dome.fits"
    if stripped.endswith(".fits"):
        stripped = stripped[: -len(".fits")]  # "R-Bessell_dome"

    # Check if the remainder ends with _dome or _sky and strip it
    if stripped.endswith("_dome"):
        return stripped[: -len("_dome")]
    if stripped.endswith("_sky"):
        return stripped[: -len("_sky")]

    return stripped


def _flat_precedence(filename: str) -> int:
    """Return a precedence score for flat filename selection.

    Lower scores have higher precedence (are preferred).
    - master_flat_<filter>.fits (bare, no kind suffix): 0 (highest precedence)
    - master_flat_<filter>_dome.fits: 1
    - master_flat_<filter>_sky.fits: 2 (lowest precedence)

    Args:
        filename: Flat file basename.

    Returns:
        Precedence score (lower is better).
    """
    if filename.endswith("_dome.fits"):
        return 1
    if filename.endswith("_sky.fits"):
        return 2
    return 0  # bare name
