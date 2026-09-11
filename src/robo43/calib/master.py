"""Shared ``MasterFrame`` container for combined calibration frames.

Every master calibration frame (bias, dark, dark-current map, flat) is
assembled into a :class:`MasterFrame` before being written to disk, so the
FITS layout (SCI + optional VAR/WEIGHT/DQ extensions) is defined in exactly
one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from robo43.exceptions import CalibrationError
from robo43.io.fits_io import write_fits
from robo43.products.dq import DQ_DTYPE

logger = logging.getLogger(__name__)

__all__ = ["MasterFrame"]


@dataclass(slots=True)
class MasterFrame:
    """A combined master calibration frame, ready to write to disk."""

    data: np.ndarray
    variance: np.ndarray | None = None
    weight: np.ndarray | None = None
    mask: np.ndarray | None = None
    header_cards: list[tuple[str, Any, str]] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    bunit: str = "adu"
    path: Path | None = None

    def to_hdulist(self) -> fits.HDUList:
        """Assemble the SCI (+ VAR/WEIGHT/DQ) HDUList for this master frame."""
        primary = fits.PrimaryHDU(data=np.asarray(self.data, dtype=np.float32))
        primary.header["BUNIT"] = self.bunit
        for name, value, comment in self.header_cards:
            primary.header[name] = (value, comment)
        for line in self.history:
            primary.header.add_history(line)

        hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [primary]

        if self.variance is not None:
            var_hdu = fits.ImageHDU(
                data=np.asarray(self.variance, dtype=np.float32), name="VAR"
            )
            hdus.append(var_hdu)

        if self.weight is not None:
            weight_hdu = fits.ImageHDU(
                data=np.asarray(self.weight, dtype=np.float32), name="WEIGHT"
            )
            hdus.append(weight_hdu)

        if self.mask is not None:
            dq_hdu = fits.ImageHDU(
                data=np.asarray(self.mask).astype(np.int32), name="DQ"
            )
            hdus.append(dq_hdu)

        return fits.HDUList(hdus)

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write this master frame to ``path``, atomically."""
        written = write_fits(path, self.to_hdulist(), overwrite=overwrite)
        self.path = written
        return written

    @classmethod
    def load(cls, path: str | Path) -> MasterFrame:
        """Read a master frame previously written by :meth:`write`."""
        path = Path(path)
        try:
            with fits.open(path) as hdul:
                # Read primary HDU data
                primary = hdul[0]
                data = np.asarray(primary.data, dtype=np.float32)

                # Read BUNIT from header
                bunit = primary.header.get("BUNIT", "adu")

                # Read optional extensions
                variance = None
                weight = None
                mask = None

                if "VAR" in hdul:
                    variance = np.asarray(hdul["VAR"].data, dtype=np.float32)

                if "WEIGHT" in hdul:
                    weight = np.asarray(hdul["WEIGHT"].data, dtype=np.float32)

                if "DQ" in hdul:
                    mask = np.asarray(hdul["DQ"].data, dtype=DQ_DTYPE)

                # Construct and return the instance
                return cls(
                    data=data,
                    variance=variance,
                    weight=weight,
                    mask=mask,
                    bunit=bunit,
                    path=path,
                )
        except FileNotFoundError as e:
            msg = f"Master frame not found: {path}"
            raise CalibrationError(msg) from e
