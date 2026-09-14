"""Cosmic-ray rejection via L.A.Cosmic (``astroscrappy``).

This is the only cosmic-ray rejection in the pipeline.  ``astroscrappy`` is
imported lazily inside :func:`detect_cosmics` so that this module -- and
anything that imports it -- loads cleanly even when astroscrappy is not
installed (``robo43 --help`` must not break because of an optional
dependency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from robo43.config import CalibSettings
from robo43.exceptions import ExternalToolError
from robo43.products.dq import DQ, FATAL_FLAGS

__all__ = ["CosmicRayResult", "apply_to_dq", "detect_cosmics"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CosmicRayResult:
    """Result of one :func:`detect_cosmics` call.

    Extended beyond the original 4-field sketch (``cleaned``, ``mask``,
    ``n_flagged``, ``fraction``) with ``enabled`` and ``sigclip`` so that
    :meth:`header_cards` has what it needs.
    """

    cleaned: np.ndarray
    mask: np.ndarray
    n_flagged: int
    fraction: float
    enabled: bool
    sigclip: float

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """FITS header card triples describing this cosmic-ray pass."""
        try:
            import astroscrappy

            software_version = astroscrappy.__version__
        except ImportError:
            software_version = "unknown"

        return [
            ("CRREJ", bool(self.enabled), "cosmic-ray rejection was performed"),
            ("CRNPIX", int(self.n_flagged), "number of pixels flagged as cosmic rays"),
            ("CRFRAC", float(self.fraction), "fraction of pixels flagged as cosmic rays"),
            ("CRSIGCLP", float(self.sigclip), "sigma-clip threshold used for cosmic-ray detection"),
            ("CRSOFT", "astroscrappy", "software used for cosmic-ray rejection"),
            ("CRSOFTV", software_version, "astroscrappy version used"),
        ]


def detect_cosmics(
    data: np.ndarray,
    *,
    settings: CalibSettings,
    gain_e_per_adu: float | None = None,
    read_noise_e: float | None = None,
    saturation: float | None = None,
    variance: np.ndarray | None = None,  # noqa: ARG001 - unused; see docstring
    dq: np.ndarray | None = None,
    in_electrons: bool = False,
) -> CosmicRayResult:
    """Detect and clean cosmic rays with L.A.Cosmic (``astroscrappy``).

    Units convention -- read carefully, this is the easy thing to get wrong:

    - ``read_noise_e`` is ALWAYS in electrons, regardless of what units
      ``data`` is in.
    - When ``in_electrons=False`` (``data`` is in ADU), ``gain_e_per_adu`` is
      required and is passed to astroscrappy as ``gain`` so it can build its
      Poisson+read-noise model.
    - When ``in_electrons=True`` (``data`` has already been multiplied by the
      gain), ``gain=1.0`` is passed to astroscrappy instead (the data must
      not be re-multiplied); ``read_noise_e`` is still in electrons.
    - ``saturation``, when given, is in the same units as ``data`` (ADU if
      ``in_electrons=False``, electrons if ``in_electrons=True``). astroscrappy
      always compares its ``satlevel`` parameter against its own internal
      ``data * gain`` array (its docstring calls ``satlevel`` "Saturation of
      level of the image (electrons)"), so ``saturation`` is multiplied by
      the same ``gain`` passed to astroscrappy before being forwarded as
      ``satlevel`` -- it is not passed through unchanged. When
      ``in_electrons=True`` that ``gain`` is 1.0, so ``saturation`` is
      unaffected. When ``None``, ``satlevel`` is left at astroscrappy's own
      default.

    If cosmic-ray rejection is disabled (``settings.cosmic_ray_rejection`` is
    False), astroscrappy is never called; ``cleaned`` is returned as a
    **copy** of ``data`` cast to float32 -- not a view -- so mutating the
    result never mutates the caller's original array, whether rejection ran
    or not.

    ``variance`` is accepted for API-contract compatibility with a future
    caller that may want to pass a real variance map, but astroscrappy
    builds its own internal noise model and does not take one, so it is
    currently unused.

    ``dq``, when given, marks pixels already known bad (any
    :data:`~robo43.products.dq.FATAL_FLAGS` bit set) so they are excluded
    from consideration as detection seeds via astroscrappy's ``inmask``.
    """
    if not settings.cosmic_ray_rejection:
        logger.debug("cosmic-ray rejection disabled; skipping astroscrappy")
        return CosmicRayResult(
            cleaned=data.astype(np.float32, copy=True),
            mask=np.zeros(data.shape, dtype=bool),
            n_flagged=0,
            fraction=0.0,
            enabled=False,
            sigclip=settings.cosmic_ray_sigclip,
        )

    try:
        import astroscrappy
    except ImportError as exc:
        msg = (
            "cosmic-ray rejection requires astroscrappy; "
            "install it with `pip install astroscrappy`"
        )
        raise ExternalToolError(msg) from exc

    if read_noise_e is None or (not in_electrons and gain_e_per_adu is None):
        msg = (
            "read_noise_e is always required, and gain_e_per_adu is required "
            "when in_electrons=False; there is no safe default for either"
        )
        raise ValueError(msg)

    gain = 1.0 if in_electrons else gain_e_per_adu

    kwargs: dict[str, Any] = {
        "gain": gain,
        "readnoise": read_noise_e,
        "sigclip": settings.cosmic_ray_sigclip,
    }
    if saturation is not None:
        # astroscrappy's satlevel is compared against its own internal
        # data * gain array (always electrons), so it must be scaled by the
        # same gain -- not passed through in the caller's input units.
        kwargs["satlevel"] = saturation * gain
    if dq is not None:
        kwargs["inmask"] = (dq & int(FATAL_FLAGS)) != 0

    mask, cleaned = astroscrappy.detect_cosmics(data, **kwargs)

    mask = mask.astype(bool)
    n_flagged = int(mask.sum())
    return CosmicRayResult(
        cleaned=cleaned.astype(np.float32),
        mask=mask,
        n_flagged=n_flagged,
        fraction=float(n_flagged) / mask.size,
        enabled=True,
        sigclip=float(settings.cosmic_ray_sigclip),
    )


def apply_to_dq(dq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OR :data:`~robo43.products.dq.DQ.COSMIC_RAY` into ``dq`` where ``mask``.

    Returns a new ``int32`` array; ``dq`` is not mutated, and any
    pre-existing bits (including at cosmic-ray pixels) survive.
    """
    result = dq.astype(np.int32, copy=True)
    result[mask] |= int(DQ.COSMIC_RAY)
    return result
