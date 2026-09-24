"""Apply a solved :class:`~robo43.astrometry.solve.AstrometrySolution` to a MEF.

Writes the solver's WCS/SIP cards and a block of ``ASTR*`` provenance cards
into the ``SCI`` (primary) header of a ``_proc.fits`` :class:`~astropy.io.fits.HDUList`,
and mirrors the identical cards into ``ERR``/``DQ``/``WEIGHT`` so every plane
of the MEF is independently navigable -- a consumer that opens only the
``WEIGHT`` extension (to build a coadd, say) still gets a working WCS without
having to know to look at the primary header instead.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from robo43.exceptions import AstrometryError

if TYPE_CHECKING:
    from astropy.io import fits

    from robo43.astrometry.solve import AstrometrySolution

__all__ = ["MIRRORED_HDU_NAMES", "apply_solution", "astrometry_cards"]

#: HDU names an astrometric solution is written into, in the MEF layout
#: documented in robo43.products.mef.
MIRRORED_HDU_NAMES: tuple[str, ...] = ("SCI", "ERR", "DQ", "WEIGHT")


def _astrometry_package_version() -> str:
    try:
        return version("astrometry")
    except PackageNotFoundError:
        return "unknown"


def astrometry_cards(solution: AstrometrySolution) -> list[tuple[str, Any, str]]:
    """Provenance cards summarising one astrometric solve attempt.

    Only ``ASTRSOLV`` (and, on failure, ``ASTRFAIL``) are returned for an
    unsolved ``solution``; the rest describe a solution that does not exist
    in that case.
    """
    cards: list[tuple[str, Any, str]] = [
        ("ASTRSOLV", solution.solved, "Astrometric solution found"),
    ]
    if not solution.solved:
        cards.append(("ASTRFAIL", solution.failure_reason or "unknown", "Reason astrometry failed"))
        return cards

    cards.extend(
        [
            ("ASTRRMS", solution.rms_arcsec, "Astrometric residual RMS (arcsec)"),
            ("ASTRNSRC", solution.n_detected, "Sources detected for astrometry"),
            ("ASTRNMAT", solution.n_matched, "Sources matched to the index catalogue"),
            ("ASTRLODD", solution.logodds, "Solver log-odds of the match"),
            ("ASTRSCL", solution.scale_arcsec_per_pixel, "Solved plate scale (arcsec/pixel)"),
            ("ASTRSIP", solution.sip_order, "SIP polynomial order"),
            ("ASTRSERI", solution.index_series or "NONE", "Astrometry.net index series used"),
            (
                "ASTRSCLS",
                ",".join(str(s) for s in solution.index_scales),
                "Index scales used",
            ),
            ("ASTRRUNG", solution.rung, "Fallback-ladder rung that solved"),
            ("ASTRBLND", solution.blind, "Solved without a position hint (blind)"),
            ("ASTRSOFT", "astrometry", "Astrometric solver package"),
            ("ASTRSFTV", _astrometry_package_version(), "Astrometric solver package version"),
        ]
    )
    return cards


def apply_solution(hdul: fits.HDUList, solution: AstrometrySolution) -> None:
    """Write ``solution``'s WCS/SIP and provenance cards into every HDU in
    :data:`MIRRORED_HDU_NAMES` present in ``hdul``, in place.

    Raises :class:`~robo43.exceptions.AstrometryError` if ``solution.solved``
    is ``False`` -- there is no WCS to apply -- or if none of
    :data:`MIRRORED_HDU_NAMES` is present in ``hdul``. Callers must check
    ``solution.solved`` themselves before calling this (see
    ``robo43.cli.astrometry``, which records a failed attempt in its status
    CSV without calling this function at all).
    """
    if not solution.solved:
        msg = f"cannot apply an unsolved astrometric solution: {solution.failure_reason}"
        raise AstrometryError(msg)

    present = [name for name in MIRRORED_HDU_NAMES if name in hdul]
    if not present:
        msg = "no SCI/ERR/DQ/WEIGHT HDU found to apply the astrometric solution to"
        raise AstrometryError(msg)

    cards = astrometry_cards(solution)
    for name in present:
        header = hdul[name].header
        for key, value, comment in solution.wcs_cards:
            header[key] = (value, comment)
        for key, value, comment in cards:
            header[key] = (value, comment)
        if "WCSSRC" in header:
            header["WCSSRC"] = ("solved", "WCS from astrometric plate solve")

    hdul[present[0]].header.add_history(
        f"astrometry: solved rung={solution.rung} blind={solution.blind} "
        f'rms={solution.rms_arcsec:.3f}" nmatch={solution.n_matched}/{solution.n_detected}'
    )
