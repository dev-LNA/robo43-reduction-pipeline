"""Per-frame change log: what the pipeline did to a frame's data and header.

Every reduction step in :mod:`robo43.reduce.frame` already records, as FITS
header cards on the reduced product, exactly which corrections it applied or
skipped and with what parameters (``OSCANLVL``, ``BIASFIL``, ``DARKSCL``,
``GAINVAL``, ``CRNPIX``, and so on), plus a running per-step median
(``MEDRAW``, ``MEDOSC``, ``MEDBIAS``, ``MEDDARK``, ``MEDFLAT``, ``MEDCOSM``,
``MEDGAIN``) that makes each step's effect on the data auditable. This module
does not recompute or duplicate any of that: it reads the raw and product
headers plus the frame's :class:`~robo43.reduce.frame.ReductionStatus`
(``steps_applied``/``steps_skipped``) and turns them into one durable record
per frame, appended to ``<outdir>/change_log.jsonl``.

Format: JSON Lines, one JSON object per reduced frame, appended rather than
rewritten. A single growing JSON array would require reading and rewriting
the whole file for every frame and would be corrupted by an interrupted run;
JSON Lines needs only an append, survives a mid-write crash with at most one
truncated trailing line, and is trivially streamed, grepped, or loaded one
record at a time.

Writing the log is never allowed to fail a reduction: every entry point here
catches its own exceptions, logs a warning, and returns.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.io.fits.card import CARD_LENGTH, KEYWORD_LENGTH

if TYPE_CHECKING:
    from astropy.io import fits

    from robo43.io.meta import FrameMeta
    from robo43.reduce.frame import ReductionStatus

__all__ = ["DEFAULT_CHANGE_LOG_NAME", "append_change_log"]

logger = logging.getLogger(__name__)

#: Default basename of the change log written into the reduce output directory.
DEFAULT_CHANGE_LOG_NAME = "change_log.jsonl"

#: Header keywords never diffed: pure FITS structural boilerplate (unchanged
#: SIMPLE/EXTEND) or free-text cards astropy exposes specially (COMMENT/
#: HISTORY return their *last* value through ``header[key]``, so a positional
#: diff would be misleading); HISTORY lines are instead copied verbatim below.
_SKIP_HEADER_KEYS = frozenset({"SIMPLE", "EXTEND", "COMMENT", "HISTORY", ""})

#: Reduction step name -> (before-median keyword, after-median keyword) among
#: the ``MED*`` cards :func:`robo43.reduce.frame.reduce_frame` writes to every
#: product header. "mask" has no entry: it only ever changes the DQ plane, not
#: the SCI data, so its before/after data summary is identical (MEDFLAT) and
#: is filled in below rather than listed here.
_STEP_MEDIAN_CARDS: dict[str, tuple[str, str]] = {
    "overscan": ("MEDRAW", "MEDOSC"),
    "trim": ("MEDRAW", "MEDOSC"),
    "bias": ("MEDOSC", "MEDBIAS"),
    "dark": ("MEDBIAS", "MEDDARK"),
    "flat": ("MEDDARK", "MEDFLAT"),
    "cosmics": ("MEDFLAT", "MEDCOSM"),
    "gain": ("MEDCOSM", "MEDGAIN"),
}

#: Reduction step name -> header keywords carrying that step's parameters
#: (already written by :func:`robo43.reduce.frame.reduce_frame`); copied
#: verbatim into the log rather than re-derived.
_STEP_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "overscan": ("OSCANCOR", "OSCANLVL", "OSCANREG"),
    "trim": ("TRIMMED", "TRIMREG"),
    "bias": ("BIASFIL",),
    "dark": ("DARKFIL", "DARKMETH", "DARKSCL"),
    "flat": ("FLATFIL",),
    "mask": ("MASKFIL",),
    "cosmics": ("CRNPIX", "CRFRAC", "CRSIGCLP", "CRSOFT", "CRSOFTV"),
    # GAINSET/READOUTM/GAINRN/GAINNPR/GAINSCT are written whenever a
    # measurement exists (robo43.calib.gain.GainMeasurement.header_cards());
    # GAINPAT only when the per-tile gains came from a tile gain pattern table;
    # GAINVAL="NONE"/GAINSRC="none" alone (via no_measurement_cards()) is
    # what makes a skipped gain step -- for want of a measurement -- visible
    # in this log without a measurement's other keys being present.
    "gain": (
        "GAINVAL", "GAINSRC", "GAINSET", "READOUTM", "GAINRN", "GAINNPR", "GAINSCT", "GAINPAT",
        "BUNIT",
    ),
    "weight": (),
}

#: Canonical execution order of reduction steps, matching the "--- Step N:
#: ... ---" comments in :func:`robo43.reduce.frame.reduce_frame`. Used to sort
#: ``steps`` entries into the order the pipeline actually executes; a change
#: log is an audit trail and must read that way regardless of which steps
#: were applied vs. skipped. A step name that appears in
#: ``status.steps_applied``/``steps_skipped`` but not here is emitted after
#: every recognized step, in the order it was first seen, rather than dropped.
_STEP_ORDER: tuple[str, ...] = (
    "overscan",
    "trim",
    "bias",
    "dark",
    "flat",
    "mask",
    "cosmics",
    "gain",
    "variance",
    "weight",
)

#: Wrap width of a HISTORY/COMMENT card's value. Commentary cards have no
#: "= " value indicator, so the full width past the keyword field is
#: available to the value: ``CARD_LENGTH`` (80) minus ``KEYWORD_LENGTH`` (8)
#: leaves 72. Astropy splits a HISTORY string longer than this across
#: multiple physical cards with no marker linking them back together, so a
#: line that exactly fills this width is treated as continued by the next
#: line rather than as a coincidentally sized, complete entry.
_HISTORY_WRAP_WIDTH = CARD_LENGTH - KEYWORD_LENGTH


def _header_diff(old: fits.Header, new: fits.Header) -> list[dict[str, Any]]:
    """Every keyword added, modified, or removed between ``old`` and ``new``.

    Keys are compared case-insensitively, in first-seen order across both
    headers. ``COMMENT``/``HISTORY`` are excluded (see ``_SKIP_HEADER_KEYS``);
    ``HISTORY`` lines are reported separately, verbatim, by the caller.
    """
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_keys: list[str] = []
    for key in list(old.keys()) + list(new.keys()):
        upper = key.upper()
        if upper in _SKIP_HEADER_KEYS or upper in seen:
            continue
        seen.add(upper)
        ordered_keys.append(key)

    for key in ordered_keys:
        old_value = old.get(key, None)
        new_value = new.get(key, None)
        if old_value == new_value:
            continue
        comment = new.comments[key] if key in new else ""
        if not comment and key in old:
            comment = old.comments[key]
        changes.append(
            {"keyword": key, "old": old_value, "new": new_value, "reason": comment}
        )
    return changes


def _step_entry(
    name: str, *, applied: bool, product_header: fits.Header
) -> dict[str, Any]:
    """Build one ``steps`` entry from the header cards already written for ``name``."""
    before_key, after_key = _STEP_MEDIAN_CARDS.get(name, (None, None))
    if name == "mask":
        before_key = after_key = "MEDFLAT"
    data_before = (
        float(product_header[before_key])
        if before_key and before_key in product_header
        else None
    )
    data_after = (
        float(product_header[after_key])
        if after_key and after_key in product_header
        else None
    )
    params = {
        key: product_header[key]
        for key in _STEP_PARAM_KEYS.get(name, ())
        if key in product_header
    }
    return {
        "step": name,
        "applied": applied,
        "params": params,
        "data_before": data_before,
        "data_after": data_after,
    }


def _ordered_step_names(
    steps_applied: tuple[str, ...], steps_skipped: tuple[str, ...]
) -> list[str]:
    """Every step name from both lists, deduplicated and put in execution order.

    Steps in ``_STEP_ORDER`` sort by their position there; any other step
    name sorts after all of them, in the order it was first seen across
    ``steps_applied`` then ``steps_skipped``, rather than being dropped.
    """
    seen: list[str] = []
    for name in list(steps_applied) + list(steps_skipped):
        if name not in seen:
            seen.append(name)

    def _sort_key(name: str) -> tuple[int, int]:
        if name in _STEP_ORDER:
            return (0, _STEP_ORDER.index(name))
        return (1, seen.index(name))

    return sorted(seen, key=_sort_key)


def _rejoin_history(lines: list[str]) -> list[str]:
    """Rejoin HISTORY lines astropy split across multiple physical cards.

    A HISTORY value longer than ``_HISTORY_WRAP_WIDTH`` characters is stored
    as consecutive cards, each holding one fragment; a fragment exactly
    ``_HISTORY_WRAP_WIDTH`` characters long is therefore continued by the
    line after it, which is concatenated onto it with no separator. Applied
    repeatedly so a value spanning three or more cards rejoins in full.
    """
    rejoined: list[str] = []
    for line in lines:
        if rejoined and len(rejoined[-1]) == _HISTORY_WRAP_WIDTH:
            rejoined[-1] += line
        else:
            rejoined.append(line)
    return rejoined


def append_change_log(
    outdir: str | os.PathLike[str],
    meta: FrameMeta,
    raw_header: fits.Header,
    product_header: fits.Header,
    status: ReductionStatus,
    *,
    filename: str = DEFAULT_CHANGE_LOG_NAME,
) -> None:
    """Append one JSON Lines record describing everything this reduction changed.

    ``raw_header`` is the frame's header before reduction; ``product_header``
    is ``ReducedProduct.header`` after. The record combines: which steps ran
    (``status.steps_applied``/``steps_skipped``), each step's parameters and
    before/after data median (read back from the header cards the reduction
    already wrote), the full header diff between ``raw_header`` and
    ``product_header``, any ``HISTORY`` lines, and the frame's field-level
    provenance (:meth:`~robo43.io.meta.FrameMeta.provenance_json`).

    ``steps`` is emitted in the pipeline's canonical execution order (see
    ``_STEP_ORDER``), not in applied-then-skipped order, so the log reads as
    an audit trail of what actually ran. ``history`` rejoins any HISTORY
    value astropy wrapped across multiple physical cards (see
    ``_rejoin_history``) before it is recorded.

    Never raises: failure to write the log is logged at ``WARNING`` and
    swallowed, since a missing audit entry must never fail a reduction.
    """
    try:
        applied_set = set(status.steps_applied)
        steps = [
            _step_entry(
                name, applied=name in applied_set, product_header=product_header
            )
            for name in _ordered_step_names(status.steps_applied, status.steps_skipped)
        ]
        history = _rejoin_history(
            [str(value) for value in product_header.get("HISTORY", [])]
        )

        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "raw_path": str(meta.path),
            "product_path": str(status.product_path) if status.product_path else None,
            "frame_type": meta.frame_type,
            "proc_status": status.proc_status,
            "steps": steps,
            "header_changes": _header_diff(raw_header, product_header),
            "history": history,
            "field_provenance": meta.provenance_json(),
        }

        path = Path(outdir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
    except Exception as exc:
        logger.warning("failed to append change-log entry for %s: %s", meta.path, exc)
