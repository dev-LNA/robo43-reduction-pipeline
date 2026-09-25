"""Built-in header-keyword aliases and frame-type classification tables.

These are the fallback tables used when an :class:`~robo43.instruments.profile.InstrumentProfile`
does not declare its own aliases or classification maps.  An instrument
profile can extend or override any of this per-field via its own
``keywords``/``imagetyp_map`` tables; this module supplies the defaults every
profile inherits.
"""

from __future__ import annotations

#: Canonical field name -> ordered tuple of header keyword aliases, most
#: preferred first.  Matching against a FITS header is case-insensitive.
DEFAULT_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "naxis1": ("NAXIS1",),
    "naxis2": ("NAXIS2",),
    "imagetyp": ("IMAGETYP", "IMGTYPE", "OBSTYPE", "FRAMETYP"),
    "exptime": ("EXPTIME", "EXPOSURE", "EXPTIMES"),
    "date_obs": ("DATE-OBS", "DATEOBS"),
    "filter": ("FILTER", "FILTER1", "FILTNAM"),
    "ra": ("RA", "OBJCTRA", "TELRA", "CRVAL1"),
    "dec": ("DEC", "OBJCTDEC", "TELDEC", "CRVAL2"),
    "pixel_scale": ("SECPIX", "PIXSCALE", "SCALE"),
    "ccd_temp": ("CCD-TEMP", "CCDTEMP", "SET-TEMP", "TEMPERAT"),
    "gain_setting": ("GAIN", "EGAIN"),
    "read_noise": ("RDNOISE", "READNOIS"),
    "readout_mode": ("READOUTM", "READOUT", "READMODE"),
    "saturation": ("SATURATE", "SATLEVEL"),
    "airmass": ("AIRMASS",),
    "object": ("OBJECT",),
    "telescope": ("TELESCOP",),
    "instrument": ("INSTRUME",),
    "observer": ("OBSERVER",),
    "binning": ("XBINNING", "CCDXBIN", "BINX"),
    "focal_length": ("FOCALLEN",),
    "pixel_size": ("XPIXSZ", "PIXSIZE1"),
    "mjd": ("MJD", "MJD-OBS"),
}

#: Raw IMAGETYP value (lower-case) -> canonical frame type.  Callers must
#: lower-case the raw header value before looking it up here.
DEFAULT_IMAGETYP_MAP: dict[str, str] = {
    "light frame": "science",
    "bias frame": "bias",
    "dark frame": "dark",
    "flat field": "flat",
    "zero": "bias",
    "bias": "bias",
    "dark": "dark",
    "flat": "flat",
    "object": "science",
    "science": "science",
    "light": "science",
    "dome flat": "flat",
    "sky flat": "flat",
    "twilight flat": "flat",
    "focus frame": "focus",
    "focus": "focus",
}

#: Fallback substring classifier used when IMAGETYP is absent or unmapped:
#: canonical frame type -> substrings to look for in a lower-cased OBJECT
#: value.  ``science`` intentionally has no patterns -- it is the default
#: returned when nothing else matches.
OBJECT_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "bias": ("bias", "zero"),
    "dark": ("dark",),
    "flat": ("flat", "sky flat", "twilight", "dome flat"),
    "focus": ("focus",),
    "science": (),
}

#: Internal frame_type (bias/dark/flat/focus/science) -> the pipeline's
#: canonical, user-facing IMAGETYP vocabulary. Never "BIAS" -- bias frames
#: are always "ZERO".
CANONICAL_IMAGETYP: dict[str, str] = {
    "bias": "ZERO", "dark": "DARK", "flat": "FLAT", "focus": "FOCUS", "science": "OBJECT",
}
