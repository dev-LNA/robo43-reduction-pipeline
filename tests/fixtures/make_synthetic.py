"""Synthetic FITS night generator for robo43 pipeline tests.

Generates deterministic, fast-to-create synthetic FITS "nights" (bias, dark,
flat, and science frames) for two fictitious instruments, ``robo43`` and
``synth2``, along with a :class:`SyntheticTruth` record of the exact values
used to synthesize the data so downstream tests can assert recovery.

No FITS test data lives in the repository; everything is generated on the
fly under a caller-supplied output directory (normally a pytest ``tmp_path``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclasses.dataclass(frozen=True)
class SyntheticTruth:
    """Ground truth for a synthetic night, so tests can assert recovery."""

    instrument: str
    seed: int
    naxis1: int
    naxis2: int
    bias_level: float
    dark_current_e_per_s: float
    gain_e_per_adu: float
    read_noise_e: float
    star_positions_px: tuple[tuple[float, float], ...]
    star_fluxes_adu: tuple[float, ...]
    flat_gradient_coeffs: dict[str, tuple[float, float, float]]
    hot_pixel_coords: tuple[tuple[int, int], ...]
    hot_column: int | None
    cosmic_ray_coords: tuple[tuple[int, int], ...]

    def to_json(self) -> dict:
        return {
            "instrument": self.instrument,
            "seed": self.seed,
            "naxis1": self.naxis1,
            "naxis2": self.naxis2,
            "bias_level": self.bias_level,
            "dark_current_e_per_s": self.dark_current_e_per_s,
            "gain_e_per_adu": self.gain_e_per_adu,
            "read_noise_e": self.read_noise_e,
            "star_positions_px": [list(p) for p in self.star_positions_px],
            "star_fluxes_adu": list(self.star_fluxes_adu),
            "flat_gradient_coeffs": {
                k: list(v) for k, v in self.flat_gradient_coeffs.items()
            },
            "hot_pixel_coords": [list(c) for c in self.hot_pixel_coords],
            "hot_column": self.hot_column,
            "cosmic_ray_coords": [list(c) for c in self.cosmic_ray_coords],
        }

    @classmethod
    def from_json(cls, data: dict) -> SyntheticTruth:
        return cls(
            instrument=data["instrument"],
            seed=data["seed"],
            naxis1=data["naxis1"],
            naxis2=data["naxis2"],
            bias_level=data["bias_level"],
            dark_current_e_per_s=data["dark_current_e_per_s"],
            gain_e_per_adu=data["gain_e_per_adu"],
            read_noise_e=data["read_noise_e"],
            star_positions_px=tuple(tuple(p) for p in data["star_positions_px"]),
            star_fluxes_adu=tuple(data["star_fluxes_adu"]),
            flat_gradient_coeffs={
                k: tuple(v) for k, v in data["flat_gradient_coeffs"].items()
            },
            hot_pixel_coords=tuple(tuple(c) for c in data["hot_pixel_coords"]),
            hot_column=data["hot_column"],
            cosmic_ray_coords=tuple(tuple(c) for c in data["cosmic_ray_coords"]),
        )


def _render_gaussian_star(
    array: np.ndarray, x0: float, y0: float, flux: float, sigma: float
) -> None:
    """Add a 2D Gaussian of total integrated flux ``flux`` centered at (x0, y0).

    Rendered on a local cutout window (+/- 8 sigma) for speed; the Gaussian
    PDF is normalized to integrate to 1.0 over the infinite plane, so the
    total added flux is (very closely) exactly ``flux``.
    """
    ny, nx = array.shape
    half = int(np.ceil(8 * sigma))
    x_lo, x_hi = max(0, int(x0) - half), min(nx, int(x0) + half + 1)
    y_lo, y_hi = max(0, int(y0) - half), min(ny, int(y0) + half + 1)
    xs = np.arange(x_lo, x_hi)
    ys = np.arange(y_lo, y_hi)
    xx, yy = np.meshgrid(xs, ys)
    pdf = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2)) / (
        2 * np.pi * sigma**2
    )
    array[y_lo:y_hi, x_lo:x_hi] += flux * pdf


def _inject_hot_pixels_and_cr(
    array: np.ndarray,
    hot_pixel_coords: tuple[tuple[int, int], ...],
    hot_column: int | None,
    cosmic_ray_coords: tuple[tuple[int, int], ...],
    saturation_adu: float,
) -> None:
    for row, col in hot_pixel_coords:
        array[row, col] = saturation_adu * 0.98
    if hot_column is not None:
        array[:, hot_column] = saturation_adu * 0.98
    for row, col in cosmic_ray_coords:
        array[row, col] = saturation_adu * 0.9


def make_synthetic_night(
    outdir: Path,
    *,
    instrument: str = "robo43",
    seed: int = 0,
    strip: tuple[str, ...] = (),
) -> tuple[dict[str, list[Path]], SyntheticTruth]:
    """Generate a synthetic FITS "night" for the given instrument.

    Returns (paths, truth). ``paths`` maps frame category to a list of
    written FITS paths under ``outdir``. ``truth`` is also written as JSON
    to ``outdir / "truth.json"``.
    """
    valid_strip = {"gain", "pixel_scale", "filter"}
    for s in strip:
        if s not in valid_strip:
            raise ValueError(f"unknown strip value: {s!r} (valid: {sorted(valid_strip)})")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if strip:
        paths, truth = _make_robo43_night(outdir, seed=seed, strip=strip)
    elif instrument == "robo43":
        paths, truth = _make_robo43_night(outdir, seed=seed, strip=())
    elif instrument == "synth2":
        paths, truth = _make_synth2_night(outdir, seed=seed)
    else:
        raise ValueError(f"unknown instrument: {instrument!r}")

    with (outdir / "truth.json").open("w") as f:
        json.dump(truth.to_json(), f, indent=2)

    return paths, truth


def _make_robo43_night(
    outdir: Path, *, seed: int, strip: tuple[str, ...]
) -> tuple[dict[str, list[Path]], SyntheticTruth]:
    rng = np.random.default_rng(seed)

    naxis1, naxis2 = 256, 256
    bias_level = 1000.0
    gain_e_per_adu = 1.5
    read_noise_e = 8.0
    dark_current_e_per_s = 0.01
    saturation_adu = 55000.0
    read_noise_adu = read_noise_e / gain_e_per_adu

    small = bool(strip)  # keep the stripped-keyword variant small & fast

    n_bias = 3 if small else 5
    n_dark = 2 if small else 5
    n_flat = 3 if small else 5
    n_sci = 2 if small else 3

    filters_raw = ["Slot 0/Slot 3", "Slot 0/Slot 2", "Slot 0/Slot 4"]
    if small:
        filters_raw = filters_raw[:1]

    dark_exptimes = [30.0] if small else [30.0, 300.0]

    paths: dict[str, list[Path]] = {"bias": [], "dark": [], "flat": [], "science": []}

    date = _DateCursor("2026-01-15T01:00:00.000")

    def base_header(imagetyp: str, exptime: float, obj: str) -> fits.Header:
        hdr = fits.Header()
        hdr["NAXIS1"] = naxis1
        hdr["NAXIS2"] = naxis2
        hdr["IMAGETYP"] = imagetyp
        hdr["EXPTIME"] = float(exptime)
        hdr["DATE-OBS"] = date.next()
        hdr["OBJECT"] = obj
        hdr["TELESCOP"] = "ROBO43"
        hdr["INSTRUME"] = "ccd1"
        hdr["CCD-TEMP"] = -10.0
        hdr["XBINNING"] = 1
        if "gain" not in strip:
            hdr["GAIN"] = gain_e_per_adu
        if "pixel_scale" not in strip:
            hdr["XPIXSZ"] = 13.5
            hdr["FOCALLEN"] = 5257.0
        hdr["OBSERVER"] = "robo43-synth"
        return hdr

    # --- bias ---
    for i in range(n_bias):
        hdr = base_header("Bias Frame", 0.0, "bias")
        data = bias_level + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
        data = data.astype(np.float32)
        p = outdir / f"bias_{i + 1:04d}.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(p, overwrite=True)
        paths["bias"].append(p)

    # --- dark ---
    for exptime in dark_exptimes:
        tag = f"{int(exptime):03d}s"
        for i in range(n_dark):
            hdr = base_header("Dark Frame", exptime, "dark")
            dark_signal_adu = dark_current_e_per_s * exptime / gain_e_per_adu
            data = (
                bias_level
                + rng.poisson(max(dark_signal_adu, 0.0), (naxis2, naxis1)).astype(float)
                + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
            )
            data = data.astype(np.float32)
            p = outdir / f"dark_{tag}_{i + 1:04d}.fits"
            fits.PrimaryHDU(data=data, header=hdr).writeto(p, overwrite=True)
            paths["dark"].append(p)

    # --- flat ---
    flat_gradient_coeffs: dict[str, tuple[float, float, float]] = {}
    xn = (np.arange(naxis1) - naxis1 / 2) / (naxis1 / 2)
    yn = (np.arange(naxis2) - naxis2 / 2) / (naxis2 / 2)
    xx, yy = np.meshgrid(xn, yn)
    flat_level = 20000.0
    for filt in filters_raw:
        a, b, c = 1.0, rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
        flat_gradient_coeffs[filt] = (a, b, c)
        gradient = a + b * xx + c * yy
        for i in range(n_flat):
            hdr = base_header("Flat Field", 5.0, "flat")
            if "filter" not in strip:
                hdr["FILTER"] = filt
            signal = flat_level * gradient
            noise = rng.normal(0.0, np.sqrt(np.maximum(signal, 1.0)), (naxis2, naxis1))
            data = bias_level + signal + noise
            data = np.clip(data, 0.0, saturation_adu).astype(np.float32)
            filt_tag = filt.replace("Slot ", "S").replace("/", "-").replace(" ", "")
            p = outdir / f"flat_{filt_tag}_{i + 1:04d}.fits"
            fits.PrimaryHDU(data=data, header=hdr).writeto(p, overwrite=True)
            paths["flat"].append(p)

    # --- science ---
    n_stars = 4 if small else 6
    star_positions_px = tuple(
        (
            float(rng.uniform(30, naxis1 - 30)),
            float(rng.uniform(30, naxis2 - 30)),
        )
        for _ in range(n_stars)
    )
    star_fluxes_adu = tuple(float(rng.uniform(20000, 200000)) for _ in range(n_stars))
    star_sigma = 2.0

    n_hot = 2 if small else 5
    hot_pixel_coords = tuple(
        (int(rng.integers(0, naxis2)), int(rng.integers(0, naxis1))) for _ in range(n_hot)
    )
    hot_column = None if small else int(rng.integers(10, naxis1 - 10))

    n_cr = 2 if small else 4
    cosmic_ray_coords = tuple(
        (int(rng.integers(0, naxis2)), int(rng.integers(0, naxis1))) for _ in range(n_cr)
    )

    sky_level = 300.0
    ra_deg = 83.8221
    dec_deg = -5.3911
    ra_sexagesimal = "05:35:17.30"
    dec_sexagesimal = "-05:23:28.0"
    sci_filter = filters_raw[0]

    for i in range(n_sci):
        exptime = 60.0
        hdr = base_header("Light Frame", exptime, "NGC1976")
        if "filter" not in strip:
            hdr["FILTER"] = sci_filter
        if i == 0:
            hdr["RA"] = ra_deg
            hdr["DEC"] = dec_deg
        else:
            hdr["RA"] = ra_sexagesimal
            hdr["DEC"] = dec_sexagesimal
        if "pixel_scale" not in strip:
            hdr["CTYPE1"] = "RA---TAN"
            hdr["CTYPE2"] = "DEC--TAN"
            hdr["CRVAL1"] = ra_deg
            hdr["CRVAL2"] = dec_deg
            hdr["CRPIX1"] = naxis1 / 2.0
            hdr["CRPIX2"] = naxis2 / 2.0
            secpix = 0.529
            cdelt = secpix / 3600.0
            hdr["CDELT1"] = -cdelt
            hdr["CDELT2"] = cdelt
            hdr["CD1_1"] = -cdelt
            hdr["CD1_2"] = 0.0
            hdr["CD2_1"] = 0.0
            hdr["CD2_2"] = cdelt

        data = (
            bias_level
            + sky_level
            + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
        ).astype(np.float64)

        for (x0, y0), flux in zip(star_positions_px, star_fluxes_adu, strict=True):
            _render_gaussian_star(data, x0, y0, flux, star_sigma)

        _inject_hot_pixels_and_cr(
            data, hot_pixel_coords, hot_column, cosmic_ray_coords, saturation_adu
        )

        data = data.astype(np.float32)
        p = outdir / f"science_{i + 1:04d}.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(p, overwrite=True)
        paths["science"].append(p)

    truth = SyntheticTruth(
        instrument="robo43",
        seed=seed,
        naxis1=naxis1,
        naxis2=naxis2,
        bias_level=bias_level,
        dark_current_e_per_s=dark_current_e_per_s,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_e=read_noise_e,
        star_positions_px=star_positions_px,
        star_fluxes_adu=star_fluxes_adu,
        flat_gradient_coeffs=flat_gradient_coeffs,
        hot_pixel_coords=hot_pixel_coords,
        hot_column=hot_column,
        cosmic_ray_coords=cosmic_ray_coords,
    )
    return paths, truth


def _make_synth2_night(outdir: Path, *, seed: int) -> tuple[dict[str, list[Path]], SyntheticTruth]:
    rng = np.random.default_rng(seed)

    naxis1, naxis2 = 200, 180
    bias_level = 800.0
    gain_e_per_adu = 2.0
    read_noise_e = 6.0
    dark_current_e_per_s = 0.02
    saturation_adu = 65000.0
    read_noise_adu = read_noise_e / gain_e_per_adu

    n_bias, n_dark, n_flat, n_sci = 3, 2, 3, 2
    filters = ["V", "B"]
    dark_exptime = 60.0

    paths: dict[str, list[Path]] = {"bias": [], "dark": [], "flat": [], "science": []}
    date = _DateCursor("2026-02-01T01:00:00.000")

    def base_header(imgtype: str, exposure: float, obj: str) -> fits.Header:
        hdr = fits.Header()
        hdr["IMGTYPE"] = imgtype
        hdr["EXPOSURE"] = float(exposure)
        hdr["DATE-OBS"] = date.next()
        hdr["OBJECT"] = obj
        hdr["TELESCOP"] = "SYNTH2-TEL"
        hdr["INSTRUME"] = "synth2cam"
        hdr["SET-TEMP"] = -15.0
        return hdr

    def write_mef(path: Path, hdr: fits.Header, data: np.ndarray) -> None:
        primary = fits.PrimaryHDU(data=None, header=hdr)
        sci = fits.ImageHDU(data=data.astype(np.float32), name="SCI")
        fits.HDUList([primary, sci]).writeto(path, overwrite=True)

    # --- bias ---
    for i in range(n_bias):
        hdr = base_header("bias", 0.0, "bias")
        data = bias_level + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
        p = outdir / f"bias_{i + 1:04d}.fits"
        write_mef(p, hdr, data)
        paths["bias"].append(p)

    # --- dark ---
    for i in range(n_dark):
        hdr = base_header("dark", dark_exptime, "dark")
        dark_signal_adu = dark_current_e_per_s * dark_exptime / gain_e_per_adu
        data = (
            bias_level
            + rng.poisson(max(dark_signal_adu, 0.0), (naxis2, naxis1)).astype(float)
            + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
        )
        p = outdir / f"dark_060s_{i + 1:04d}.fits"
        write_mef(p, hdr, data)
        paths["dark"].append(p)

    # --- flat ---
    flat_gradient_coeffs: dict[str, tuple[float, float, float]] = {}
    xn = (np.arange(naxis1) - naxis1 / 2) / (naxis1 / 2)
    yn = (np.arange(naxis2) - naxis2 / 2) / (naxis2 / 2)
    xx, yy = np.meshgrid(xn, yn)
    flat_level = 15000.0
    for filt in filters:
        a, b, c = 1.0, rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
        flat_gradient_coeffs[filt] = (a, b, c)
        gradient = a + b * xx + c * yy
        for i in range(n_flat):
            hdr = base_header("flat", 3.0, "flat")
            hdr["FILTER"] = filt
            signal = flat_level * gradient
            noise = rng.normal(0.0, np.sqrt(np.maximum(signal, 1.0)), (naxis2, naxis1))
            data = np.clip(bias_level + signal + noise, 0.0, saturation_adu)
            p = outdir / f"flat_{filt}_{i + 1:04d}.fits"
            write_mef(p, hdr, data)
            paths["flat"].append(p)

    # --- science ---
    n_stars = 3
    star_positions_px = tuple(
        (
            float(rng.uniform(25, naxis1 - 25)),
            float(rng.uniform(25, naxis2 - 25)),
        )
        for _ in range(n_stars)
    )
    star_fluxes_adu = tuple(float(rng.uniform(15000, 150000)) for _ in range(n_stars))
    star_sigma = 1.8

    hot_pixel_coords = tuple(
        (int(rng.integers(0, naxis2)), int(rng.integers(0, naxis1))) for _ in range(2)
    )
    cosmic_ray_coords = tuple(
        (int(rng.integers(0, naxis2)), int(rng.integers(0, naxis1))) for _ in range(2)
    )

    sky_level = 250.0
    ra_sexagesimal = "10:44:32.10"
    dec_sexagesimal = "-59:41:02.4"
    ra_deg = 161.1338
    dec_deg = -59.6840
    sci_filter = filters[0]

    for i in range(n_sci):
        exptime = 45.0
        hdr = base_header("object", exptime, "eta_car_field")
        hdr["FILTER"] = sci_filter
        hdr["OBJCTRA"] = ra_sexagesimal
        hdr["OBJCTDEC"] = dec_sexagesimal
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = ra_deg
        hdr["CRVAL2"] = dec_deg
        hdr["CRPIX1"] = naxis1 / 2.0
        hdr["CRPIX2"] = naxis2 / 2.0
        secpix = 0.35
        cdelt = secpix / 3600.0
        hdr["CDELT1"] = -cdelt
        hdr["CDELT2"] = cdelt
        hdr["CD1_1"] = -cdelt
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = cdelt

        data = (
            bias_level
            + sky_level
            + rng.normal(0.0, read_noise_adu, (naxis2, naxis1))
        ).astype(np.float64)

        for (x0, y0), flux in zip(star_positions_px, star_fluxes_adu, strict=True):
            _render_gaussian_star(data, x0, y0, flux, star_sigma)

        _inject_hot_pixels_and_cr(
            data, hot_pixel_coords, None, cosmic_ray_coords, saturation_adu
        )

        p = outdir / f"science_{i + 1:04d}.fits"
        write_mef(p, hdr, data)
        paths["science"].append(p)

    truth = SyntheticTruth(
        instrument="synth2",
        seed=seed,
        naxis1=naxis1,
        naxis2=naxis2,
        bias_level=bias_level,
        dark_current_e_per_s=dark_current_e_per_s,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_e=read_noise_e,
        star_positions_px=star_positions_px,
        star_fluxes_adu=star_fluxes_adu,
        flat_gradient_coeffs=flat_gradient_coeffs,
        hot_pixel_coords=hot_pixel_coords,
        hot_column=None,
        cosmic_ray_coords=cosmic_ray_coords,
    )
    return paths, truth


class _DateCursor:
    """Produces monotonically increasing ISO8601 DATE-OBS strings."""

    def __init__(self, start_iso: str, step_seconds: float = 45.0) -> None:
        from astropy.time import Time

        self._time = Time(start_iso, format="isot", scale="utc")
        self._step = step_seconds

    def next(self) -> str:
        from astropy.time import TimeDelta

        value = self._time.isot
        self._time = self._time + TimeDelta(self._step, format="sec")
        return value
