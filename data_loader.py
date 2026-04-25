"""
Data loader for PowerGrid training/evaluation.

Priority:
  1. Real load data from PowerGridworld (NREL) — downloaded once and cached
  2. Realistic synthetic fallback if download fails

Solar data: generated using a clear-sky model with seasonal + cloud variation
            (no real solar CSV exists in either reference repo)

Usage:
    from data_loader import get_day_profiles
    solar, load, price = get_day_profiles(day_of_year=180, noise=0.05)
    # returns three lists of length 96 (one per 15-min step)
"""

import math
import os
import random
import urllib.request

# ── Constants ─────────────────────────────────────────────────────────────────
SOLAR_PEAK_KW   = 20.0
LOAD_PEAK_KW    = 23.0   # scale real normalized data to this peak
LOAD_BASE_KW    = 3.0
STEPS           = 96
DT              = 0.25   # hours per step

_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "data")
_CACHE_FILE = os.path.join(_CACHE_DIR, "real_load_profile.csv")
_LOAD_URL   = (
    "https://raw.githubusercontent.com/NatLabRockies/PowerGridworld/"
    "main/gridworld/distribution_system/data/ieee_13_dss/"
    "annual_hourly_load_profile.csv"
)

_PRICE_OFF_PEAK = 0.08
_PRICE_MID_PEAK = 0.15
_PRICE_ON_PEAK  = 0.30


# ── Real data download + cache ────────────────────────────────────────────────

def _ensure_data_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _download_load_data() -> list[float]:
    """Download normalized load profile from PowerGridworld (NREL). Returns 8760 values."""
    _ensure_data_dir()
    if not os.path.exists(_CACHE_FILE):
        print(f"Downloading real load data from PowerGridworld (NREL)...")
        try:
            urllib.request.urlretrieve(_LOAD_URL, _CACHE_FILE)
            print(f"  Saved to {_CACHE_FILE}")
        except Exception as e:
            print(f"  Download failed: {e}. Using synthetic fallback.")
            return []

    values = []
    with open(_CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                pass  # skip header if any
    return values


_REAL_LOAD_ANNUAL: list[float] = []   # loaded lazily, 8760 entries (hourly)


def _get_real_load_annual() -> list[float]:
    global _REAL_LOAD_ANNUAL
    if not _REAL_LOAD_ANNUAL:
        _REAL_LOAD_ANNUAL = _download_load_data()
    return _REAL_LOAD_ANNUAL


# ── Solar clear-sky model ─────────────────────────────────────────────────────

def _solar_clearsky(step: int, day_of_year: int) -> float:
    """
    Simplified Iqbal clear-sky GHI model.
    Returns PV output in kW for a 20 kW peak system.
    """
    h = step * DT                          # fractional hour 0–24
    # Solar declination (degrees)
    decl = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    lat  = 37.0                            # approximate US mid-latitude
    lat_r, decl_r = math.radians(lat), math.radians(decl)

    # Hour angle (solar noon = 0)
    ha = math.radians(15 * (h - 12))

    # Cosine of solar zenith angle
    cos_z = (math.sin(lat_r) * math.sin(decl_r)
             + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha))
    if cos_z <= 0:
        return 0.0

    # Clear-sky GHI (W/m^2 normalised to 1 at cos_z=1)
    ghi = cos_z ** 0.8
    # Scale to system peak
    return min(SOLAR_PEAK_KW, SOLAR_PEAK_KW * ghi)


def _solar_profile_96(day_of_year: int, noise: float = 0.05) -> list[float]:
    """96-step solar profile with Gaussian cloud noise and random cloud events."""
    rng_state = random.getstate()  # preserve caller's RNG state
    cloud_days = random.random() < 0.3   # 30% chance of partially cloudy day
    cloud_scale = random.uniform(0.5, 0.8) if cloud_days else 1.0

    profile = []
    for step in range(STEPS):
        base = _solar_clearsky(step, day_of_year) * cloud_scale
        if noise > 0 and base > 0.1:
            base *= max(0.0, 1.0 + noise * random.gauss(0, 1))
        profile.append(max(0.0, min(SOLAR_PEAK_KW, base)))
    return profile


# ── Load profile (real data or synthetic fallback) ────────────────────────────

def _synthetic_load_96(noise: float = 0.05) -> list[float]:
    """Bimodal residential load: morning + evening peaks. Synthetic fallback."""
    profile = []
    for step in range(STEPS):
        h = step * DT
        morning = 8.0 * math.exp(-((h - 8.0) ** 2) / 2.0)
        evening = 12.0 * math.exp(-((h - 19.0) ** 2) / 2.0)
        base = LOAD_BASE_KW + morning + evening
        if noise > 0:
            base *= max(0.3, 1.0 + noise * random.gauss(0, 0.5))
        profile.append(max(0.5, base))
    return profile


def _real_load_96(day_of_year: int, noise: float = 0.05) -> list[float]:
    """
    Extract 24 hours of real load from annual profile, upsample hourly→15-min,
    then scale from normalized [0,1] to kW.
    """
    annual = _get_real_load_annual()
    if len(annual) < 24:
        return _synthetic_load_96(noise)

    # Clamp day_of_year to available data
    day = max(0, min(day_of_year - 1, (len(annual) // 24) - 1))
    hour_start = day * 24
    hourly = annual[hour_start: hour_start + 24]

    if len(hourly) < 24:
        hourly = hourly + [hourly[-1]] * (24 - len(hourly))

    # Upsample: linear interpolation from hourly (24) to 15-min (96)
    profile = []
    for step in range(STEPS):
        h_idx = step / 4.0              # fractional hour index
        lo    = int(h_idx) % 24
        hi    = (lo + 1) % 24
        frac  = h_idx - int(h_idx)
        val   = hourly[lo] * (1 - frac) + hourly[hi] * frac
        # Scale: normalized [0.4, 0.9] → kW
        kwh = LOAD_BASE_KW + val * (LOAD_PEAK_KW - LOAD_BASE_KW)
        if noise > 0:
            kwh *= max(0.3, 1.0 + noise * random.gauss(0, 0.3))
        profile.append(max(0.5, kwh))

    return profile


# ── Price profile ─────────────────────────────────────────────────────────────

def _price_profile_96() -> list[float]:
    profile = []
    for step in range(STEPS):
        h = step * DT
        if (7.0 <= h < 11.0) or (17.0 <= h < 21.0):
            profile.append(_PRICE_ON_PEAK)
        elif 11.0 <= h < 17.0:
            profile.append(_PRICE_MID_PEAK)
        else:
            profile.append(_PRICE_OFF_PEAK)
    return profile


# ── Public API ────────────────────────────────────────────────────────────────

def get_day_profiles(
    day_of_year: int = 180,
    noise: float = 0.05,
    use_real_data: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """
    Returns (solar_kw[96], load_kw[96], price[96]) for a given day of year.

    Args:
        day_of_year: 1–365. Affects solar seasonality and which real load day is picked.
        noise:       Gaussian noise fraction added to both solar and load.
        use_real_data: If True, attempt to download/use real load from PowerGridworld.
    """
    solar = _solar_profile_96(day_of_year, noise)
    load  = (_real_load_96(day_of_year, noise) if use_real_data
             else _synthetic_load_96(noise))
    price = _price_profile_96()
    return solar, load, price


def get_dataset(
    n_days: int = 100,
    noise: float = 0.05,
    use_real_data: bool = True,
    seed: int | None = None,
) -> list[tuple[list[float], list[float], list[float]]]:
    """
    Generate a dataset of n_days episodes (solar, load, price profiles).
    Samples random days of year; uses real load data where available.

    Returns list of (solar[96], load[96], price[96]) tuples.
    """
    if seed is not None:
        random.seed(seed)
    dataset = []
    for _ in range(n_days):
        doy = random.randint(1, 365)
        dataset.append(get_day_profiles(doy, noise, use_real_data))
    return dataset


def dataset_stats(dataset: list) -> dict:
    """Return min/max/mean stats across the dataset for quick inspection."""
    all_solar = [v for s, _, _ in dataset for v in s]
    all_load  = [v for _, l, _ in dataset for v in l]
    return {
        "episodes": len(dataset),
        "solar_min": min(all_solar), "solar_max": max(all_solar),
        "solar_mean": sum(all_solar) / len(all_solar),
        "load_min": min(all_load),   "load_max": max(all_load),
        "load_mean": sum(all_load) / len(all_load),
    }


if __name__ == "__main__":
    print("Loading data...")
    solar, load, price = get_day_profiles(day_of_year=180)
    print(f"Solar   : min={min(solar):.2f}  max={max(solar):.2f}  kW")
    print(f"Load    : min={min(load):.2f}   max={max(load):.2f}  kW")
    print(f"Price   : {set(price)} $/kWh")

    ds = get_dataset(n_days=50, seed=42)
    stats = dataset_stats(ds)
    print(f"\nDataset stats (50 days):")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
