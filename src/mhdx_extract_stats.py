# sim_noise.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, Any, Tuple, Optional, Sequence

from scipy.stats import norm

import pickle
import os

def save_variability_results(filename, rt_avg_std_of_means, rt_avg_std_of_stds, rt_std_global_mean, dt_df, dt_map_pf, auc_mean, auc_std):
    """
    Save variability results into a pickle file.
    
    Parameters
    ----------
    filename : str
        Path to output pickle file.
    rt_avg_std_of_means : float
    rt_avg_std_of_stds : float
    rt_std_global_mean : float
    dt_df : pd.DataFrame
    dt_map_pf : dict
    auc_mean : float
    auc_std : float
    """
    results = {
        "rt_avg_std_of_means": rt_avg_std_of_means,
        "rt_avg_std_of_stds": rt_avg_std_of_stds,
        "rt_std_global_mean": rt_std_global_mean,
        "dt_df": dt_df,
        "dt_map_pf": dt_map_pf,
        "auc_mean": auc_mean,
        "auc_std": auc_std
    }
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_variability_results(filename):
    """
    Load variability results from a pickle file.
    
    Returns
    -------
    tuple : (rt_avg_std_of_means, rt_avg_std_of_stds, rt_std_global_mean, dt_df, dt_map_pf)
    """
    with open(filename, "rb") as f:
        results = pickle.load(f)
    
    return (
        results["rt_avg_std_of_means"],
        results["rt_avg_std_of_stds"],
        results["rt_std_global_mean"],
        results["dt_df"],
        results["dt_map_pf"],
        results["auc_mean"],
        results["auc_std"]
    )


def save_noise_arrays(out, filename: str):
    """
    Save the result of generate_noise_arrays to a pickle file.
    
    Parameters
    ----------
    out : dict
        Dictionary returned by generate_noise_arrays.
    filename : str
        Path to save the pickle file.
    """
    # Ensure numpy arrays are converted into something pickle-safe
    safe_out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    
    with open(filename, "wb") as f:
        pickle.dump(safe_out, f)


def load_noise_arrays(filename: str):
    """
    Load the result of generate_noise_arrays from a pickle file.
    
    Returns
    -------
    dict : Restored dictionary with numpy arrays reconstructed.
    """
    with open(filename, "rb") as f:
        safe_out = pickle.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    restored_out = {k: (np.array(v) if isinstance(v, list) else v) for k, v in safe_out.items()}
    
    return restored_out


# =========================================================
# Utilities
# =========================================================
def _read_df(fp: str) -> Optional[pd.DataFrame]:
    """Read a dataframe from .pkl/.df.pkl/.json; return None on failure."""
    try:
        return pd.read_json(fp)
    except Exception as e:
        print(f"[WARN] could not read {fp}: {e}")
        return None


def moments_from_profile(axis: Sequence[float],
                         intensity: Sequence[float],
                         use_bin_width: bool = True) -> Tuple[float, float]:
    """
    Return (mean, std) for a discrete profile I(x) along axis x (handles uneven spacing).
    """
    I = np.asarray(intensity, float).clip(min=0)
    x = np.asarray(axis, float)
    if I.size != x.size or I.size == 0 or I.max() <= 0:
        return float("nan"), float("nan")
    if use_bin_width and I.size > 1:
        dx = np.r_[np.diff(x), x[-1] - x[-2]]
        w = I * dx
    else:
        w = I
    tot = w.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    mu = (w * x).sum() / tot
    var = (w * (x - mu) ** 2).sum() / tot
    return float(mu), float(np.sqrt(var))


# =========================================================
# RT: global variability summary (no charge binning)
# =========================================================
def rt_variability_summary(files: Iterable[str],
                           *,
                           winner_only: bool = True,
                           use_bin_width: bool = True
                           ) -> Tuple[float, float, float]:
    """
    Compute:
      (1) avg_std_of_means  = average (across files) of [std of RT means across timepoints]
      (2) avg_std_of_stds   = average (across files) of [std of RT stds across timepoints]
      (3) rt_std_global_mean= typical RT std across all ICs (pooled mean of per-IC RT stds)

    Returns (nan, nan, nan) if nothing qualified.
    """
    std_means_per_file: list[float] = []
    std_stds_per_file:  list[float] = []
    all_sds:            list[float] = []

    for fp in files:
        df = _read_df(fp)
        if df is None or df.empty:
            continue
        df_use = df[df["winner"] == 1] if (winner_only and "winner" in df.columns) else df

        mus, sds = [], []
        for _, row in df_use.iterrows():
            ic = row["ic"] if "ic" in row else getattr(row, "ic", None)
            if not ic or "retention_labels" not in ic or "rts" not in ic:
                continue
            mu, sd = moments_from_profile(ic["retention_labels"], ic["rts"], use_bin_width=use_bin_width)
            if np.isfinite(mu) and np.isfinite(sd):
                mus.append(float(mu))
                sds.append(float(sd))

        if sds:
            all_sds.extend(sds)
        if len(mus) >= 2 and len(sds) >= 2:
            std_means_per_file.append(float(np.std(mus, ddof=1)))
            std_stds_per_file.append(float(np.std(sds, ddof=1)))

    if not all_sds or not std_means_per_file or not std_stds_per_file:
        return float("nan"), float("nan"), float("nan")

    avg_std_of_means   = float(np.mean(std_means_per_file))
    avg_std_of_stds    = float(np.mean(std_stds_per_file))
    rt_std_global_mean = float(np.mean(all_sds))
    return avg_std_of_means, avg_std_of_stds, rt_std_global_mean


# =========================================================
# DT: per-charge summary (averaging across timepoints per file)
# =========================================================
def _extract_dt_mu_sigma(ic: Dict[str, Any],
                         use_bin_width: bool = True) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    ic dict keys expected:
      'dts' (intensity, arbitrary units),
      'drift_labels' (ms),
      'charge_states' (list/array-like), first entry is the charge z.
    Returns (z, mu_ms, sd_ms) or (None, None, None).
    """
    try:
        zlist = ic.get("charge_states", None)
        if zlist is None or len(zlist) == 0:
            return None, None, None
        z = int(np.asarray(zlist).flatten()[0])
        mu, sd = moments_from_profile(ic["drift_labels"], ic["dts"], use_bin_width=use_bin_width)
        if not (np.isfinite(mu) and np.isfinite(sd)):
            return None, None, None
        return z, float(mu), float(sd)
    except Exception:
        return None, None, None


def collect_dt_mean_and_std_summary_per_file(
    file_paths: Iterable[str],
    *,
    winner_only: bool = True,
    min_examples_per_file: int = 3,
    use_bin_width: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    For each file:
      • compute DT (mu, sd) for each qualifying IC/timepoint
      • per charge z, require >= min_examples_per_file in that file
      • compute per-file averages across timepoints: mean(mu), mean(sd), and within-file SD of sd
    Then pool those per-file values across files and summarize per charge.

    Returns:
      summary_df with columns:
        ['z','mean_means','std_means','dt_std_mean','dt_std_std',
         'dt_std_within_mean','dt_std_within_std','n_files']
      summary_map[z] = dict(...) mirroring the same stats.
    """
    pooled_mu_means: Dict[int, list] = {}
    pooled_sd_means: Dict[int, list] = {}
    pooled_sd_within: Dict[int, list] = {}
    files_seen_by_z: Dict[int, set] = {}

    for fp in file_paths:
        df = _read_df(fp)
        if df is None or df.empty:
            continue
        df_use = df[df["winner"] == 1] if (winner_only and "winner" in df.columns) else df

        per_file_mu: Dict[int, list] = {}
        per_file_sd: Dict[int, list] = {}
        for _, row in df_use.iterrows():
            ic = row["ic"] if "ic" in row else getattr(row, "ic", None)
            if ic is None:
                continue
            z, mu, sd = _extract_dt_mu_sigma(ic, use_bin_width=use_bin_width)
            if z is None:
                continue
            per_file_mu.setdefault(z, []).append(mu)
            per_file_sd.setdefault(z, []).append(sd)

        for z in set(per_file_mu) | set(per_file_sd):
            mus = np.asarray(per_file_mu.get(z, []), float)
            sds = np.asarray(per_file_sd.get(z, []), float)
            if mus.size >= min_examples_per_file:
                mu_mean = float(np.mean(mus))
                sd_mean = float(np.mean(sds)) if sds.size else float("nan")
                sd_within = float(np.std(sds, ddof=1)) if sds.size > 1 else 0.0

                pooled_mu_means.setdefault(z, []).append(mu_mean)
                pooled_sd_means.setdefault(z, []).append(sd_mean)
                pooled_sd_within.setdefault(z, []).append(sd_within)
                files_seen_by_z.setdefault(z, set()).add(fp)

    rows, summary_map = [], {}
    for z in sorted(pooled_mu_means):
        mu_means = np.asarray(pooled_mu_means[z], float)
        sd_means = np.asarray(pooled_sd_means.get(z, []), float)
        sd_within = np.asarray(pooled_sd_within.get(z, []), float)

        mean_means = float(np.mean(mu_means)) if mu_means.size else float("nan")
        std_means  = float(np.std(mu_means, ddof=1)) if mu_means.size > 1 else 0.0

        dt_std_mean = float(np.mean(sd_means)) if sd_means.size else float("nan")
        dt_std_std  = float(np.std(sd_means, ddof=1)) if sd_means.size > 1 else 0.0

        dt_std_within_mean = float(np.mean(sd_within)) if sd_within.size else 0.0
        dt_std_within_std  = float(np.std(sd_within, ddof=1)) if sd_within.size > 1 else 0.0

        rows.append(dict(
            z=int(z),
            mean_means=mean_means,
            std_means=std_means,
            dt_std_mean=dt_std_mean,
            dt_std_std=dt_std_std,
            dt_std_within_mean=dt_std_within_mean,
            dt_std_within_std=dt_std_within_std,
            n_files=int(len(files_seen_by_z.get(z, set()))),
        ))
        summary_map[int(z)] = dict(
            mean_means=mean_means,
            std_means=std_means,
            dt_std_mean=dt_std_mean,
            dt_std_std=dt_std_std,
            dt_std_within_mean=dt_std_within_mean,
            dt_std_within_std=dt_std_within_std,
            n_files=int(len(files_seen_by_z.get(z, set()))),
        )

    cols = ["z","mean_means","std_means","dt_std_mean","dt_std_std",
            "dt_std_within_mean","dt_std_within_std","n_files"]
    summary_df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
    return summary_df[cols], summary_map


# =========================================================
# AUC stats (pooled)
# =========================================================
def get_aucs_stats(file_paths: Iterable[str],
                   *,
                   winner_only: bool = True) -> Tuple[float, float]:
    """Return (mean, std) of AUCs pooled over files."""
    aucs_all = []
    for fp in file_paths:
        df = _read_df(fp)
        if df is None or df.empty or "auc" not in df.columns:
            continue
        use = df[df["winner"] == 1] if (winner_only and "winner" in df.columns) else df
        aucs_all.append(np.asarray(use["auc"], float))
    if not aucs_all:
        return float("nan"), float("nan")
    aucs = np.concatenate(aucs_all)
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1 if aucs.size > 1 else 0))


# =========================================================
# Samplers: intensities, RT, DT, mass accuracy
# =========================================================
def sample_total_intensities(
    timepoints: Sequence[float],
    max_intensity: float,
    *,
    decay_start: float = 1.0,
    decay_end: float = 0.5,
    noise_low: float = 0.95,
    noise_high: float = 1.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate per-timepoint totals:
      trend = max_intensity * linspace(decay_start, decay_end, T)
      noise ~ Uniform[noise_low, noise_high]
      intensities = trend * noise

    Returns: (intensities, trend, noise)
    """
    t = np.asarray(timepoints)
    if t.size == 0:
        raise ValueError("timepoints is empty.")
    if max_intensity < 0:
        raise ValueError("max_intensity must be non-negative.")
    rng = np.random.default_rng() if rng is None else rng

    trend = max_intensity * np.linspace(decay_start, decay_end, t.size, dtype=float)
    noise = rng.uniform(noise_low, noise_high, size=t.shape)
    intensities = trend * noise
    return intensities, trend, noise


def sample_rt_error_mean(T: int,
                         sd_mean: float,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Per-timepoint RT mean error ~ N(0, sd_mean)."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0.0, max(0.0, sd_mean), size=T)


def sample_rt_error_std(T: int,
                        sd_std: float,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Per-timepoint RT std error ~ N(0, sd_std)."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0.0, max(0.0, sd_std), size=T)


def sample_dt_baselines_for_charges(
    charges: Iterable[int],
    dt_map_pf: Dict[int, Dict[str, float]],
    *,
    rng: Optional[np.random.Generator] = None,
    min_sd: float = 1e-9,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    For each charge z:
      μ_file ~ N(mean_means, std_means)
      σ0     ~ N(dt_std_mean, dt_std_std) (then clipped to >0)

    Returns:
      dt_mean_baseline[z], dt_std_baseline[z]
    """
    rng = np.random.default_rng() if rng is None else rng
    dt_mean_baseline: Dict[int, float] = {}
    dt_std_baseline:  Dict[int, float] = {}
    for z in charges:
        s = dt_map_pf[int(z)]
        mu = rng.normal(s["mean_means"], max(0.0, s["std_means"]))
        sd = rng.normal(s["dt_std_mean"], max(0.0, s["dt_std_std"]))
        dt_mean_baseline[int(z)] = float(mu)
        dt_std_baseline[int(z)]  = float(max(min_sd, sd))
    return dt_mean_baseline, dt_std_baseline


def sample_dt_systematic_errors(
    charges: Iterable[int],
    T: int,
    dt_map_pf: Dict[int, Dict[str, float]],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    """
    For each z: systematic per-timepoint error for DT mean ~ N(0, dt_std_within_mean).
    (Shared across sequences if you reuse it.)
    """
    rng = np.random.default_rng() if rng is None else rng
    d: Dict[int, np.ndarray] = {}
    for z in charges:
        within = max(0.0, dt_map_pf[int(z)]["dt_std_within_mean"])
        d[int(z)] = rng.normal(0.0, within, size=T)
    return d


def sample_dt_random_errors(
    charges: Iterable[int],
    T: int,
    dt_map_pf: Dict[int, Dict[str, float]],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    """
    For each z: random per-timepoint error for DT mean ~ N(0, dt_std_within_mean).
    (Unique per protein/sequence.)
    """
    rng = np.random.default_rng() if rng is None else rng
    d: Dict[int, np.ndarray] = {}
    for z in charges:
        within = max(0.0, dt_map_pf[int(z)]["dt_std_within_mean"])
        d[int(z)] = rng.normal(0.0, within, size=T)
    return d


def sample_dt_std_errors(
    charges: Iterable[int],
    T: int,
    dt_map_pf: Dict[int, Dict[str, float]],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    """
    For each z: per-timepoint error for DT std ~ N(0, dt_std_within_mean).
    """
    rng = np.random.default_rng() if rng is None else rng
    d: Dict[int, np.ndarray] = {}
    for z in charges:
        within = max(0.0, dt_map_pf[int(z)]["dt_std_within_mean"])
        d[int(z)] = rng.normal(0.0, within, size=T)
    return d


def sample_ppm_errors(T: int,
                      *,
                      sd: float = 3.0,
                      clip: float = 10.0,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Mass-accuracy ppm errors ~ N(0, sd), clipped to ±clip."""
    rng = np.random.default_rng() if rng is None else rng
    ppm = rng.normal(0.0, max(0.0, sd), size=T)
    return np.clip(ppm, -abs(clip), abs(clip))


# =========================================================
# Orchestrator
# =========================================================
def generate_noise_arrays(
    x_row: pd.Series,
    *,
    charges: Iterable[int],
    dt_map_pf: Dict[int, Dict[str, float]],
    rt_avg_std_of_means: float,
    rt_avg_std_of_stds: float,
    auc_mean: float,
    auc_std: float,
    rt_std_global_mean: Optional[float] = None,  # optional baseline σ_RT
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Build arrays used downstream for simulation.

    Inputs from x_row expected:
      - 'sequence', 'timepoints', 'rates_mean' (log space), 'backexchange', 'RT'
    """
    rng = np.random.default_rng() if rng is None else rng

    # Unpack exemplar row
    sequence   = x_row["sequence"]
    timepoints = np.asarray(x_row["timepoints"])
    rates      = np.exp(np.asarray(x_row["rates_mean"]))  # log->linear
    backex     = x_row["backexchange"]
    rt_raw     = x_row["RT"]
    T          = int(timepoints.size)
    charges    = [int(z) for z in charges]

    # --- RT errors (global) ---
    rt_sys    = sample_rt_error_mean(T, sd_mean=rt_avg_std_of_means, rng=rng)
    rt_random = sample_rt_error_mean(T, sd_mean=rt_avg_std_of_means, rng=rng)
    rt_total_error_mean = rt_sys + rt_random
    rt_total_error_std  = sample_rt_error_std(T, sd_std=rt_avg_std_of_stds, rng=rng)
    rt_sigma_t = None
    if rt_std_global_mean is not None and np.isfinite(rt_std_global_mean):
        rt_sigma_t = np.clip(rt_std_global_mean + rt_total_error_std, 1e-9, None)

    # --- DT baselines and errors (per charge) ---
    dt_mean_baseline, dt_std_baseline = sample_dt_baselines_for_charges(charges, dt_map_pf, rng=rng)
    dts_sys    = sample_dt_systematic_errors(charges, T, dt_map_pf, rng=rng)
    dts_random = sample_dt_random_errors   (charges, T, dt_map_pf, rng=rng)
    dt_total_error_mean = {z: dts_sys[z] + dts_random[z] for z in charges}
    dt_total_error_std  = sample_dt_std_errors(charges, T, dt_map_pf, rng=rng)

    # --- mass-accuracy ppm errors ---
    ppm_errors = sample_ppm_errors(T, sd=3.0, clip=10.0, rng=rng)

    # --- total intensities (AUC-like) ---
    if np.isnan(auc_mean) or np.isnan(auc_std):
        auc_max = 1.0  # fallback neutral scale
    else:
        q25, q75 = norm.ppf([0.25, 0.75], loc=auc_mean, scale=auc_std)
        auc_max = np.clip(abs(rng.normal(auc_mean, auc_std)), q25, q75)
    intensities, intensity_trend, intensity_noise = sample_total_intensities(timepoints, auc_max, rng=rng)

    # Package
    out: Dict[str, Any] = dict(
        # core input echoes
        sequence=sequence,
        timepoints=timepoints,
        rates=rates,
        backexchange=backex,
        rt=rt_raw,
        charges=np.asarray(charges, int),

        # RT components
        rt_sys=rt_sys,
        rt_random=rt_random,
        rt_total_error_mean=rt_total_error_mean,   # (T,)
        rt_total_error_std=rt_total_error_std,     # (T,)
        rt_sigma_baseline=rt_std_global_mean,      # scalar or None
        rt_sigma_t=rt_sigma_t,                     # (T,) or None

        # DT components
        dt_mean_baseline=dt_mean_baseline,         # dict[z] -> scalar
        dt_std_baseline=dt_std_baseline,           # dict[z] -> scalar (>0)
        dts_sys=dts_sys,                           # dict[z] -> (T,)
        dts_random=dts_random,                     # dict[z] -> (T,)
        dt_total_error_mean=dt_total_error_mean,   # dict[z] -> (T,)
        dt_total_error_std=dt_total_error_std,     # dict[z] -> (T,)

        # Intensities
        auc_mean=auc_mean,
        auc_std=auc_std,
        auc_max=auc_max,
        intensities=intensities,                   # (T,)
        intensity_trend=intensity_trend,           # (T,)
        intensity_noise=intensity_noise,           # (T,)

        # Mass accuracy
        ppm_errors=ppm_errors,                     # (T,)
    )
    return out


"""
hdx_prepare_inputs.py

Compute or load:
  1) RT/DT variability summaries (cached in variability_results.pkl)
  2) Noise arrays for simulation (cached in noise_arrays.pkl)

Examples
--------
python hdx_prepare_inputs.py \
  --input-glob "../../HDX_analysis/Lib01/20220212_pH6/hdx_limit-pipeline/resources/10_ic_time_series/*/multibody/*_winner_multibody.df.pkl" \
  --hx-json "../../HX_paper/Dataset_3_MeasurablyStable.json" \
  --hx-index 0 \
  --charges "5:11" \
  --outdir "./outputs" \
  --min-examples 3 \
  --seed 42

Notes
-----
- If cache files exist and --force is not set, they will be loaded.
- --charges accepts "start:stop[:step]" (Python-style, stop is exclusive) or a comma list "5,6,7,8".
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


# -------------------------- helpers --------------------------

def _parse_charges(arg: str) -> np.ndarray:
    """
    Parse charges from "a:b[:s]" (stop exclusive) or "a,b,c".
    Examples: "5:11" -> [5,6,7,8,9,10]; "5:11:2" -> [5,7,9]; "5,6,7"
    """
    arg = arg.strip()
    if ":" in arg:
        parts = [int(x) for x in arg.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid charges spec: {arg}")
        return np.arange(start, stop, step, dtype=int)
    if "," in arg:
        return np.array([int(x) for x in arg.split(",")], dtype=int)
    # single integer
    return np.array([int(arg)], dtype=int)


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _join(outdir: str, name: str) -> str:
    return os.path.join(outdir, name)


# -------------------------- CLI --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute/load RT/DT variability and noise arrays for HDX simulation."
    )
    p.add_argument(
        "--inputs",
        nargs="+",            
        required=True,
        help="Glob(s) for input files. May be given multiple times."
    )
    p.add_argument("--hx-json", required=True,
                   help="Path to Dataset_3_MeasurablyStable.json (or similar).")
    p.add_argument("--hx-index", type=int, default=0,
                   help="Row index to pick from the HX JSON table (default 0).")
    p.add_argument("--charges", type=str, default="5:11",
                   help='Charge states, e.g. "5:11" (stop exclusive) or "5,6,7".')
    p.add_argument("--min-examples", type=int, default=3,
                   help="min_examples_per_file for DT summary filtering.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for noise generation (optional).")
    p.add_argument("--outdir", type=str, default="./outputs",
                   help="Output directory (created if missing).")
    p.add_argument("--var-file", type=str, default="variability_results.pkl",
                   help="Filename for variability results cache (inside outdir).")
    p.add_argument("--noise-file", type=str, default="noise_arrays.pkl",
                   help="Filename for noise arrays cache (inside outdir).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache files exist.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve IO
    _ensure_outdir(args.outdir)
    var_path = _join(args.outdir, args.var_file)
    noise_path = _join(args.outdir, args.noise_file)

    # Expand input files
    files = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            files.extend(matches)
        elif os.path.exists(pat):
            files.append(pat)
    fs = sorted(set(files))
    if not fs:
        print(f"[ERROR] No files matched --input-glob: {args.input_glob}", file=sys.stderr)
        return 2
    print(f"[INFO] Matched {len(fs)} input files.")

    # Load HX table + select row
    try:
        df_hx = pd.read_json(args.hx_json)
    except Exception as e:
        print(f"[ERROR] Could not load HX JSON: {args.hx_json} ({e})", file=sys.stderr)
        return 3
    if not (0 <= args.hx_index < len(df_hx)):
        print(f"[ERROR] --hx-index {args.hx_index} out of range [0, {len(df_hx)-1}]", file=sys.stderr)
        return 4
    x = df_hx.iloc[int(args.hx_index)]

    # Charges + RNG
    charges = _parse_charges(args.charges)
    if charges.size == 0:
        print("[ERROR] Parsed empty charges.", file=sys.stderr)
        return 5
    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    # ---------------- Variability results ----------------
    need_var = args.force or (not os.path.isfile(var_path))
    if need_var:
        print("[INFO] Computing variability results...")
        # rt variability
        rt_avg_std_of_means, rt_avg_std_of_stds, rt_std_global_mean = rt_variability_summary(fs)
        # dt variability per file
        dt_df, dt_map_pf = collect_dt_mean_and_std_summary_per_file(
            fs, min_examples_per_file=int(args.min_examples)
        )
        auc_mean, auc_std = get_aucs_stats(fs)
        # save
        save_variability_results(
            var_path,
            rt_avg_std_of_means, rt_avg_std_of_stds, rt_std_global_mean,
            dt_df, dt_map_pf, auc_mean, auc_std
        )
        print(f"[INFO] Saved variability results → {var_path}")
    else:
        print(f"[INFO] Loading variability results from cache: {var_path}")
        (rt_avg_std_of_means,
         rt_avg_std_of_stds,
         rt_std_global_mean,
         dt_df,
         dt_map_pf,
         auc_mean,
         auc_std) = load_variability_results(var_path)

    # ---------------- Noise arrays ----------------
    need_noise = args.force or (not os.path.isfile(noise_path))
    if need_noise:
        print("[INFO] Generating noise arrays...")
        out = generate_noise_arrays(
            x,
            charges=charges,
            dt_map_pf=dt_map_pf,
            rt_avg_std_of_means=rt_avg_std_of_means,
            rt_avg_std_of_stds=rt_avg_std_of_stds,
            auc_mean=auc_mean,
            auc_std=auc_std,
            rt_std_global_mean=rt_std_global_mean,
            rng=rng,
        )
        save_noise_arrays(out, noise_path)
        print(f"[INFO] Saved noise arrays → {noise_path}")
    else:
        print(f"[INFO] Loading noise arrays from cache: {noise_path}")
        out = load_noise_arrays(noise_path)

    # Minimal summary to stdout
    print("[OK] Done.")
    try:
        # Best-effort peek into structures
        print("  • rt_avg_std_of_means:", float(rt_avg_std_of_means))
        print("  • rt_avg_std_of_stds :", float(rt_avg_std_of_stds))
        print("  • rt_std_global_mean :", float(rt_std_global_mean))
        print("  • dt_df rows         :", int(getattr(dt_df, 'shape', [0])[0]))
        print("  • dt_map_pf keys     :", len(getattr(dt_map_pf, 'keys', lambda: [])()))
        if isinstance(out, dict):
            print("  • noise arrays keys  :", ", ".join(sorted(out.keys())[:8])
                  + (" ..." if len(out.keys()) > 8 else ""))
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    
