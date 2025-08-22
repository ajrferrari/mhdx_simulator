# hdx_timecourse_engine.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union, Dict
import numpy as np
from molmass import Formula

import json


# ==== constants ====
PROTON_MASS   = 1.00727646688     # Da
C13_C12_DIFF  = 1.00335483507     # Da
D_H_MASS_DIFF = 1.006276746       # Da
SQRT_8LN2     = 2.35482004503     # FWHM = SQRT_8LN2 * sigma

SEC_PER_MIN   = 60.0
RT_STEP_MIN   = 1.0 / SEC_PER_MIN     # minutes (1 s)
DT_STEP_MS    = 0.06925               # milliseconds per step (instrument sampling)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _as_time_array(x: Union[float, Sequence[float]], T: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(T, float(arr))
    if arr.size != T:
        raise ValueError(f"{name} must be scalar or length T={T}; got {arr.size}")
    return arr

def _round_to_step(value: np.ndarray, step: float) -> np.ndarray:
    """Round values to the nearest multiple of 'step'."""
    return np.round(np.asarray(value) / step) * step

def _floor_to_step(value: float, step: float) -> float:
    """Floor 'value' to the nearest lower multiple of 'step'."""
    return np.floor(value / step) * step

# -----------------------------------------------------------------------------
# Core engine (new class)
# -----------------------------------------------------------------------------
class HDXTimecourseEngine:
    """
    Compose 4D HDX-MS tensors (T, R, D, M) with a clean 3-stage pipeline:
      1) ideal_tensor     : mean RT/DT across time, backexchange=0, ppm=0, max=1
      2) perturbed_tensor : per-timepoint RT/DT μ/σ, ppm error, backexchange, TIC
      3) noisy_tensor     : integer noise added voxel-wise

    Parameters
    ----------
    sequence : str
        Protein sequence (string used by molmass.Formula).
    timepoints : Iterable[float]
        (T,) timepoints; same unit used for rate constants ks.
    ks : Iterable[float]
        (N,) per-site exchange rates.
    charge : int
        Charge state (z).
    backexchange : None|float|Iterable[float], optional
        If provided, per-timepoint fraction in [0,1]. Scalar is broadcast to T.
    ppm_error : float|Iterable[float], default 0
        Scalar or (T,) ppm mass error (applied in stage 2).
    # RT
    rt_center : float (minutes)
    rt_mean   : float|(T,) array (minutes) — used in stage 2; stage 1 uses mean(rt_mean)
    rt_std    : float|(T,) array (minutes) — used in stage 2; stage 1 uses mean(rt_std)
    # DT
    dt_center : float (milliseconds)
    dt_mean   : float|(T,) array (milliseconds) — used in stage 2; stage 1 uses mean(dt_mean)
    dt_std    : float|(T,) array (milliseconds) — used in stage 2; stage 1 uses mean(dt_std)
    # Tensor size
    rt_radius : float (minutes) — RT grid spans [rt_center ± rt_radius], sampled every 1 s
    dt_radius_scale : float (unitless) — DT grid spans [dt_center*(1−s), dt_center*(1+s)],
                      sampled every 0.06925 ms
    # TIC
    tics : None|float|Iterable[float]
        If None in stage 1, tensor is max-normalized to 1. In stage 2, if provided,
        each timepoint is scaled so sum equals tics[ti].
    # m/z & profiles
    resolving_power : float, default 25_000
    min_profile_intensity : float, default 1e-3
        Threshold for RT/DT tails (values below set to 0).
    iso_min_rel : float, default 1e-3
        Minimum relative stick intensity to keep from molmass spectrum.
    iso_n_isotopes : int|None, default None
        Optional cap on number of base isotopic sticks.
    grid_step_mz : float|None, default 0.001
        If None, an automatic step is chosen from resolving power.
        If provided (e.g. 0.001), the m/z grid START is floored to a multiple of the step,
        e.g., 500.14455 → 500.144, 500.145, 500.146, ...

    Attributes (after generation)
    -----------------------------
    rt_labels : (R,) minutes
    dt_labels : (D,) milliseconds
    mz_labels : (M,) m/z
    ideal_tensor     : (T, R, D, M) float
    perturbed_tensor : (T, R, D, M) float
    noisy_tensor     : (T, R, D, M) float
    """

    def __init__(
        self,
        *,
        sequence: str,
        timepoints: Iterable[float],
        ks: Iterable[float],
        charge: int,
        backexchange: Optional[Union[float, Iterable[float]]] = None,
        d2o_fraction: float = 1,
        d2o_purity: float = 1, 
        ppm_error: Union[float, Iterable[float]] = 0.0,
        # RT
        rt_center: float,
        rt_mean: Union[float, Sequence[float]],
        rt_std: Union[float, Sequence[float]],
        # DT
        dt_center: float,
        dt_mean: Union[float, Sequence[float]],
        dt_std: Union[float, Sequence[float]],
        # Tensor size
        rt_radius: float = 0.4,
        dt_radius_scale: float = 0.06,
        # TIC
        tics: Optional[Union[float, Iterable[float]]] = None,
        # m/z & profiles
        resolving_power: float = 25000.0,
        min_profile_intensity: float = 1e-2,
        iso_min_rel: float = 1e-3,
        iso_n_isotopes: Optional[int] = None,
        grid_step_mz: Optional[float] = 0.001,
    ) -> None:

        # --- inputs ---
        self.sequence   = str(sequence)
        self.timepoints = np.asarray(list(timepoints), dtype=float)
        self.ks         = np.asarray(ks, dtype=float)
        self.charge     = int(charge)
        self.T          = self.timepoints.size

        # Optional backexchange (broadcast/scalar/array or None)
        if backexchange is None:
            self.backexchange = None
        else:
            be = np.asarray(backexchange, dtype=float)
            self.backexchange = np.full(self.T, float(be)) if be.ndim == 0 else be
            if self.backexchange.size != self.T:
                raise ValueError("backexchange must be None, scalar, or length T.")
                
        self.d2o_fraction = d2o_fraction
        self.d2o_purity = d2o_purity

        # ppm error (scalar or array)
        pe = np.asarray(ppm_error, dtype=float)
        self.ppm_error = np.full(self.T, float(pe)) if pe.ndim == 0 else pe
        if self.ppm_error.size != self.T:
            raise ValueError("ppm_error must be scalar or length T.")

        # RT/DT parameters (arrays of length T)
        self.rt_center = float(rt_center)
        self.dt_center = float(dt_center)
        self.rt_mean_arr = _as_time_array(rt_mean, self.T, "rt_mean")
        self.rt_std_arr  = _as_time_array(rt_std,  self.T, "rt_std")
        self.dt_mean_arr = _as_time_array(dt_mean, self.T, "dt_mean")
        self.dt_std_arr  = _as_time_array(dt_std,  self.T, "dt_std")

        # Snap RT means to seconds; DT means to instrument step
        self.rt_mean_arr = _round_to_step(self.rt_mean_arr, RT_STEP_MIN)
        self.dt_mean_arr = _round_to_step(self.dt_mean_arr, DT_STEP_MS)

        # Grid spans
        self.rt_radius       = float(rt_radius)
        self.dt_radius_scale = float(dt_radius_scale)

        # TICs
        if tics is None:
            self.tics = None
        else:
            t = np.asarray(tics, dtype=float)
            self.tics = np.full(self.T, float(t)) if t.ndim == 0 else t
            if self.tics.size != self.T:
                raise ValueError("tics must be None, scalar, or length T.")

        # m/z & options
        self.resolving_power       = float(resolving_power)
        self.min_profile_intensity = float(min_profile_intensity)
        self.iso_min_rel           = float(iso_min_rel)
        self.iso_n_isotopes        = iso_n_isotopes
        self.grid_step_mz          = None if grid_step_mz is None else float(grid_step_mz)

        # --- outputs / grids ---
        self.rt_labels: Optional[np.ndarray] = None
        self.dt_labels: Optional[np.ndarray] = None
        self.mz_labels: Optional[np.ndarray] = None

        self.ideal_tensor: Optional[np.ndarray] = None
        self.perturbed_tensor: Optional[np.ndarray] = None
        self.noisy_tensor: Optional[np.ndarray] = None

        # cache (for ratio-updates)
        self._mztc_ideal: Optional[np.ndarray] = None  # (T, M)
        self._rt_prof_ideal: Optional[np.ndarray] = None  # (T, R)
        self._dt_prof_ideal: Optional[np.ndarray] = None  # (T, D)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def generate_ideal_tensor(self, rtn=False) -> np.ndarray:
        """
        Stage 1: Build the *ideal* tensor using:
            • RT mean/std = averages over timepoints (scalars),
            • backexchange = 0,
            • ppm_error = 0,
            • max-normalize tensor to 1.
        Grids:
            • RT grid: [rt_center ± rt_radius], step=1 s (minutes)
            • DT grid: [dt_center*(1−s) .. *(1+s)], step=0.06925 ms
            • m/z grid: fixed step (default 0.001) with floored start to step multiple.
        """
        # 1) base isotopic sticks & m/z grid
        masses, base_sticks = self._isotopic_sticks_from_sequence(
            self.sequence, min_rel=self.iso_min_rel, n_isotopes=self.iso_n_isotopes
        )
        if masses.size == 0:
            raise ValueError("No isotopic sticks generated for sequence.")
        mz_min, mz_max = self._infer_mz_window_strict(masses, self.sequence, self.charge)
        mz_grid = self._make_mz_grid(mz_min, mz_max, self.resolving_power, self.grid_step_mz)

        # 2) mass timecourse with BE=0
        mztc_ideal = self._timecourse_profiles_over_mz(
            mz_grid, masses, base_sticks, float(masses[0]), self.charge,
            self.timepoints, self.ks, backexchange=None,
            d2o_purity=self.d2o_purity, d2o_fraction=self.d2o_fraction,
            resolving_power=self.resolving_power
        )  # (T, M)

        # 3) RT/DT grids with scalar means/stds = averages (rounded where required)
        rt_mu0 = float(np.mean(self.rt_mean_arr))
        rt_sd0 = float(np.mean(self.rt_std_arr))
        dt_mu0 = float(np.mean(self.dt_mean_arr))
        dt_sd0 = float(np.mean(self.dt_std_arr))

        # snap means to grid steps
        rt_mu0 = float(_round_to_step(rt_mu0, RT_STEP_MIN))
        dt_mu0 = float(_round_to_step(dt_mu0, DT_STEP_MS))

        rt_g, rt_prof = self._rt_profiles_fixed_grid(self.rt_center,
                                                     np.full(self.T, rt_mu0),
                                                     np.full(self.T, rt_sd0),
                                                     self.rt_radius)
        dt_g, dt_prof = self._dt_profiles_fixed_grid(self.dt_center,
                                                     np.full(self.T, dt_mu0),
                                                     np.full(self.T, dt_sd0),
                                                     self.dt_radius_scale)

        # 4) Compose 4D tensor
        T, M = mztc_ideal.shape
        R, D = rt_g.size, dt_g.size
        tensor = np.zeros((T, R, D, M), dtype=float)
        for ti in range(T):
            rd = rt_prof[ti][:, None] * dt_prof[ti][None, :]  # (R, D)
            tensor[ti] = rd[:, :, None] * mztc_ideal[ti][None, None, :]

        # 5) Max-normalize to 1
        mx = tensor.max()
        if mx > 0:
            tensor /= mx

        # Store
        self.rt_labels = rt_g
        self.dt_labels = dt_g
        self.mz_labels = mz_grid
        self.ideal_tensor = tensor
        self._mztc_ideal = mztc_ideal
        self._rt_prof_ideal = rt_prof
        self._dt_prof_ideal = dt_prof
        if rtn:
            return tensor 

    def apply_perturbations(self, rtn=False) -> np.ndarray:
        """
        Stage 2: Apply *ignored* perturbations on top of the ideal tensor:
            • per-timepoint RT/DT mean & std,
            • backexchange (if provided),
            • ppm error per timepoint (warp along m/z),
            • TIC scaling if tics is provided (per timepoint).
        """
        if self.ideal_tensor is None or self._mztc_ideal is None:
            raise RuntimeError("Call generate_ideal_tensor() first.")

        sim_rt_g, sim_dt_g, mz_g = self.rt_labels, self.dt_labels, self.mz_labels
        T, R, D, M = self.ideal_tensor.shape
        out = self.ideal_tensor.copy()
        eps = 1e-8

        # (a) RT/DT profile reweighting via ratio (per timepoint)
        rt_mu_new = _round_to_step(self.rt_mean_arr, RT_STEP_MIN)
        dt_mu_new = _round_to_step(self.dt_mean_arr, DT_STEP_MS)
        rt_sd_new = self.rt_std_arr.copy()
        dt_sd_new = self.dt_std_arr.copy()


        # (a) Recompute per-timepoint RT/DT profiles (already in your code)
        _, rt_prof_new = self._rt_profiles_fixed_grid(self.rt_center, rt_mu_new, rt_sd_new, self.rt_radius)
        _, dt_prof_new = self._dt_profiles_fixed_grid(self.dt_center, dt_mu_new, dt_sd_new, self.dt_radius_scale)

        # (b) Recompute mass timecourse with backexchange 
        mztc_new = self._timecourse_profiles_over_mz(
            mz_g, *self._base_isotopic_cache(self.sequence),
            float(self._base_isotopic_cache(self.sequence)[0][0]),
            self.charge, self.timepoints, self.ks,
            d2o_fraction=self.d2o_fraction, d2o_purity=self.d2o_purity,
            backexchange=self.backexchange, resolving_power=self.resolving_power
        )  # (T, M)

        # (c) Optional ppm warp on the 1D mass profile, then compose RD × M
        out = np.zeros_like(self.ideal_tensor)
        for ti in range(T):
            rd = rt_prof_new[ti][:, None] * dt_prof_new[ti][None, :]        # (R,D)
            m_prof = mztc_new[ti].copy()                                     # (M,)
            

            ppm = float(self.ppm_error[ti]) * 1e-6
            if ppm != 0.0:
                inv_scale = 1.0 / (1.0 + ppm)                                # f_new(x)=f_old(x/(1+ppm))
                sample_x = mz_g * inv_scale
                # edge-fill to avoid artificial truncation at the boundaries
                m_prof = np.interp(sample_x, mz_g, m_prof, left=m_prof[0], right=m_prof[-1])

            out[ti] = rd[:, :, None] * m_prof[None, None, :]

        # (d) Optional TIC scaling
        if self.tics is not None:
            for ti in range(T):
                s = np.max(out[ti].sum(axis=(0,1)))
                tgt = self.tics[ti]
                if s > 0 and np.isfinite(tgt):
                    out[ti] *= (tgt / s)

        self.perturbed_tensor = out
        if rtn:
            return out  
    
    def add_random_noise(
        self, *, low: int = 0, high: int = 30, rng: Optional[np.random.Generator] = None,
        return_noise: bool = False,
        rtn: bool = False
    ):
        """
        Stage 3: Add integer noise U{low..high} to the *perturbed* tensor.
        """
        if self.perturbed_tensor is None:
            raise RuntimeError("Call apply_perturbations() first.")
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.integers(low, high + 1, size=self.perturbed_tensor.shape, dtype=np.int16)
        noisy = self.perturbed_tensor.astype(np.float32, copy=True)
        noisy += noise
        np.maximum(noisy, 0.0, out=noisy)
        self.noisy_tensor = noisy
        if return_noise:
            return (noisy, noise) 
        if rtn:
            return noisy
    
    @staticmethod
    def _meta_array(meta: dict) -> np.ndarray:
        """
        Store meta as a 1-element *unicode* array (dtype='U') to avoid
        object dtype → pickle on load.
        """
        import json
        import numpy as np
        return np.array([json.dumps(meta)], dtype="U")
    
    
    def save_npz(self, path: str) -> None:
        """
        Save engine state (parameters, axes, and any computed tensors) to a single .npz.
    
        Always saved
        ------------
        - timepoints, ks
        - backexchange (empty array if None)
        - ppm_error
        - rt_mean_arr, rt_std_arr, dt_mean_arr, dt_std_arr
        - tics (empty array if None)
        - meta_json (engine/scalar params, sequence, charge, steps)
    
        Saved if available
        ------------------
        - rt_labels, dt_labels, mz_labels (empty arrays if not generated)
        - ideal_tensor, perturbed_tensor, noisy_tensor
        """
        import numpy as np
    
        # if you already have SEC_PER_MIN in scope
        RT_STEP_MIN = 1.0 / SEC_PER_MIN
    
        meta = dict(
            engine_version=1,
            sequence=self.sequence,
            charge=int(self.charge),
            resolving_power=float(self.resolving_power),
            min_profile_intensity=float(self.min_profile_intensity),
            iso_min_rel=float(self.iso_min_rel),
            iso_n_isotopes=None if self.iso_n_isotopes is None else int(self.iso_n_isotopes),
            grid_step_mz=None if self.grid_step_mz is None else float(self.grid_step_mz),
            rt_center_min=float(self.rt_center),
            rt_radius_min=float(self.rt_radius),
            dt_center_ms=float(self.dt_center),
            dt_radius_scale=float(self.dt_radius_scale),
            dt_step_ms=float(DT_STEP_MS),
            rt_step_min=float(RT_STEP_MIN),
        )
    
        payload = {
            "timepoints": self.timepoints.astype(float, copy=False),
            "ks":         self.ks.astype(float, copy=False),
            "backexchange": (np.array([], dtype=float) if self.backexchange is None
                             else np.asarray(self.backexchange, dtype=float)),
            "ppm_error":   np.asarray(self.ppm_error, dtype=float),
            "rt_mean_arr": np.asarray(self.rt_mean_arr, dtype=float),
            "rt_std_arr":  np.asarray(self.rt_std_arr, dtype=float),
            "dt_mean_arr": np.asarray(self.dt_mean_arr, dtype=float),
            "dt_std_arr":  np.asarray(self.dt_std_arr, dtype=float),
            "tics":        (np.array([], dtype=float) if self.tics is None
                            else np.asarray(self.tics, dtype=float)),
            # >>> changed line: use unicode array instead of object array
            "meta_json":   self._meta_array(meta),
            # axes (empty arrays if not present)
            "rt_labels": (np.array([], dtype=float) if self.rt_labels is None else self.rt_labels),
            "dt_labels": (np.array([], dtype=float) if self.dt_labels is None else self.dt_labels),
            "mz_labels": (np.array([], dtype=float) if self.mz_labels is None else self.mz_labels),
        }
    
        # tensors (only if present)
        if self.ideal_tensor is not None:
            payload["ideal_tensor"] = self.ideal_tensor
        if self.perturbed_tensor is not None:
            payload["perturbed_tensor"] = self.perturbed_tensor
        if self.noisy_tensor is not None:
            payload["noisy_tensor"] = self.noisy_tensor
    
        np.savez_compressed(path, **payload)
        
    @classmethod
    def load_npz(cls, path: str) -> "HDXTimecourseEngine":
        """
        Load engine state from an .npz file created by `save_npz`.
        Returns an HDXTimecourseEngine with parameters restored and
        any saved tensors/axes attached. Does NOT recompute.
        """
        with np.load(path, allow_pickle=False) as npz:
            # metadata
            if "meta_json" not in npz.files:
                raise ValueError("NPZ missing 'meta_json'; not a compatible save.")
            meta = json.loads(str(npz["meta_json"][0]))

            # required arrays
            timepoints = np.array(npz["timepoints"], dtype=float)
            ks         = np.array(npz["ks"], dtype=float)

            # optional arrays (None if empty)
            def _maybe(arrname):
                arr = np.array(npz[arrname])
                return None if arr.size == 0 else arr

            backexchange = _maybe("backexchange")
            tics         = _maybe("tics")

            # per-timepoint params
            rt_mean_arr = np.array(npz["rt_mean_arr"], dtype=float)
            rt_std_arr  = np.array(npz["rt_std_arr"], dtype=float)
            dt_mean_arr = np.array(npz["dt_mean_arr"], dtype=float)
            dt_std_arr  = np.array(npz["dt_std_arr"], dtype=float)
            ppm_error   = np.array(npz["ppm_error"], dtype=float)

            # construct engine
            eng = cls(
                sequence=meta["sequence"],
                timepoints=timepoints,
                ks=ks,
                charge=int(meta["charge"]),
                backexchange=backexchange,
                ppm_error=ppm_error,
                rt_center=float(meta["rt_center_min"]),
                rt_mean=rt_mean_arr,
                rt_std=rt_std_arr,
                dt_center=float(meta["dt_center_ms"]),
                dt_mean=dt_mean_arr,
                dt_std=dt_std_arr,
                rt_radius=float(meta["rt_radius_min"]),
                dt_radius_scale=float(meta["dt_radius_scale"]),
                tics=tics,
                resolving_power=float(meta["resolving_power"]),
                min_profile_intensity=float(meta["min_profile_intensity"]),
                iso_min_rel=float(meta["iso_min_rel"]),
                iso_n_isotopes=(None if meta["iso_n_isotopes"] is None
                                else int(meta["iso_n_isotopes"])),
                grid_step_mz=(None if meta["grid_step_mz"] is None
                              else float(meta["grid_step_mz"])),
            )

            # attach axes if present
            eng.rt_labels = _maybe("rt_labels")
            eng.dt_labels = _maybe("dt_labels")
            eng.mz_labels = _maybe("mz_labels")

            # attach tensors if present
            if "ideal_tensor" in npz.files:
                eng.ideal_tensor = np.array(npz["ideal_tensor"])
            if "perturbed_tensor" in npz.files:
                eng.perturbed_tensor = np.array(npz["perturbed_tensor"])
            if "noisy_tensor" in npz.files:
                eng.noisy_tensor = np.array(npz["noisy_tensor"])

        return eng

    # -------------------------------------------------------------------------
    # Internal methods (ported from your working class, kept accurate)
    # -------------------------------------------------------------------------
    def _isotopic_sticks_from_sequence(
        self, sequence: str, min_rel: float = 1e-3, n_isotopes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        spec = Formula(sequence).spectrum(min_intensity=min_rel)
        if not spec:
            return np.array([]), np.array([])
        pairs = np.array([(i.mass, i.intensity) for i in spec.values()], dtype=float)
        masses, intens = pairs[:, 0], pairs[:, 1]
        order = np.argsort(masses)
        masses, intens = masses[order], intens[order]
        intens = intens / intens.max()
        if n_isotopes is not None:
            K = len(masses)
            if n_isotopes < K:
                masses = masses[:n_isotopes]
                intens = intens[:n_isotopes]
            elif n_isotopes > K:
                pad = n_isotopes - K
                extra_masses = masses[-1] + np.arange(1, pad + 1) * C13_C12_DIFF
                masses = np.concatenate([masses, extra_masses])
                intens = np.concatenate([intens, np.zeros(pad, dtype=float)])
        return masses, intens

    def _base_isotopic_cache(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        # quick 1-call cache (could be expanded if needed)
        if not hasattr(self, "_base_iso_cache"):
            self._base_iso_cache = self._isotopic_sticks_from_sequence(
                sequence, min_rel=self.iso_min_rel, n_isotopes=self.iso_n_isotopes
            )
        return self._base_iso_cache

    @staticmethod
    def _n_exchangeable(sequence: str) -> int:
        s = sequence.upper()
        return max(len(s) - s.count("P") - 2, 0)

    def _infer_mz_window_strict(self, masses: np.ndarray, sequence: str, z: int) -> Tuple[float, float]:
        if masses.size == 0:
            raise ValueError("Empty masses array in _infer_mz_window_strict")
        mono_mass = float(masses[0])
        last_mass = float(masses[-1])
        maxD = self._n_exchangeable(sequence)
        zf = float(z)
        mz_min = (mono_mass + zf * PROTON_MASS) / zf - 0.5 / zf
        mz_max = (last_mass + maxD * D_H_MASS_DIFF + zf * PROTON_MASS) / zf #+ 0.1 / zf
        return mz_min, mz_max

    def _make_mz_grid(self, mz_min: float, mz_max: float, resolving_power: float, grid_step_mz: Optional[float]) -> np.ndarray:
        if grid_step_mz is None:
            fwhm_min = max(mz_min, 1.0) / resolving_power
            step = max(fwhm_min / 5.0, 0.002)
        else:
            step = float(grid_step_mz)
        start = _floor_to_step(mz_min, step)  # anchor start to step multiple
        # ensure coverage
        return np.arange(start, mz_max + step * 1.5, step, dtype=float)

    @staticmethod
    def _poibin_pmf(p: np.ndarray) -> np.ndarray:
        """Poisson-binomial PMF via FFT, with numeric clipping/renorm. (Your working version)"""
        p = np.asarray(p, dtype=float)
        n = p.size
        omega = 2 * np.pi / (n + 1)
        chi = np.empty(n + 1, dtype=complex)
        chi[0] = 1
        half = int(n / 2 + n % 2)

        expv = np.exp(omega * np.arange(1, half + 1) * 1j)[:, None]
        xy   = 1 - p + p * expv
        argz = np.arctan2(xy.imag, xy.real).sum(axis=1)
        dval = np.exp(np.log(np.abs(xy)).sum(axis=1))
        chi[1:half + 1] = dval * np.exp(1j * argz)
        chi[half + 1:n + 1] = np.conjugate(chi[1:n - half + 1][::-1])

        chi /= (n + 1)
        pmf = np.fft.fft(chi).real
        pmf = np.clip(pmf, 0.0, None)
        s = pmf.sum()
        if s > 0:
            pmf /= s
        return pmf  # (n+1,)

    @staticmethod
    def _place_broadened_sticks(
        mz_grid: np.ndarray, mz_centroids: np.ndarray, amps: np.ndarray, resolving_power: float
    ) -> np.ndarray:
        out = np.zeros_like(mz_grid, dtype=float)
        for mz_i, a in zip(mz_centroids, amps):
            if a <= 0:
                continue
            sigma = (mz_i / resolving_power) / SQRT_8LN2
            lo = np.searchsorted(mz_grid, mz_i - 5 * sigma)
            hi = np.searchsorted(mz_grid, mz_i + 5 * sigma)
            if lo >= hi:
                continue
            x = mz_grid[lo:hi]
            out[lo:hi] += a * np.exp(-0.5 * ((x - mz_i) / sigma) ** 2)
        return out

    def _timecourse_profiles_over_mz(
        self,
        mz_grid: np.ndarray,
        neutral_masses: np.ndarray,
        base_sticks: np.ndarray,
        mono_mass: float,
        z: int,
        times: np.ndarray,
        ks: np.ndarray,
        backexchange: Optional[np.ndarray],
        d2o_fraction: float,
        d2o_purity: float,
        resolving_power: float,
    ) -> np.ndarray:
        """(T, M) mass timecourse via Poisson–binomial + index-domain convolution + broadening."""
        T = times.size
        out = np.zeros((T, mz_grid.size), dtype=float)
        zf = float(z)
        
        # ensure base sticks are normalized and positive
        a0 = np.asarray(base_sticks, float).clip(min=0.0)
        if a0.sum() > 0:
            a0 = a0 / a0.sum()

        I = a0.size

        for ti, t in enumerate(times):
            # per-site success prob at time t
            p = 1.0 - np.exp(-ks * float(t)) * (d2o_fraction * d2o_purity)
            if backexchange is not None:
                p = np.clip((1.0 - float(backexchange[ti])) * p, 0.0, 1.0)

            pmf = self._poibin_pmf(p)                      # (N+1,)         
            
            '''
            The base isotopic envelope already encodes the distribution of ^13C, ^15N, etc. at their true masses (neutral_masses).
            Deuteration adds k\cdot\Delta m_{\mathrm{D-H}} on top; its probability is your Poisson–binomial PMF over the number of exchanged sites.
            The total line spectrum is the cartesian sum of these two discrete sets (outer sum in mass, outer product in intensity).
            '''
            
            K = pmf.size

            # Outer composition of masses and amplitudes:
            # masses_{i,k} = neutral_masses[i] + k * D_H_MASS_DIFF
            # amps_{i,k}   = a0[i] * pmf[k]
            k = np.arange(K, dtype=float)[None, :]                  # (1, K)
            m2 = neutral_masses[:, None] + D_H_MASS_DIFF * k        # (I, K)
            a2 = (a0[:, None] * pmf[None, :]).astype(float, copy=False)  # (I, K)

            # prune tiny amplitudes for speed, then renormalize
            thresh = max(self.iso_min_rel, 1e-8) * float(a2.max(initial=0.0))
            if thresh > 0.0:
                mask = a2 >= thresh
                m_list = m2[mask].ravel()
                a_list = a2[mask].ravel()
            else:
                m_list = m2.ravel()
                a_list = a2.ravel()

            s = a_list.sum()
            if s > 0.0:
                a_list /= s

            # convert to m/z centroids and place broadened sticks
            mz_centroids = (m_list + zf * PROTON_MASS) / zf
            out[ti] = self._place_broadened_sticks(mz_grid, mz_centroids, a_list, resolving_power)           

        return out

    # ---- RT/DT per-timepoint profiles on fixed grids ----
    def _rt_profiles_fixed_grid(
        self,
        center_min: float,           # minutes (global grid center)
        means_min: np.ndarray,       # (T,)
        sigmas_min: np.ndarray,      # (T,)
        radius_min: float,           # minutes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a fixed RT grid (minutes) and per-timepoint profiles [T,R]."""
        step_min = RT_STEP_MIN
        c = _round_to_step(center_min, step_min)          # grid center snapped to 1 s
        lo = c - radius_min
        hi = c + radius_min
        k_lo = int(np.floor(lo / step_min))
        k_hi = int(np.ceil(hi / step_min))
        rt_g = (np.arange(k_lo, k_hi + 1, dtype=float) * step_min)

        T = means_min.size
        R = rt_g.size
        profs = np.zeros((T, R), dtype=float)
        for ti in range(T):
            mu = float(_round_to_step(means_min[ti], step_min))  # snap means to 1 s
            s  = float(sigmas_min[ti])
            if s <= 0:
                idx = np.argmin(np.abs(rt_g - mu))
                prof = np.zeros(R, dtype=float); prof[idx] = 1.0
            else:
                prof = np.exp(-0.5 * ((rt_g - mu) / s) ** 2)
                m = prof.max()
                if m > 0:
                    prof /= m
            prof[prof < self.min_profile_intensity] = 0.0
            profs[ti] = prof
        return rt_g, profs

    def _dt_profiles_fixed_grid(
        self,
        center_ms: float,            # milliseconds (global grid center)
        means_ms: np.ndarray,        # (T,)
        sigmas_ms: np.ndarray,       # (T,)
        radius_scale: float,         # unitless
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a fixed DT grid (milliseconds) and per-timepoint profiles [T,D]."""
        step_ms = DT_STEP_MS
        c = _round_to_step(center_ms, step_ms)            # grid center snapped to instrument step
        lo = c * (1.0 - radius_scale)
        hi = c * (1.0 + radius_scale)
        k_lo = int(np.floor(lo / step_ms))
        k_hi = int(np.ceil(hi / step_ms))
        dt_g = (np.arange(k_lo, k_hi + 1, dtype=float) * step_ms)

        T = means_ms.size
        D = dt_g.size
        profs = np.zeros((T, D), dtype=float)
        for ti in range(T):
            mu = float(_round_to_step(means_ms[ti], step_ms))    # snap means to 0.06925 ms
            s  = float(sigmas_ms[ti])
            if s <= 0:
                idx = np.argmin(np.abs(dt_g - mu))
                prof = np.zeros(D, dtype=float); prof[idx] = 1.0
            else:
                prof = np.exp(-0.5 * ((dt_g - mu) / s) ** 2)
                m = prof.max()
                if m > 0:
                    prof /= m
            prof[prof < self.min_profile_intensity] = 0.0
            profs[ti] = prof
        return dt_g, profs
