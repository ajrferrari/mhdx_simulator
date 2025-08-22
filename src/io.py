from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import json
import numpy as np

@dataclass
class LoadedTensor:
    """Container for a single tensor loaded from an engine NPZ."""
    name: str                      # 'ideal' | 'perturbed' | 'noisy' | 'tensor'
    tensor: np.ndarray             # shape (T, R, D, M)
    timepoints: np.ndarray         # (T,)
    rt_labels: np.ndarray          # (R,)
    dt_labels: np.ndarray          # (D,)
    mz_labels: np.ndarray          # (M,)
    meta: Dict[str, Any]           # parsed meta_json


def available_tensors_in_npz(path: str) -> List[str]:
    """
    Return a list of tensor names present in an engine NPZ.
    Names are normalized to: 'ideal', 'perturbed', 'noisy', 'tensor'.
    """
    key2name = {
        "ideal_tensor": "ideal",
        "perturbed_tensor": "perturbed",
        "noisy_tensor": "noisy",
        "tensor": "tensor",
    }
    with np.load(path, allow_pickle=False) as npz:
        out = []
        for k, nm in key2name.items():
            if k in npz.files and np.array(npz[k]).size > 0:
                out.append(nm)
        return out


def load_engine_tensor_npz(
    path: str,
    which: Optional[str] = None,
) -> LoadedTensor:
    """
    Load a SINGLE tensor + axes + meta from an NPZ saved by your engine.

    Parameters
    ----------
    path : str
        Path to the .npz file.
    which : str or None
        One of {'ideal','perturbed','noisy','tensor'}.
        If None, chooses in this order: 'perturbed' -> 'ideal' -> 'noisy' -> 'tensor'.

    Returns
    -------
    LoadedTensor
    """
    # Helper: prefer these keys; normalize 'which'
    pref_order = ["perturbed", "ideal", "noisy", "tensor"]
    if which is not None:
        which = str(which).lower()
        if which not in {"perturbed", "ideal", "noisy", "tensor"}:
            raise ValueError(f"Invalid 'which'={which!r}")

    name2key = {
        "ideal": "ideal_tensor",
        "perturbed": "perturbed_tensor",
        "noisy": "noisy_tensor",
        "tensor": "tensor",
    }

    def _get_axis(npz, *candidates: str) -> np.ndarray:
        # return the first available candidate; raise if none
        for c in candidates:
            if c in npz.files:
                arr = np.array(npz[c])
                return arr
        raise KeyError(f"Missing axis among candidates: {candidates}")

    with np.load(path, allow_pickle=False) as npz:
        # --- meta ---
        if "meta_json" not in npz.files:
            raise ValueError("NPZ missing 'meta_json'; not a compatible engine save.")
        try:
            meta = json.loads(str(npz["meta_json"][0]))
        except Exception as e:
            raise ValueError(f"Failed to parse meta_json: {e}") from e

        # --- axes (support old/new field names) ---
        timepoints = _get_axis(npz, "timepoints", "t")
        rt_labels  = _get_axis(npz, "rt_labels", "rt")
        dt_labels  = _get_axis(npz, "dt_labels", "dt")
        mz_labels  = _get_axis(npz, "mz_labels", "mz")

        # --- decide which tensor to load ---
        if which is None:
            # pick the first available in preference order
            avail = available_tensors_in_npz(path)
            for candidate in pref_order:
                if candidate in avail:
                    which = candidate
                    break
            if which is None:
                raise ValueError("No tensor arrays found in NPZ (looked for ideal/perturbed/noisy/tensor).")

        key = name2key[which]
        if key not in npz.files:
            raise ValueError(f"Requested tensor '{which}' not found in NPZ.")

        tensor = np.array(npz[key])

    # --- sanity checks on shapes vs axes ---
    T, R, D, M = tensor.shape
    if T != timepoints.size:
        raise ValueError(f"Tensor T={T} vs timepoints size {timepoints.size} mismatch.")
    if R != rt_labels.size:
        raise ValueError(f"Tensor R={R} vs rt_labels size {rt_labels.size} mismatch.")
    if D != dt_labels.size:
        raise ValueError(f"Tensor D={D} vs dt_labels size {dt_labels.size} mismatch.")
    if M != mz_labels.size:
        raise ValueError(f"Tensor M={M} vs mz_labels size {mz_labels.size} mismatch.")

    return LoadedTensor(
        name=which,
        tensor=tensor,
        timepoints=timepoints.astype(float, copy=False),
        rt_labels=rt_labels.astype(float, copy=False),
        dt_labels=dt_labels.astype(float, copy=False),
        mz_labels=mz_labels.astype(float, copy=False),
        meta=meta,
    )



## Helpers


def _meta_array(meta: Dict[str, Any] | None) -> np.ndarray:
    """1-element unicode array → avoids object dtype/pickle on load."""
    return np.array([json.dumps(meta or {})], dtype="U")

def save_combined_npz(
    path: str,
    tensor: np.ndarray,
    t: np.ndarray,
    rt: np.ndarray,
    dt: np.ndarray,
    mz: np.ndarray,
    *,
    compress: bool = True,
    meta: Dict[str, Any] | None = None,
) -> str:
    """Lightweight functional saver (no combiner instance required)."""
    # (Optional) sanity checks
    T, R, D, M = tensor.shape
    if T != len(t) or R != len(rt) or D != len(dt) or M != len(mz):
        raise ValueError("Tensor shape does not match provided axes.")

    payload = dict(
        tensor=np.asarray(tensor),
        t=np.asarray(t, dtype=float),
        rt=np.asarray(rt, dtype=float),
        dt=np.asarray(dt, dtype=float),
        mz=np.asarray(mz, dtype=float),
        meta_json=_meta_array(meta),
    )
    (np.savez_compressed if compress else np.savez)(path, **payload)
    return path

def load_combined_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Functional loader returning (tensor, t, rt, dt, mz, meta_dict)."""
    # load numeric arrays without pickle
    with np.load(path, allow_pickle=False) as npz:
        tensor = np.array(npz["tensor"])
        t = np.array(npz["t"], dtype=float)
        rt = np.array(npz["rt"], dtype=float)
        dt = np.array(npz["dt"], dtype=float)
        mz = np.array(npz["mz"], dtype=float)
        meta_json = npz.get("meta_json", None)

    # parse meta (robust to legacy object-typed saves)
    meta: Dict[str, Any] = {}
    if meta_json is not None:
        try:
            s = np.ravel(meta_json)[0]
            s = s.decode("utf-8", "replace") if isinstance(s, (bytes, np.bytes_)) else str(s)
            meta = json.loads(s)
        except Exception:
            # legacy fallback: reopen allowing pickle just for meta_json
            with np.load(path, allow_pickle=True) as npz2:
                s = np.ravel(npz2["meta_json"])[0]
                s = s.decode("utf-8", "replace") if isinstance(s, (bytes, np.bytes_)) else str(s)
                try:
                    meta = json.loads(s)
                except Exception:
                    meta = {"raw_meta_json": s}

    return tensor, t, rt, dt, mz, meta
