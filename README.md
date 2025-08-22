# mHDX Simulation Notebooks

This repository provides a compact, end‑to‑end workflow to (i) quantify empirical variability from HDX‑MS libraries, (ii) sample realistic perturbations from those distributions, (iii) simulate 4‑D HDX tensors (Time × RT × DT × m/z), and (iv) combine signals across experiments.

> **Precomputed data**
> We provide ready‑made outputs so you can skip heavy processing: copy the `datasets/` folder (see link below) into the **repository root** after cloning.

---

## 00_Collect_Stats

**Purpose.** Compute variability summaries from HDX libraries across conditions.

**What it produces.**
- **Retention time (RT)** variability across timepoints and runs (charge‑independent):
  - Typical spread of RT **means** across timepoints.
  - Typical spread of RT **standard deviations** across timepoints.
  - Typical **global RT width** (baseline sigma).
- **Drift time (DT)** statistics per **charge state** (charge‑dependent):
  - Typical DT **means** and their between‑example spread.
  - Typical DT **standard deviations** and their between‑example spread.
  - Within‑timepoint variability factors used to mimic per‑timepoint jitter.
- **Total intensity (AUC/TIC)** distribution across winners to capture dynamic range.

**Output location.** `datasets/variability_results/<library>_<pH>.stats.pkl`

---

## 01_Example_SampleFromStats

**Purpose.** Sample realistic per‑timepoint and per‑charge **features** from the precomputed statistics for each protein in the study (see preprint below).

**What it produces.** For each protein and charge:
- **RT offsets/jitter** (charge‑independent): per‑timepoint shifts of RT mean and width drawn from the empirical spreads.
- **DT baselines and jitter** (charge‑dependent): DT means/widths sampled from charge‑specific distributions, plus within‑timepoint variation.
- **m/z ppm errors:** small per‑timepoint mass errors, clipped to realistic bounds.
- **Per‑timepoint TICs:** a gentle, noisy decay profile for total intensity over time.

**Output location.** `datasets/noise_arrays/Index_XXXXX.pkl` (precomputed for **3,589** proteins from Lib01 pH6).

---

## 02_Example_SimulateHDXData

**Purpose.** Turn one protein’s sampled features into **4‑D HDX tensors** using a physically grounded engine (isotopic envelopes, Poisson‑binomial deuteration, instrument broadening).

**What it produces.**
- **Ideal tensor:** uses time‑averaged RT/DT centers and widths, no ppm error, no backexchange, unit total intensity—captures the theoretical shape.
- **Perturbed tensor:** applies the sampled per‑timepoint RT/DT shifts, ppm errors, and backexchange, then normalizes each timepoint to the sampled TIC.
- **Noisy tensor:** adds voxel‑wise random noise to emulate detector/background effects.

**Visualization.** For each timepoint, the notebook plots (i) the **m/z marginal** (sum over RT×DT) and (ii) an **RT×DT** map (sum over m/z) with consistent axes for side‑by‑side comparison of ideal vs perturbed vs noisy.

---

## 03_Example_CombineTensors

**Purpose.** Build mixtures by **adding multiple 4‑D tensors** onto a unified grid.

**What it does.**
- Creates the **sorted union** of time, RT, DT, and m/z axes.
- Places each input tensor on the union grid without interpolation (bins that share identical labels are summed; disjoint regions remain separate).
- Supports **per‑tensor weights** to reflect different contributions.
- Optionally adds noise to the final combined tensor.

**Output.** A combined tensor plus its union axes, suitable for downstream tasks (e.g., benchmarking or model training).

---

## Datasets

Precomputed statistics and sampled features are available here:

- **SharePoint:** <https://nuwildcat.sharepoint.com/:f:/s/FSM-RocklinLab/EkHh1bqvjSNGrAmvm6LprAgB3ay5RqWZq5ufcm0-lNy6UQ?e=qjftie>

After downloading, copy the `datasets/` directory into the repository root:
- `datasets/variability_results/` — variability summaries per library/pH
- `datasets/noise_arrays/` — sampled feature packs per protein (Lib01 pH6)

---

## Citation

If you use these notebooks or datasets, please cite:

> Ferrari, Á. J. R., *et al.* **Large‑scale discovery, analysis, and design of protein energy landscapes.** bioRxiv (2025). <https://doi.org/10.1101/2025.03.20.644235>

---

## Notes & Tips

- RT grids are discretized at **1 s**; DT grids at **0.06925 ms**. Matching discretization ensures precise bin alignment when combining tensors.
- Set a fixed RNG seed for reproducibility when sampling (`numpy.random.default_rng(seed=...)`).

