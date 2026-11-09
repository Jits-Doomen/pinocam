# Physics-Informed Neural Operator (PINO) for All-vs-All Satellite Conjunction Assessment

An end-to-end astrodynamics and scientific machine learning pipeline for large-scale satellite conjunction screening, rapid collision probability (*P*<sub>c</sub>) estimation, and Energy-Optimal Collision Avoidance Maneuver (EOP CAM) synthesis.

---

## 1. Scientific Motivation & Problem Formulation

Traditional operational conjunction assessment relies on numerical 2D Gaussian integration over the Hard-Body Radius (HBR) in the encounter B-plane (E.G., Foster's method). While it is accurate, evaluating thousands of close approaches across an active catalog of more than 15,000 space objects and optimizing avoidance maneuvers for high-risk events is computationally very expensive.

This repository addresses that bottleneck by combining classical relative orbital mechanics with a Physics-Informed Neural Operator (PINO):
* **Log-Space Surrogate Modeling:** Because operational collision probabilities span over 20 orders of magnitude (10<sup>−20</sup> ≤ *P*<sub>c</sub> ≤ 1), direct linear regression suffers from severe floating-point underflow and gradient dominance. The network predicts log<sub>10</sub>(*P*<sub>c</sub>) directly, regularized by an analytical physics residual term.
* **Sub-Millisecond ONNX Inference:** The trained PyTorch model is exported to ONNX (`conjunction_pino.onnx`) for deterministic, low-latency execution in both Python and native C++ pipelines.
* **Closed-Form Optimal Maneuver Synthesis:** For conjunctions exceeding risk thresholds, optimal impulsive Δ**v** maneuvers are computed analytically in the B-plane by solving a quartic Lagrange-multiplier polynomial coupled with Clohessy-Wiltshire (CW) state transition matrices.

---

## 2. Repository Architecture

```text
├── all_vs_all_screening.py     # Live catalog propagation (SGP4), cKDTree filtering, ONNX inference & EOP CAM solver
├── conjunction_pino.onnx       # Exported ONNX model graph
├── conjunction_pino.onnx.data  # Externalized model weights
├── normalization_params.txt    # Training set feature means and standard deviations (SI units: m, m/s)
└── extra/
    ├── cdm_gen.py              # Synthetic CCSDS 508.0-B-1 (KVN) Conjunction Data Message generator
    ├── pc2d_foster.py          # Exact 2D Foster B-plane collision probability integrator
    ├── train_pino.py           # Physics-regularized PyTorch training pipeline & ONNX exporter
    ├── benchmark_pino.py       # Latency and accuracy benchmarking against ground-truth Foster integration
    ├── validate_pc.py          # Numerical validation of Pc edge cases and covariance regimes
    ├── verify.py               # Formal SymPy symbolic & dimensional verification of B-plane & EOP CAM math
    ├── main.cpp                # Native C++ entry point
    ├── pino.cpp / pino.h       # Core C++ astrodynamics & data structures
    ├── pino_inference.cpp / .h # C++ ONNX Runtime C-API inference wrapper
    ├── CMakeLists.txt          # CMake build configuration
    ├── CMakePresets.json       # Build presets (x64-release / debug)
    └── vcpkg.json              # C++ dependency manifest (ONNX Runtime, Eigen/CSV utilities)
```

---

## 3. Mathematical & Physical Methodology

### 3.1 Coordinate Frames & B-Plane Projection (`calculate_bplane_parameters`)
Relative states between primary and secondary objects at Time of Closest Approach (TCA) are transformed from the Local-Vertical Local-Horizontal (**LVLH**) frame (**x̂** radial, **ŷ** along-track, **ẑ** cross-track) into Earth-Centered Inertial (**ECI**) coordinates, and subsequently projected onto the 2D **encounter B-plane** perpendicular to the relative velocity vector **v**<sub>rel</sub>:

```text
t_hat = v_rel / ||v_rel||
n_hat = (r_rel × v_rel) / ||r_rel × v_rel||
r_hat = t_hat × n_hat
```

The 2 × 3 projection operator **M**<sub>rot</sub> = [**n̂**  **r̂**]<sup>T</sup> maps the 3D ECI relative position and combined position covariance into the 2D B-plane:

```text
b_p = M_rot · r_rel                                      (2 × 1 B-plane miss vector)
C_B = M_rot · (R_LVLH_to_ECI · C_LVLH · R_LVLH_to_ECI^T) · M_rot^T   (2 × 2 B-plane covariance)
```

### 3.2 Physics-Informed Neural Operator (`train_pino.py`)
* **Input Feature Vector (**x** ∈ ℝ<sup>5</sup>, standardized in SI units):**
    1. `miss_distance_m` (‖**r**<sub>rel</sub>‖ in m)
    2. `rel_speed_m_s` (‖**v**<sub>rel</sub>‖ in m/s)
    3. `sigma_primary_m` (characteristic primary position uncertainty in m)
    4. `sigma_secondary_m` (characteristic secondary position uncertainty in m)
    5. `hbr_m` (combined Hard-Body Radius in m)
* **Target Output:** *ŷ* = log<sub>10</sub>(*P*<sub>c</sub>) ∈ [−20, 0].
* **Composite Loss Function:**

```text
L_total = MSE(y_hat, log10(P_c,Foster)) + lambda_phys(epoch) · L_physics(y_hat, x)
```

Trained with `AdamW` and a `CosineAnnealingLR` schedule (10<sup>−3</sup> → 10<sup>−5</sup>) with linear physics-loss warmup.

### 3.3 Energy-Optimal Collision Avoidance Maneuver (`compute_eop_bplane_cam`)
To reduce collision risk below a target Squared Mahalanobis Distance threshold (SMD<sub>bar</sub> = 9.0, corresponding to a 3σ B-plane separation), an impulsive maneuver Δ**v** applied at lead time Δ*t*<sub>maneuver</sub> = 3600 s is propagated via the 6 × 6 **Clohessy-Wiltshire State Transition Matrix** **Φ**(Δ*t*).

Defining the 2 × 3 B-plane sensitivity matrix **M**<sub>sens</sub> = **M**<sub>rot</sub> **R**<sub>LVLH</sub><sup>ECI</sup> **Φ**<sub>rv</sub> and its 2 × 2 Gramian **Φ**<sub>12</sub> = **M**<sub>sens</sub> **M**<sub>sens</sub><sup>T</sup> (units: s<sup>2</sup>), minimizing ‖Δ**v**‖<sup>2</sup> subject to **b**<sub>p</sub>'<sup>T</sup> **C**<sub>B</sub><sup>−1</sup> **b**<sub>p</sub>' = SMD<sub>bar</sub> yields a 4th-order polynomial in *x* = −λ:

```text
c_4 · x^4 + c_3 · x^3 + c_2 · x^2 + c_1 · x + c_0 = 0
```

where **Q** = **C**<sub>B</sub><sup>−1</sup>, **F** = 2 **Φ**<sub>12</sub> **Q**, *T* = tr(**F**), *D* = det(**F**), and **F̃** = *D* **F**<sup>−1</sup>. Real roots yielding minimum ‖**λ**<sub>0</sub>‖ / 2 give the fuel-optimal avoidance impulse.

---

## 4. Verification & Empirical Performance

### 4.1 Held-Out Test Set Accuracy
Evaluated on unseen conjunction scenarios (*P*<sub>c</sub> ∈ [10<sup>−20</sup>, 1]) against exact 2D Foster integration:
* **Log-MAE (log<sub>10</sub> *P*<sub>c</sub>):** `0.1811`
* **Log-Median Error (log<sub>10</sub> *P*<sub>c</sub>):** `0.0623`
* **Median Relative Error:** `14.23%` across 20 orders of magnitude

### 4.2 Formal Symbolic & Dimensional Proof (`extra/verify.py`)
The repository includes an automated SymPy verification suite (`verify.py`) proving:
1. **Basis Orthonormality & Right-Handedness:** **M**<sub>rot</sub> **M**<sub>rot</sub><sup>T</sup> = **I**<sub>2</sub>, **M**<sub>rot</sub> **t̂** = **0**, and det([**t̂**, **n̂**, **r̂**]<sup>T</sup>) = +1.
2. **Algebraic Equivalence:** Exact symbolic equivalence (*c*<sub>k</sub><sup>sym</sup> − *c*<sub>k</sub><sup>code</sup> ≡ 0 for *k* = 0 … 4) between the Lagrangian stationarity condition and the implemented quartic solver.
3. **Dimensional Homogeneity:** Every term *c*<sub>k</sub> *x*<sup>k</sup> is strictly dimensionless (*L*<sup>0</sup> *T*<sup>0</sup> = 1), and the recovered maneuver vector Δ**v** carries exact velocity units (*L* *T*<sup>−1</sup>).

---

## 5. Installation & Usage

### Python Environment
```bash
pip install numpy scipy pandas torch onnx onnxruntime sgp4 requests sympy
```

### 1. Run Symbolic & Dimensional Verification
```bash
python extra/verify.py
```

### 2. Run Live All-vs-All Catalog Screening
Fetches active catalog TLEs (with local caching), propagates states over a 24-hour horizon in 5-minute steps, filters pairs within 50 km via `cKDTree` (excluding co-located/formation pairs with *v*<sub>rel</sub> < 50 m/s), predicts *P*<sub>c</sub> via ONNX, and computes EOP CAM candidates:
```bash
python all_vs_all_screening.py
```

### 3. Generate CCSDS CDM Datasets & Retrain
```bash
python extra/cdm_gen.py --out-dir cdms_gen --count 1000 --seed 42
python extra/train_pino.py
python extra/benchmark_pino.py
```

### 4. Build C++ ONNX Inference Engine
```bash
cd extra
cmake --preset x64-release
cmake --build --preset x64-release
```

---

## 6. Current Limitations & Future Work
* **Covariance Propagation:** The live screening script currently applies a static diagonal LVLH covariance matrix (`Base_Covariance`). Future releases will couple time-varying state transition uncertainty propagation (accounting for atmospheric drag and solar radiation pressure growth).
* **Formation-Flight Filtering:** Co-located satellites and tight formation flyers (*v*<sub>rel</sub> < 0.05 km/s) are filtered out prior to B-plane projection because standard TLE position uncertainties exceed their nominal separation.
* **Eccentric Encounter Geometry:** The synthetic CDM generator constructs circular equal-altitude encounters to guarantee analytical linear TCA orthogonality (**r**<sub>rel</sub> · **v**<sub>rel</sub> = 0); extending training coverage to high-eccentricity GTO/HEO regimes is planned for v2.

## 7. Disclaimer
**This code was made with assistance from LLM's.**