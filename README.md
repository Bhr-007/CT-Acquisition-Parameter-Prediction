# Project 8: AI-Based CT Acquisition Parameter Prediction

## 1. Project Overview & Scientific Objectives
Setting up X-ray Computed Tomography (CT) acquisition parameters—tube voltage (kV), exposure per projection (mA*s), number of projection views (N_proj), and spectral filtration thickness (d_Cu)—for a new sample material and geometry traditionally relies on physical trial-and-error, trading off image quality (Signal-to-Noise Ratio, SNR, and Contrast-to-Noise Ratio, CNR) against total scan time and X-ray dose.

This project implements a complete physics-simulation-driven machine learning and inverse-optimisation pipeline:
1. **Polychromatic Forward Physics Simulator (`Code/ct_simulator.py`):** Models continuous Kramers bremsstrahlung and Tungsten K-shell characteristic radiation (12–220 keV), intrinsic Aluminium (1.5 mm) and variable Copper (0.0–3.0 mm) filtration, authoritative **NIST XCOM** energy-dependent mass attenuation coefficients (mu/rho)(E) for **Polymer (PMMA)**, **Aluminium**, and **Steel**, polychromatic Beer-Lambert path-length integration, **Poisson photon-counting noise**, and GPU-accelerated **3D Cone-Beam FDK (`3D_FDK_CUDA`)** central-slice reconstruction ($128 \times 128$) via the **ASTRA Toolbox**.
2. **Balanced Latin-Hypercube Dataset & Extrapolation Probe (`Data/`):** Generates **3,150 in-distribution configurations** (retaining **2,940 physically valid** configurations split 70/15/15 into Training/Validation/Testing and explicitly logging **210 photon-starved** configurations) plus a dedicated **450-configuration out-of-distribution extrapolation dataset** (**397 valid**, **53 photon-starved**) across unseen sample thicknesses, low-kV/heavy-filtration, and extreme projection regimes.
3. **Multi-Model Surrogate Regression (`Models/trained_surrogates_bundle.pkl`):** Compares **XGBoost**, **Random Forest**, **Small MLP ($64 \rightarrow 64 \rightarrow 32$)**, and **Gaussian Process Regression** trained with 5-fold cross-validation and zero data leakage.
4. **Constrained Inverse Parameter Recommendation & Pareto Frontier Analysis:** Wraps the trained surrogates and a **Photon-Starvation Feasibility Classifier** inside constrained **Grid Search** and **Optuna Bayesian Optimisation** to recommend minimum-scan-time settings that achieve a requested target CNR within a scan-time budget, validated by re-running the full ASTRA 3D-FDK physics simulator.

---

## 2. Complete Directory Structure
* **Trained Model:** [Download trained_surrogates_bundle.pkl (185 MB)](https://drive.google.com/drive/folders/1NGQMSgaMFfxDRuMsaroEPRY8yyH68td3?usp=drive_link)
- `cws_project/`
  - `Code/ct_simulator.py` (Core polychromatic forward simulator module)
  - `Code/Demo_Worked_Recommendation.ipynb` (Standalone worked inverse-recommendation demo notebook)
  - `Code/main_pipeline.ipynb` (Master notebook executing Stages 0–6)
  - `Data/Reference/nist_xcom_attenuation.csv` (NIST XCOM attenuation tables, 10–250 keV)
  - `Data/Pilot/pilot_30_summary.csv` (30-run timed pilot verification dataset)
  - `Data/Raw_Sweep/` (Complete 3,600-run raw & master CSVs including starvation logs)
  - `Data/Training/train_data.csv` (70% stratified training split, 2,058 valid rows)
  - `Data/Validation/val_data.csv` (15% validation split, 441 valid rows)
  - `Data/Testing/test_data.csv` (15% in-distribution test split, 441 valid rows)
  - `Data/Extrapolation/extrapolation_test_data.csv` (Out-of-distribution test set, 397 valid rows)
  - `Data/Slices/` (54 representative $128 \times 128$ float32 `.npy` reconstructed slices)
  - `Models/trained_surrogates_bundle.pkl`[From the downloaded folder via link] (Fitted scalers, XGB, RF, MLP, GP & Starvation Classifier)
  - `Figures/` (8 publication-ready 300-DPI PNG figures, `fig01` through `fig08`)
  - `Docs/` (Evaluation metrics, sensitivity, inverse validation, and Pareto CSV tables)
  - `README.md` (Master documentation file)

---

## 3. Summary of Key Quantitative Results
- **Surrogate Prediction Accuracy (Held-Out In-Distribution Test Set, $n = 441$):**
  - **Gaussian Process:** SNR $R^2 = 0.9833$ (MAE = 0.94), CNR $R^2 = 0.9854$ (MAE = 1.04)
  - **MLP ($64 \rightarrow 64 \rightarrow 32$):** SNR $R^2 = 0.9757$ (MAE = 1.35), CNR $R^2 = 0.9775$ (MAE = 1.35)
  - **XGBoost (Primary):** SNR $R^2 = 0.9734$ (MAE = 1.30), CNR $R^2 = 0.9647$ (MAE = 1.59)
  - **Random Forest:** SNR $R^2 = 0.8959$ (MAE = 2.81), CNR $R^2 = 0.8866$ (MAE = 3.08)
- **Extrapolation Risk Probe ($n = 397$):**
  - Overall XGBoost extrapolation achieves SNR $R^2 = 0.8862$ and CNR $R^2 = 0.9071$, dropping to CNR $R^2 = 0.5493$ in the unseen low-kV / heavy-filtration regime (`E2_LOW_KV_HEAVY_FILTER`).
- **Inverse Recommendation Validation (Re-Simulated in ASTRA 3D-FDK across 6 Tasks):**
  - **Surrogate Optuna BO:** Mean $|\text{Achieved CNR} - \text{Target CNR}|$ error = **2.94%**, Mean scan time = **9.27 s**
  - **Surrogate Grid Search:** Mean $|\text{Achieved CNR} - \text{Target CNR}|$ error = **4.82%**, Mean scan time = **7.87 s**
  - **Rule-of-Thumb Heuristic Baseline:** Mean error = **16.99%**, Mean scan time = **21.88 s**
- **Pareto Frontier Case Studies (Thin Aluminium $8\text{ mm}$ vs. Thick Steel $15\text{ mm}$):**
  - Re-simulated surrogate Pareto frontiers match brute-force reference frontiers within **4.40%** mean CNR error for Thin Aluminium ($	ext{HV}_{\text{resim}} = 3088$ vs. $	ext{HV}_{\text{BF}} = 3108$) and **3.62%** for Thick Steel ($	ext{HV}_{\text{resim}} = 670$ vs. $	ext{HV}_{\text{BF}} = 669$).

---

## 4. Reproduction Instructions
1. Activate the Conda environment: `conda activate ct-project` (Python 3.10, `astra-toolbox` with CUDA, `xgboost`, `optuna`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`).
2. Run `Code/main_pipeline.ipynb` top-to-bottom to execute Stages 0–6 (~3 minutes total GPU runtime).
3. Open `Code/Demo_Worked_Recommendation.ipynb` to run the standalone worked example.
