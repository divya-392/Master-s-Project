# Master-s-Project
# Intelligent Intrusion Detection System (IDS) with MBGWO Feature Selection

This repository contains the artefacts for a Master’s project that builds and evaluates an Intrusion Detection System (IDS) using **CICIDS2017** dataset. 
We compare tree-based classifiers trained on **all features** versus models trained on features selected via **Modified Binary Grey Wolf Optimisation (MBGWO)**. A future-work notebook outlines a per-class ensemble idea (LCCDE).

The pipeline integrates **feature engineering** (IG/FCBF/KPCA), **Modified Binary Grey Wolf Optimisation (MBGWO)** for feature selection, and **tree-based classifiers** (DT/RF/ET/XGBoost), with **hyperparameter tuning** and an optional **stacking ensemble**.  
All experiments are reproducible via numbered Jupyter notebooks.

**Thesis (PDF):** `Thesis/Your_Thesis.pdf`  
**Screencast (5–10 min):** `media/screencast.mp4`

---

## 1. Objectives
- Develop a reproducible IDS pipeline for tabular network traffic.  
- Quantify the effect of **explicit feature selection (MBGWO)** on model quality and efficiency.  
- Provide clear artefacts for assessment: notebooks, metrics tables, thesis PDF, and a short demo video.

---

## 2. Methods (Brief)
**Models.** Decision Tree, Random Forest, Extra Trees, XGBoost.  
**Feature Selection.** Modified Binary Grey Wolf Optimisation (MBGWO) to search a binary mask of features.  
**Fitness (multi-objective).**  
\[
\text{Fitness} = \alpha \cdot \text{CV-Accuracy} + (1-\alpha)\cdot\left(1 - \frac{\#\text{selected}}{\#\text{total}}\right)
\]
This rewards **predictive performance** and **sparsity** simultaneously (α ∈ [0,1]).  
**Evaluation.** F1-weighted, Precision, Recall, Accuracy, and weighted OVR AUC; confusion matrices and per-class reports.

> Rationale: removing redundant features often benefits bagging-style trees (RF/ET/DT). Boosting (XGBoost) already performs implicit selection, so gains may be smaller.

---

## 3. Repository Structure (run in this order)
```
├─data
     ├─ CICIDS2017_sample.csv                       # Small balanced subset for quick runs
     ├─ CICIDS2017_sample_km.csv                    # Generated Cleaned dataset from 03_MTH_IDS.ipynb file.
     ├─ CICIDS2017_sample_km_portscan.csv           # Generated Portscan dataset from 03_MTH_IDS.ipynb file.
     └─ CICIDS2017_sample_km_without_portscan.csv   # Generated Without Portscan dataset from 03_MTH_IDS.ipynb file.
├─ 01_Data_Sampling.ipynb                          #  Dataset preprocessing, balancing, and sampled CSV creation.
├─ 02_IDS_TreeBase.ipynb                           # Baselines: DT, RF, ET, XGB (all features)
├─ 03_MTH_IDS.ipynb                                # End-to-end narrative and MBGWO framing
├─ 04_IDS_MBGWO_Implementation.ipynb               # All-features vs MBGWO-selected (final comparisons)
├─ 05_LCCDE_IDS_FutureWork.ipynb                   # Future per-class ensemble (LCCDE) concept
├─ Data_Combining.ipynb                            # Combining the .csv files
├─ FCBF_module.py                                  # Optional Module used for correlation-based filter experiments
├─ requirements.txt                                # Pinned Python dependencies
├─ Thesis/G00473080_Thesis.pdf                          # Master’s thesis (PDF)
└─ media/screencast.mp4                            # 5–10 minute demo video
```
> If you include the full CICIDS2017 dataset, place it under `data/` and adjust paths in notebooks as needed.

---

## 4. Data
- **Primary dataset:** CICIDS2017 (Canadian Institute for Cybersecurity). The CICIDS2017 dataset is publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html
- After downloading the dataset files from the above link, combine the per-day CSVs into a single CSV (use your own script or provide `data/Data_combining.ipynb` in the repo if you want this to be reproducible).  
- **Included for convenience:** `CICIDS2017_sample.csv` — a small, balanced subset to reproduce results quickly.  
- **Sanitisation:** numeric coercion, ±Inf→NaN handling (common in rate features), clipping to float‑safe ranges, and median imputation.

---

## 5. Environment & Installation
- **Python:** 3.10–3.11 (recommended)  
- **OS:** Windows / macOS / Linux  
- **Jupyter:** Notebook, Lab, or VS Code

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

Note: using a virtual environment is optional. In my execution, I did not use a virtual environment and used Python 3.11.

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins core packages (pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, scipy, matplotlib, seaborn, joblib, tqdm, jupyter, imbalanced-learn).

---

## 6. How to Run

### 1) Launch Jupyter
```bash
jupyter notebook        # or: jupyter lab
```
### 2) Execute notebooks in order
1. **`01_Data_Sampling.ipynb`** — build `CICIDS2017_sample.csv`. *Already provided for quick runs.*  
2. **`02_IDS_TreeBase.ipynb`** — trains DT/RF/ET/XGB(models) on all features; prints baseline metrics.  
3. **`03_MTH_IDS.ipynb`** — narrative notebook linking the pipeline decisions and summarising the approach/results of MBGWO.
4. **`04_IDS_MBGWO_Implementation.ipynb`** — train after MBGWO selection; print the **Final Comparison Table** (F1-weighted, Precision, Recall, Accuracy, AUC; also report number of selected features).  
5. **`05_LCCDE_IDS_FutureWork.ipynb`** — demonstrated **Leader Class & Confidence Decision Ensemble (LCCDE)** as a future direction.  

---

## 7. Results (Summary)
- On the included sample: **RF/ET/DT** typically achieve **higher F1-weighted** after MBGWO (fewer, more informative features).  
- **XGBoost** often remains strongest on all features; MBGWO remains competitive.  
- Weighted OVR **AUC** is generally high (≈0.99+ on the provided split).  
- Exact values vary with the train/test split and search settings (population, iterations, CV).

> Final tables and plots are produced in `04_IDS_MBGWO_Implementation.ipynb` and `03_MTH_IDS.ipynb`.

---

## 8. Screencast (5–10 minutes)
Placed a short video at `media/screencast.mp4` showing: environment setup, running `02_` (baselines), running `04_` (MBGWO comparison), and key takeaways.
Download Screencast Video - https://github.com/divya-392/Master-s-Project/blob/main/media/screencast.mp4)

---

## 9. Key Implementation Details
- **FCBF:** Fast correlation-based filter to remove obviously redundant features (when enabled).
- **MBGWO feature selection:** Modified Binary Grey Wolf Optimisation used to search a binary feature mask via an accuracy–sparsity objective.
- **Hyperparameter tuning**: `hyperopt` (TPE) and `scikit-optimize` (Bayesian GP) used to optimize tree-based models and ensemble meta-learner.
---

## 10. Limitations & Next Steps
- **Minority classes:** small supports cause variance in per-class F1 → consider class-weights, threshold calibration, targeted oversampling.  
- **Runtime:** metaheuristic searches are compute-intensive → tighten search budgets or use hybrid pre-filters.  
- **External validity:** cross-dataset validation is planned as future work.

---

## 11. Troubleshooting
- **“Input X contains infinity or a value too large”** — ensure sanitisation/imputation cells run before training.  
- **Slow execution** — reduce population/iterations and CV folds; start with RF + XGB; use the provided sample CSV.  
- **Package conflicts** — use a fresh virtual environment and `requirements.txt` pins.
- If imports fail, ensure the correct **Python 3.11** environment is active (`python --version`).  
- For large notebooks, restart kernel and **Run All** after installing dependencies.  
- If dataset paths differ, update the `data/` path at the top of each notebook.

---

## 12. Acknowledgements
- CICIDS2017 — Canadian Institute for Cybersecurity.  
- Grey Wolf Optimiser literature — foundational background for MBGWO.

---

## 13. Licence
This repository is provided for academic evaluation of a Master’s project.
