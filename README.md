# Immuno-Metabolic Stratification in Schizophrenia: Analysis & Reproduction Pipeline

This repository contains the complete statistical analysis code, data preprocessing pipeline, and anonymized datasets to reproduce all empirical findings, models, and figures for the study:

> **"Immuno-Metabolic Stratification in Schizophrenia: Associations of C-Reactive Protein, Gamma-Glutamyl Transferase, and Uric Acid with Psychiatric and Somatic Outcomes in an Algerian Clinical Cohort"**  
> *Target Journal: Metabolic Brain Disease (Springer Nature)*  
> **Corresponding Authors:** Dr. Atika Sadoki (`sadatika0000@gmail.com`) & Dr. Abdelbasset Boukelia (`a.boukelia@crbt.dz`)

---

## Repository Structure

```text
.
├── dataset/
│   ├── patients_english.csv          # Anonymized clinical cohort data (N=200)
│   └── control_group_english.csv     # Anonymized healthy blood donors (N=1,000)
├── scripts/
│   ├── 01_audit_and_preprocess.py    # Data auditing, eGFR calculation, standardization
│   ├── 02_case_control_descriptive.py# Descriptive stats & Standardized Mean Differences (SMD)
│   ├── 03_primary_models.py          # Modified Poisson & Firth penalized logistic models
│   ├── 04_synergy_and_interaction.py # Additive interaction (RERI, AP, S) & burden scores
│   ├── 05_clustering_and_phenotyping.py # Gaussian Mixture Modeling & cluster profiling
│   ├── 06_sensitivity_and_discrimination.py # Sensitivity audits, E-values, ROC & discrimination
│   └── 07_generate_figures_and_tables.py    # Publication figure and table generation
├── output/                           # Precomputed output tables and intermediate datasets
│   ├── figures/                      # Rendered publication figures (PDF and PNG)
│   └── tables/                       # Formatted LaTeX and CSV tables
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment definition
├── run_reproduction.sh               # Master one-step execution script
└── LICENSE                           # MIT Open Source License
```

---

## Quick Start & Installation

### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/BoukeliaAbdelbasset/Schizophrenia-ImmunoMetabolic.git
cd Schizophrenia-ImmunoMetabolic

# Create and activate environment
conda env create -f environment.yml
conda activate schizophrenia-immunometabolic

# Run the complete pipeline
bash run_reproduction.sh
```

### Option B: Using Standard Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_reproduction.sh
```

---

## Workflow Overview

1. **Phase 1 & 2: Audit and Preprocessing (`01_audit_and_preprocess.py`)**  
   Audits sample counts, cleans missing values, computes eGFR via the CKD-EPI 2021 equation, log-transforms skewed biomarker distributions (CRP, GGT), standardizes uric acid by sex, and outputs cleaned cohorts.

2. **Phase 3: Case-Control Comparisons (`02_case_control_descriptive.py`)**  
   Calculates Standardized Mean Differences (Cohen's d) with bootstrap 95% confidence intervals, Mann-Whitney U tests, and chi-squared tests across all demographic and biochemical markers.

3. **Phase 3: Primary Regression Models (`03_primary_models.py`)**  
   Fits modified Poisson regression models with robust sandwich standard errors for prevalent conditions (Depression, CVD) to compute Prevalence Ratios (PR), and Firth penalized logistic regression for rare events (Suicide attempts, Judicial history), across 5 nested adjustment tiers derived from causal DAGs with Benjamini-Hochberg FDR correction.

4. **Phase 3: Synergy and Interaction (`04_synergy_and_interaction.py`)**  
   Evaluates additive biological synergy (Relative Excess Risk due to Interaction [RERI], Attributable Proportion [AP], Synergy Index [S]) with delta-method confidence intervals, multiplicative interaction terms, and cumulative multi-marker burden trend tests.

5. **Phase 3: Clustering and Phenotyping (`05_clustering_and_phenotyping.py`)**  
   Applies Gaussian Mixture Modeling (GMM) with BIC/AIC model selection to identify biological clusters across continuous biomarker profiles and assesses cluster stability via 1,000 bootstrap resamples.

6. **Phase 3: Sensitivity & Discrimination (`06_sensitivity_and_discrimination.py`)**  
   Executes clinical sensitivity restrictions (excluding acute inflammation, high metabolic liability antipsychotics, liver disorders, and adjusting for eGFR/diuretics), computes VanderWeele E-values for unmeasured confounding, and calculates incremental discrimination metrics (ROC AUC, DeLong tests, continuous NRI, and IDI).

7. **Figure and Table Generation (`07_generate_figures_and_tables.py`)**  
   Renders publication-ready vector PDFs and 300-DPI PNGs into `output/figures/` and formatted tables into `output/tables/`.

---

## Ethical Standards & Anonymization

The study protocol received formal ethical approval from the Research Ethics Committee in Life and Health Sciences (CER-SVS Protocol 2022-04) at the Higher Institute of Biotechnology of Monastir, Tunisia, and institutional authorizations from CHU Frantz-Fanon (Blida, Algeria) and Mohamed Boudiaf Hospital (Médéa, Algeria). All shared datasets have been strictly de-identified in compliance with patient data privacy standards and the Declaration of Helsinki.

---

## Citation & Contact

If you use this code or data in your research, please cite:

```bibtex
@article{sadoki2026immunometabolic,
  title={Immuno-Metabolic Stratification in Schizophrenia: Associations of C-Reactive Protein, Gamma-Glutamyl Transferase, and Uric Acid with Psychiatric and Somatic Outcomes in an Algerian Clinical Cohort},
  author={Sadoki, Atika and Innoubli, Oumaima and Boukelia, Abdelbasset and Benkortbi, Ahlem and Bessedik, Khadija and Sellam, Feriel and Bel Hadj Jrad, Besma},
  journal={Metabolic Brain Disease},
  year={2026}
}
```

For questions regarding the dataset or analysis, please contact:
- **Dr. Atika Sadoki**: `sadatika0000@gmail.com`
- **Dr. Abdelbasset Boukelia**: `a.boukelia@crbt.dz`
