#!/usr/bin/env python3
"""
primary_models.py
Phase 3: Primary Hypothesis Rebuilding
- Pre-specified primary hypotheses: 3 biomarkers (log_CRP, log_GGT, UA_z) x 4 clinical outcomes
- Continuous biomarkers (per SD) + Restricted Cubic Splines (RCS) testing for non-linearity
- Modified Poisson regression with robust SE (sandwich estimator) for common outcomes (Depression 41%, CVD 29%) -> Prevalence Ratios (PR)
- Firth penalized logistic regression for sparse outcomes (Judicial history n=29, Suicide attempt n=48) -> adjusted Odds Ratios (aOR)
- Nested adjustment models (Models 1 through 5) derived from DAG
- Benjamini-Hochberg FDR correction across primary and exploratory test families
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import patsy
from scipy.stats import norm, chi2

# Paths
# Dynamic repository root and directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
DATA_DIR = os.path.join(REPO_ROOT, "dataset")
OUT_DIR = os.path.join(REPO_ROOT, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

df_p = pd.read_csv(os.path.join(OUT_DIR, 'cleaned_patients.csv'))

# Standardize UA by sex for continuous modeling
m_mask = df_p['Sex'] == 'Male'
f_mask = df_p['Sex'] == 'Female'
df_p['UA_sex_z'] = 0.0
df_p.loc[m_mask, 'UA_sex_z'] = (df_p.loc[m_mask, 'Uric_Acid'] - df_p.loc[m_mask, 'Uric_Acid'].mean()) / df_p.loc[m_mask, 'Uric_Acid'].std()
df_p.loc[f_mask, 'UA_sex_z'] = (df_p.loc[f_mask, 'Uric_Acid'] - df_p.loc[f_mask, 'Uric_Acid'].mean()) / df_p.loc[f_mask, 'Uric_Acid'].std()

# Define Primary Biomarkers (per 1 SD increase)
BIOMARKERS = {
    'log_CRP_z': 'CRP (log-transformed, per SD)',
    'log_GGT_z': 'GGT (log-transformed, per SD)',
    'UA_sex_z': 'Uric Acid (sex-standardized, per SD)'
}

# Define Primary Outcomes
OUTCOMES = {
    'Depressive_Disorder_bin': {'name': 'Depressive Disorder', 'type': 'common', 'prev': df_p['Depressive_Disorder_bin'].mean()},
    'Cardiovascular_Diseases_bin': {'name': 'Cardiovascular Disease', 'type': 'common', 'prev': df_p['Cardiovascular_Diseases_bin'].mean()},
    'Suicide_Attempt_bin': {'name': 'Suicide Attempt', 'type': 'sparse', 'prev': df_p['Suicide_Attempt_bin'].mean()},
    'Judicial_History_bin': {'name': 'Judicial History / Criminal Justice', 'type': 'sparse', 'prev': df_p['Judicial_History_bin'].mean()}
}

# Define Nested Adjustment Models (DAG-derived)
MODELS = {
    'Unadjusted': '',
    'Model_1': 'Age + Is_Male',
    'Model_2': 'Age + Is_Male + Smoker + Alcohol + Drugs',
    'Model_3': 'Age + Is_Male + Smoker + Alcohol + Drugs + BMI',
    'Model_4': 'Age + Is_Male + Smoker + Alcohol + Drugs + BMI + Med_Olanzapine + Med_Enzyme_Inducer + Med_Coaprovel_Diuretic',
    'Model_5': 'Age + Is_Male + Smoker + Alcohol + Drugs + BMI + Med_Olanzapine + Med_Enzyme_Inducer + Med_Coaprovel_Diuretic + Duration_of_Illness_Years + eGFR'
}

# Implementation of Firth Penalized Logistic Regression
def fit_firth_logit(y, X, max_iter=100, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(max_iter):
        pi = 1.0 / (1.0 + np.exp(-X @ beta))
        pi = np.clip(pi, 1e-15, 1.0 - 1e-15)
        W = pi * (1.0 - pi)
        XtW = X.T * W
        info = XtW @ X
        info_inv = np.linalg.pinv(info)
        H_diag = np.sum((X @ info_inv) * X, axis=1) * W
        score = X.T @ (y - pi + H_diag * (0.5 - pi))
        step = info_inv @ score
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    pi = 1.0 / (1.0 + np.exp(-X @ beta))
    W = pi * (1.0 - pi)
    info = (X.T * W) @ X
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.diag(cov))
    return beta, se, cov

# 1. RUN PRIMARY NESTED MODELS
results_primary = []

for out_var, out_meta in OUTCOMES.items():
    y = df_p[out_var].values
    for bio_var, bio_name in BIOMARKERS.items():
        for mod_name, cov_str in MODELS.items():
            if cov_str == '':
                formula = f'{out_var} ~ {bio_var}'
            else:
                formula = f'{out_var} ~ {bio_var} + {cov_str}'
            
            # Outcome modeling
            if out_meta['type'] == 'common':
                # Modified Poisson regression with robust SE (sandwich) -> Prevalence Ratio (PR)
                try:
                    glm_fam = sm.families.Poisson()
                    mod = smf.glm(formula, data=df_p, family=glm_fam).fit(cov_type='HC0')
                    coef = mod.params[bio_var]
                    se = mod.bse[bio_var]
                    pval = mod.pvalues[bio_var]
                    est = np.exp(coef)
                    ci_low = np.exp(coef - 1.96 * se)
                    ci_high = np.exp(coef + 1.96 * se)
                    metric_name = 'PR'
                except Exception as e:
                    print(f"Error fitting Poisson for {out_var} ~ {bio_var} ({mod_name}): {e}")
                    continue
            else:
                # Sparse outcome: Firth Penalized Logistic Regression -> aOR
                try:
                    dmatrix_X = patsy.dmatrix(formula.split('~')[1], data=df_p, return_type='dataframe')
                    X_mat = dmatrix_X.values
                    var_names = list(dmatrix_X.columns)
                    b, se_arr, _ = fit_firth_logit(y, X_mat)
                    target_idx = var_names.index(bio_var)
                    coef = b[target_idx]
                    se = se_arr[target_idx]
                    z_stat = coef / se
                    pval = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))
                    est = np.exp(coef)
                    ci_low = np.exp(coef - 1.96 * se)
                    ci_high = np.exp(coef + 1.96 * se)
                    metric_name = 'aOR'
                except Exception as e:
                    print(f"Error fitting Firth logit for {out_var} ~ {bio_var} ({mod_name}): {e}")
                    continue

            results_primary.append({
                'Outcome': out_meta['name'],
                'Outcome_Var': out_var,
                'Outcome_Type': out_meta['type'],
                'Biomarker': bio_name,
                'Biomarker_Var': bio_var,
                'Model': mod_name,
                'Effect_Metric': metric_name,
                'Estimate': est,
                'CI_Low': ci_low,
                'CI_High': ci_high,
                'Beta': coef,
                'SE': se,
                'P_value': pval
            })

df_primary_res = pd.DataFrame(results_primary)

# Multiple Testing Correction: Benjamini-Hochberg FDR
# We correct across the pre-specified primary hypothesis family for Model 5 (Full clinical model: 12 tests)
m5_mask = df_primary_res['Model'] == 'Model_5'
df_m5 = df_primary_res[m5_mask].copy()
rej, fdr_p, _, _ = multipletests(df_m5['P_value'], alpha=0.05, method='fdr_bh')
df_m5['FDR_P_value'] = fdr_p
df_m5['FDR_Significant'] = rej

# Map FDR p-values back
df_primary_res['FDR_P_value'] = np.nan
df_primary_res['FDR_Significant'] = False
df_primary_res.loc[m5_mask, 'FDR_P_value'] = fdr_p
df_primary_res.loc[m5_mask, 'FDR_Significant'] = rej

# Also correct unadjusted model for comparison
m0_mask = df_primary_res['Model'] == 'Unadjusted'
rej0, fdr_p0, _, _ = multipletests(df_primary_res.loc[m0_mask, 'P_value'], alpha=0.05, method='fdr_bh')
df_primary_res.loc[m0_mask, 'FDR_P_value'] = fdr_p0
df_primary_res.loc[m0_mask, 'FDR_Significant'] = rej0

# Save primary results
df_primary_res.to_csv(os.path.join(OUT_DIR, 'primary_nested_models.csv'), index=False)
print("Primary nested models fitted and exported successfully.")

# 2. RESTRICTED CUBIC SPLINES (RCS) TESTING FOR NON-LINEARITY
# In particular, test if Uric Acid has a J-shaped or U-shaped curve with outcomes
spline_results = []
print("\n=== RESTRICTED CUBIC SPLINES NON-LINEARITY AUDIT ===")

for out_var, out_meta in OUTCOMES.items():
    for bio_var, bio_name in BIOMARKERS.items():
        # Fit linear model (Model 3 base)
        lin_formula = f'{out_var} ~ {bio_var} + Age + Is_Male + Smoker + Alcohol + Drugs + BMI'
        # Fit natural spline with 3 degrees of freedom
        rcs_formula = f'{out_var} ~ cr({bio_var}, df=3) + Age + Is_Male + Smoker + Alcohol + Drugs + BMI'
        
        try:
            if out_meta['type'] == 'common':
                m_lin = smf.glm(lin_formula, data=df_p, family=sm.families.Poisson()).fit(cov_type='HC0')
                m_rcs = smf.glm(rcs_formula, data=df_p, family=sm.families.Poisson()).fit(cov_type='HC0')
            else:
                m_lin = smf.logit(lin_formula, data=df_p).fit(disp=0)
                m_rcs = smf.logit(rcs_formula, data=df_p).fit(disp=0)

            # Likelihood ratio / Wald test for non-linearity (df = 2)
            lr_stat = 2.0 * (m_rcs.llf - m_lin.llf)
            lr_stat = max(0.0, lr_stat)
            p_nonlin = 1.0 - chi2.cdf(lr_stat, df=2)
            
            spline_results.append({
                'Outcome': out_meta['name'],
                'Biomarker': bio_name,
                'Linear_AIC': m_lin.aic,
                'Spline_AIC': m_rcs.aic,
                'Delta_AIC': m_rcs.aic - m_lin.aic,
                'LR_Stat': lr_stat,
                'P_Nonlinear': p_nonlin,
                'Nonlinear_Shape': 'J-shaped / U-shaped' if (p_nonlin < 0.1 and m_rcs.aic < m_lin.aic) else 'Monotonic / Linear'
            })
            print(f"{bio_name} -> {out_meta['name']}: Linear AIC={m_lin.aic:.2f}, Spline AIC={m_rcs.aic:.2f}, LR p={p_nonlin:.4f}")
        except Exception as e:
            print(f"Error fitting spline for {out_var} and {bio_var}: {e}")

df_spline = pd.DataFrame(spline_results)
df_spline.to_csv(os.path.join(OUT_DIR, 'spline_nonlinearity_tests.csv'), index=False)

# Display summary of primary findings
print("\n=== SUMMARY OF FULLY ADJUSTED (MODEL 5) PRIMARY FINDINGS ===")
m5_display = df_primary_res[df_primary_res['Model'] == 'Model_5'][['Outcome', 'Biomarker', 'Effect_Metric', 'Estimate', 'CI_Low', 'CI_High', 'P_value', 'FDR_P_value', 'FDR_Significant']]
print(m5_display.to_string(index=False))

