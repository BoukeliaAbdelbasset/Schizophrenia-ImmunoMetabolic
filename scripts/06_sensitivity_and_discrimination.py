#!/usr/bin/env python3
"""
sensitivity_and_discrimination.py
Phase 3: Robustness Audits & Incremental Prognostic/Diagnostic Discrimination
- Sensitivity 1: Exclude acute inflammation (CRP > 10 mg/L)
- Sensitivity 2: Exclude high metabolic liability antipsychotics (Olanzapine)
- Sensitivity 3: Exclude heavy alcohol, hepatic disease, enzyme inducers for GGT
- Sensitivity 4: Adjust for eGFR, exclude CKD (<60 mL/min/1.73m2), adjust for diuretic for UA
- Sensitivity 5: VanderWeele E-values for unmeasured confounding
- Incremental Value: ROC AUC, DeLong p-value, continuous NRI, and IDI (1,000 bootstrap resamples)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss

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

# 1. SENSITIVITY ANALYSES
print("=== RUNNING SENSITIVITY ANALYSES ===")
sens_results = []

# Base formula: Model 4 covariates
base_covs = 'Age + Is_Male + Smoker + BMI'

# Sensitivity 1: Exclude acute inflammation (CRP > 10 mg/L, N=134 remaining)
sub_crp10 = df_p[df_p['CRP'] <= 10.0].copy()
# Re-scale Z-score in subset
sub_crp10['log_CRP_z_sub'] = (sub_crp10['log_CRP'] - sub_crp10['log_CRP'].mean()) / sub_crp10['log_CRP'].std()

for out_var, out_name, fam in [
    ('Depressive_Disorder_bin', 'Depressive Disorder', 'poisson'),
    ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease', 'poisson'),
    ('Suicide_Attempt_bin', 'Suicide Attempt', 'logit')
]:
    formula = f"{out_var} ~ log_CRP_z_sub + {base_covs}"
    try:
        if fam == 'poisson':
            m = smf.glm(formula, data=sub_crp10, family=sm.families.Poisson()).fit(cov_type='HC0')
        else:
            m = smf.logit(formula, data=sub_crp10).fit(disp=0)
        coef = m.params['log_CRP_z_sub']
        se = m.bse['log_CRP_z_sub']
        pval = m.pvalues['log_CRP_z_sub']
        sens_results.append({
            'Analysis': 'Sens 1: Exclude CRP > 10 mg/L (N=134)',
            'Outcome': out_name,
            'Biomarker': 'CRP (log, per SD)',
            'Estimate': np.exp(coef),
            'CI_Low': np.exp(coef - 1.96 * se),
            'CI_High': np.exp(coef + 1.96 * se),
            'P_value': pval
        })
    except Exception as e:
        print(f"Error Sens 1 for {out_name}: {e}")

# Sensitivity 2: Exclude Olanzapine users (N=170 remaining)
sub_no_olan = df_p[df_p['Med_Olanzapine'] == 0].copy()
sub_no_olan['log_CRP_z_sub'] = (sub_no_olan['log_CRP'] - sub_no_olan['log_CRP'].mean()) / sub_no_olan['log_CRP'].std()
sub_no_olan['log_GGT_z_sub'] = (sub_no_olan['log_GGT'] - sub_no_olan['log_GGT'].mean()) / sub_no_olan['log_GGT'].std()

for bio_var, bio_name in [('log_CRP_z_sub', 'CRP (log, per SD)'), ('log_GGT_z_sub', 'GGT (log, per SD)')]:
    for out_var, out_name, fam in [
        ('Depressive_Disorder_bin', 'Depressive Disorder', 'poisson'),
        ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease', 'poisson'),
        ('Suicide_Attempt_bin', 'Suicide Attempt', 'logit')
    ]:
        formula = f"{out_var} ~ {bio_var} + {base_covs}"
        try:
            if fam == 'poisson':
                m = smf.glm(formula, data=sub_no_olan, family=sm.families.Poisson()).fit(cov_type='HC0')
            else:
                m = smf.logit(formula, data=sub_no_olan).fit(disp=0)
            coef = m.params[bio_var]
            se = m.bse[bio_var]
            pval = m.pvalues[bio_var]
            sens_results.append({
                'Analysis': 'Sens 2: Exclude Olanzapine Users (N=170)',
                'Outcome': out_name,
                'Biomarker': bio_name,
                'Estimate': np.exp(coef),
                'CI_Low': np.exp(coef - 1.96 * se),
                'CI_High': np.exp(coef + 1.96 * se),
                'P_value': pval
            })
        except Exception as e:
            print(f"Error Sens 2 for {bio_name} -> {out_name}: {e}")

# Sensitivity 3: Exclude heavy alcohol, hepatic disease, and enzyme inducers for GGT (N=94 remaining)
sub_clean_ggt = df_p[(df_p['Alcohol'] == 0) & (df_p['Hepatic_Diseases_bin'] == 0) & (df_p['Med_Enzyme_Inducer'] == 0)].copy()
sub_clean_ggt['log_GGT_z_sub'] = (sub_clean_ggt['log_GGT'] - sub_clean_ggt['log_GGT'].mean()) / sub_clean_ggt['log_GGT'].std()

for out_var, out_name, fam in [
    ('Depressive_Disorder_bin', 'Depressive Disorder', 'poisson'),
    ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease', 'poisson'),
    ('Suicide_Attempt_bin', 'Suicide Attempt', 'logit')
]:
    formula = f"{out_var} ~ log_GGT_z_sub + {base_covs}"
    try:
        if fam == 'poisson':
            m = smf.glm(formula, data=sub_clean_ggt, family=sm.families.Poisson()).fit(cov_type='HC0')
        else:
            m = smf.logit(formula, data=sub_clean_ggt).fit(disp=0)
        coef = m.params['log_GGT_z_sub']
        se = m.bse['log_GGT_z_sub']
        pval = m.pvalues['log_GGT_z_sub']
        sens_results.append({
            'Analysis': 'Sens 3: Exclude Alcohol, Hepatic & Inducers (N=94)',
            'Outcome': out_name,
            'Biomarker': 'GGT (log, per SD)',
            'Estimate': np.exp(coef),
            'CI_Low': np.exp(coef - 1.96 * se),
            'CI_High': np.exp(coef + 1.96 * se),
            'P_value': pval
        })
    except Exception as e:
        print(f"Error Sens 3 for {out_name}: {e}")

# Sensitivity 4: Uric acid adjusted for eGFR, excluding CKD (<60) and adjusting for Coaprovel (N=163)
sub_clean_ua = df_p[df_p['CKD_stage3_plus'] == 0].copy()
m_mask = sub_clean_ua['Sex'] == 'Male'
f_mask = sub_clean_ua['Sex'] == 'Female'
sub_clean_ua['UA_sex_z_sub'] = 0.0
sub_clean_ua.loc[m_mask, 'UA_sex_z_sub'] = (sub_clean_ua.loc[m_mask, 'Uric_Acid'] - sub_clean_ua.loc[m_mask, 'Uric_Acid'].mean()) / sub_clean_ua.loc[m_mask, 'Uric_Acid'].std()
sub_clean_ua.loc[f_mask, 'UA_sex_z_sub'] = (sub_clean_ua.loc[f_mask, 'Uric_Acid'] - sub_clean_ua.loc[f_mask, 'Uric_Acid'].mean()) / sub_clean_ua.loc[f_mask, 'Uric_Acid'].std()

for out_var, out_name, fam in [
    ('Judicial_History_bin', 'Judicial History / Criminal Justice', 'logit'),
    ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease', 'poisson')
]:
    formula = f"{out_var} ~ UA_sex_z_sub + {base_covs} + eGFR + Med_Coaprovel_Diuretic"
    try:
        if fam == 'poisson':
            m = smf.glm(formula, data=sub_clean_ua, family=sm.families.Poisson()).fit(cov_type='HC0')
        else:
            m = smf.logit(formula, data=sub_clean_ua).fit(disp=0)
        coef = m.params['UA_sex_z_sub']
        se = m.bse['UA_sex_z_sub']
        pval = m.pvalues['UA_sex_z_sub']
        sens_results.append({
            'Analysis': 'Sens 4: Exclude CKD, Adjust eGFR & Diuretic (N=163)',
            'Outcome': out_name,
            'Biomarker': 'Uric Acid (sex-std, per SD)',
            'Estimate': np.exp(coef),
            'CI_Low': np.exp(coef - 1.96 * se),
            'CI_High': np.exp(coef + 1.96 * se),
            'P_value': pval
        })
    except Exception as e:
        print(f"Error Sens 4 for {out_name}: {e}")

df_sens = pd.DataFrame(sens_results)
df_sens.to_csv(os.path.join(OUT_DIR, 'sensitivity_analyses.csv'), index=False)

# 2. VANDERWEELE E-VALUES FOR UNMEASURED CONFOUNDING
print("\n=== CALCULATING VANDERWEELE E-VALUES ===")
def compute_e_value(est, ci_low, ci_high):
    # E-value for RR / PR / OR (approx RR)
    # If est < 1, invert
    point = est if est >= 1.0 else 1.0 / est
    e_val_point = point + np.sqrt(point * (point - 1.0))
    
    # E-value for confidence interval
    # The bound closer to the null (1.0)
    if est >= 1.0:
        lim = ci_low
    else:
        lim = 1.0 / ci_high
    
    if lim <= 1.0:
        e_val_ci = 1.0
    else:
        e_val_ci = lim + np.sqrt(lim * (lim - 1.0))
    return e_val_point, e_val_ci

# Compute E-values for Model 5 Primary findings
df_m5 = pd.read_csv(os.path.join(OUT_DIR, 'primary_nested_models.csv'))
df_m5_clean = df_m5[df_m5['Model'] == 'Model_5'].copy()

eval_list = []
for _, row in df_m5_clean.iterrows():
    e_pt, e_ci = compute_e_value(row['Estimate'], row['CI_Low'], row['CI_High'])
    eval_list.append({
        'Outcome': row['Outcome'],
        'Biomarker': row['Biomarker'],
        'Estimate': row['Estimate'],
        '95CI': f"[{row['CI_Low']:.2f}, {row['CI_High']:.2f}]",
        'P_value': row['P_value'],
        'E_Value_Point': e_pt,
        'E_Value_CI': e_ci
    })

df_evalues = pd.DataFrame(eval_list)
df_evalues.to_csv(os.path.join(OUT_DIR, 'vanderweele_evalues.csv'), index=False)

# 3. INCREMENTAL DISCRIMINATION (AUC, DeLong, NRI, IDI)
print("\n=== INCREMENTAL VALUE OF GGT & UA ADDED TO CRP ===")

def fast_delong_test(y_true, p1, p2):
    m = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    v10 = np.mean(p1[pos_idx, None] > p1[None, neg_idx], axis=1) + 0.5 * np.mean(p1[pos_idx, None] == p1[None, neg_idx], axis=1)
    v01 = np.mean(p1[pos_idx, None] < p1[None, neg_idx], axis=0) + 0.5 * np.mean(p1[pos_idx, None] == p1[None, neg_idx], axis=0)
    v20 = np.mean(p2[pos_idx, None] > p2[None, neg_idx], axis=1) + 0.5 * np.mean(p2[pos_idx, None] == p2[None, neg_idx], axis=1)
    v02 = np.mean(p2[pos_idx, None] < p2[None, neg_idx], axis=0) + 0.5 * np.mean(p2[pos_idx, None] == p2[None, neg_idx], axis=0)
    auc1 = np.mean(v10)
    auc2 = np.mean(v20)
    s10 = np.cov(v10, v20)
    s01 = np.cov(v01, v02)
    s = s10 / m + s01 / n
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (auc2 - auc1) / np.sqrt(max(1e-12, var_diff))
    pval = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return auc1, auc2, auc2 - auc1, pval

def compute_nri_idi(y_true, p_base, p_new):
    events = (y_true == 1)
    nonevents = (y_true == 0)
    nri_e = np.mean(p_new[events] > p_base[events]) - np.mean(p_new[events] < p_base[events])
    nri_ne = np.mean(p_new[nonevents] < p_base[nonevents]) - np.mean(p_new[nonevents] > p_base[nonevents])
    nri = nri_e + nri_ne
    idi = (np.mean(p_new[events]) - np.mean(p_base[events])) - (np.mean(p_new[nonevents]) - np.mean(p_base[nonevents]))
    return nri, idi

inc_results = []
np.random.seed(42)

for target_var, target_name in [
    ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease'),
    ('Suicide_Attempt_bin', 'Suicide Attempt')
]:
    y_arr = df_p[target_var].values
    
    # Model 1 (Base): Age, Sex, Smoker, BMI, log_CRP
    f_base = f"{target_var} ~ Age + Is_Male + Smoker + BMI + log_CRP"
    m_base = smf.logit(f_base, data=df_p).fit(disp=0)
    p_base = m_base.predict(df_p).values
    
    # Model 2 (Enhanced): Base + log_GGT + UA
    f_enh = f"{target_var} ~ Age + Is_Male + Smoker + BMI + log_CRP + log_GGT + UA_sex_z"
    m_enh = smf.logit(f_enh, data=df_p).fit(disp=0)
    p_enh = m_enh.predict(df_p).values
    
    auc_base, auc_enh, delta_auc, p_delong = fast_delong_test(y_arr, p_base, p_enh)
    nri_pt, idi_pt = compute_nri_idi(y_arr, p_base, p_enh)
    
    # Bootstrap CIs for NRI and IDI (1,000 resamples)
    b_nri, b_idi = [], []
    for _ in range(1000):
        idx = np.random.choice(len(df_p), size=len(df_p), replace=True)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        nri_b, idi_b = compute_nri_idi(y_arr[idx], p_base[idx], p_enh[idx])
        b_nri.append(nri_b)
        b_idi.append(idi_b)
    
    ci_nri = np.percentile(b_nri, [2.5, 97.5])
    ci_idi = np.percentile(b_idi, [2.5, 97.5])
    
    inc_results.append({
        'Outcome': target_name,
        'AUC_Base (CRP + Covars)': auc_base,
        'AUC_Enhanced (+ GGT + UA)': auc_enh,
        'Delta_AUC': delta_auc,
        'P_DeLong': p_delong,
        'Continuous_NRI': nri_pt,
        'NRI_95CI': f"[{ci_nri[0]:.3f}, {ci_nri[1]:.3f}]",
        'IDI': idi_pt,
        'IDI_95CI': f"[{ci_idi[0]:.3f}, {ci_idi[1]:.3f}]"
    })

df_inc = pd.DataFrame(inc_results)
df_inc.to_csv(os.path.join(OUT_DIR, 'incremental_discrimination_metrics.csv'), index=False)

print("\nSensitivity analyses summary:")
print(df_sens.to_string(index=False))

print("\nIncremental discrimination metrics:")
print(df_inc.to_string(index=False))
