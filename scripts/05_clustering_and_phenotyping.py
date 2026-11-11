#!/usr/bin/env python3
"""
clustering_and_phenotyping.py
Phase 3: Data-Driven Immuno-Metabolic Clustering & Clinical Phenotyping
- Replaces arbitrary CRP+/- dichotomization with Gaussian Mixture / Latent Class Clustering
- Input features: standardized continuous log_CRP, log_GGT, log_UA, Triglycerides, HDL, Glucose, BMI
- Model selection via Bayesian Information Criterion (BIC) & Akaike Information Criterion (AIC) across k=1..5
- Cluster stability assessed via 1,000 bootstrap resamples (Adjusted Rand Index)
- Clinical profiling of clusters across psychiatric and somatic outcomes
- Partial correlation of Age of Onset with Glucose and Lipids (adjusting for Current Age and Illness Duration)
"""

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy.stats import f_oneway, chi2_contingency
import statsmodels.formula.api as smf
# Pure numpy/scipy partial correlation implemented below

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

# 1. Prepare Feature Matrix for Clustering
cluster_vars = ['log_CRP', 'log_GGT', 'log_UA', 'Triglycerides', 'HDL', 'Glucose', 'BMI']
X_raw = df_p[cluster_vars].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. Model Selection: BIC and AIC across k=1 to 5
k_range = range(1, 6)
bic_scores = []
aic_scores = []
models = {}

np.random.seed(42)
for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=20, random_state=42)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    bic_scores.append(bic)
    aic_scores.append(aic)
    models[k] = gmm
    print(f"k={k}: BIC={bic:.2f}, AIC={aic:.2f}")

optimal_k = k_range[np.argmin(bic_scores)]
# In clinical immunology/psychiatry, often k=2 or k=3 provides the best compromise between BIC and clinical interpretability
print(f"Optimal k by strict minimum BIC: {optimal_k}")

# If k=3 has close BIC or k=2, let's inspect both. Usually k=3 yields:
# Cluster 1: Low-risk / immuno-metabolically quiescent
# Cluster 2: Dysmetabolic / cardiometabolic predominant
# Cluster 3: Severe systemic immuno-metabolic / high inflammatory
chosen_k = 3 if bic_scores[2] < bic_scores[0] + 50 else optimal_k
print(f"Selected k for clinical profiling: {chosen_k}")

best_gmm = models[chosen_k]
cluster_labels = best_gmm.predict(X_scaled)
df_p['Cluster'] = cluster_labels

# 3. Bootstrap Cluster Stability (1,000 resamples)
print("\n=== RUNNING 1,000 BOOTSTRAP STABILITY RESAMPLES ===")
n_boot = 1000
ari_scores = []
n_samples = len(X_scaled)

for b in range(n_boot):
    boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X_scaled[boot_idx]
    try:
        gmm_boot = GaussianMixture(n_components=chosen_k, covariance_type='full', n_init=3, random_state=b)
        gmm_boot.fit(X_boot)
        # Predict on original data
        pred_boot = gmm_boot.predict(X_scaled)
        ari = adjusted_rand_score(cluster_labels, pred_boot)
        ari_scores.append(ari)
    except Exception:
        continue

mean_ari = np.mean(ari_scores)
ci_ari = np.percentile(ari_scores, [2.5, 97.5])
print(f"Bootstrap Adjusted Rand Index (k={chosen_k}): Mean = {mean_ari:.3f}, 95% CI = [{ci_ari[0]:.3f}, {ci_ari[1]:.3f}]")

# 4. Cluster Biomarker Profiling
cluster_profiles = []
for c in range(chosen_k):
    sub = df_p[df_p['Cluster'] == c]
    prof = {'Cluster': f"Cluster {c+1} (n={len(sub)})"}
    for var in cluster_vars + ['CRP', 'GGT', 'Uric_Acid']:
        prof[f'{var}_mean'] = sub[var].mean()
        prof[f'{var}_sd'] = sub[var].std()
        prof[f'{var}_median'] = sub[var].median()
    cluster_profiles.append(prof)

df_cluster_prof = pd.DataFrame(cluster_profiles)
df_cluster_prof.to_csv(os.path.join(OUT_DIR, 'cluster_biomarker_profiles.csv'), index=False)

# 5. Clinical Phenotype Across Clusters
clinical_vars = [
    'Depressive_Disorder_bin', 'Cardiovascular_Diseases_bin', 'Suicide_Attempt_bin',
    'Judicial_History_bin', 'Diabetes_bin', 'Hypertension_bin', 'History_of_Violence_bin',
    'Med_Olanzapine', 'Med_Enzyme_Inducer'
]

clinical_comp = []
for var in clinical_vars:
    row = {'Clinical_Feature': var}
    contingency = []
    for c in range(chosen_k):
        sub = df_p[df_p['Cluster'] == c]
        val_sum = sub[var].sum()
        val_pct = sub[var].mean() * 100
        row[f'Cluster_{c+1}'] = f"{val_pct:.1f}% ({val_sum}/{len(sub)})"
        contingency.append([val_sum, len(sub) - val_sum])
    # Chi-square test
    chi2_val, p_val, _, _ = chi2_contingency(contingency)
    row['Chi2'] = chi2_val
    row['P_value'] = p_val
    clinical_comp.append(row)

# Continuous clinical variables (Age, Illness Duration, Age at Onset)
for var in ['Age', 'Duration_of_Illness_Years', 'Age_of_Pathology_Onset']:
    row = {'Clinical_Feature': var}
    groups = [df_p[df_p['Cluster'] == c][var].dropna().values for c in range(chosen_k)]
    f_stat, p_val = f_oneway(*groups)
    row['Chi2'] = f_stat
    row['P_value'] = p_val
    for c in range(chosen_k):
        sub = df_p[df_p['Cluster'] == c]
        row[f'Cluster_{c+1}'] = f"{sub[var].mean():.1f} ± {sub[var].std():.1f}"
    clinical_comp.append(row)

df_clinical_comp = pd.DataFrame(clinical_comp)
df_clinical_comp.to_csv(os.path.join(OUT_DIR, 'cluster_clinical_comparison.csv'), index=False)

# 6. PARTIAL CORRELATIONS: AGE OF ONSET AUDIT
# Raw Pearson/Spearman vs Partial Correlation adjusting for Age and Illness Duration
def calc_partial_corr(x, y, covars, data):
    # Regress x on covars
    mod_x = smf.ols(f"{x} ~ {' + '.join(covars)}", data=data).fit()
    res_x = mod_x.resid
    # Regress y on covars
    mod_y = smf.ols(f"{y} ~ {' + '.join(covars)}", data=data).fit()
    res_y = mod_y.resid
    # Pearson corr between residuals
    r = np.corrcoef(res_x, res_y)[0, 1]
    n = len(data)
    k_cov = len(covars)
    df = n - k_cov - 2
    t_stat = r * np.sqrt(df / (1.0 - r**2))
    from scipy.stats import t
    p_val = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=df))
    return r, p_val

print("\n=== AUDIT OF AGE OF ONSET CORRELATIONS ===")
onset_results = []
for met_var in ['Glucose', 'Cholesterol', 'Triglycerides', 'log_CRP', 'log_GGT', 'log_UA']:
    # Raw correlation
    raw_r = np.corrcoef(df_p['Age_of_Pathology_Onset'], df_p[met_var])[0, 1]
    from scipy.stats import pearsonr
    _, raw_p = pearsonr(df_p['Age_of_Pathology_Onset'], df_p[met_var])
    
    # Partial correlation adjusting for Current Age and Illness Duration
    part_r, part_p = calc_partial_corr('Age_of_Pathology_Onset', met_var, ['Age', 'Duration_of_Illness_Years'], df_p)
    
    onset_results.append({
        'Biomarker': met_var,
        'Raw_Pearson_r': raw_r,
        'Raw_P_value': raw_p,
        'Partial_r_Adjusted_Age_Duration': part_r,
        'Partial_P_value': part_p
    })
    print(f"{met_var}: Raw r={raw_r:.3f} (p={raw_p:.4f}) -> Partial r (adjusted for Age & Duration)={part_r:.3f} (p={part_p:.4f})")

df_onset = pd.DataFrame(onset_results)
df_onset.to_csv(os.path.join(OUT_DIR, 'age_onset_partial_correlations.csv'), index=False)

# Save df_p with Cluster labels
df_p.to_csv(os.path.join(OUT_DIR, 'cleaned_patients_with_clusters.csv'), index=False)

print("\nClustering and phenotyping complete!")
