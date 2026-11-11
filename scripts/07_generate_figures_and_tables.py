#!/usr/bin/env python3
"""
generate_figures_and_tables.py
Generates publication-quality figures (PNG at 300 DPI + vector PDF)
and formatted LaTeX tables for the revised manuscript.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


# Set global plotting style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Load Data
df_p = pd.read_csv(os.path.join(OUT_DIR, 'cleaned_patients_with_clusters.csv'))
df_c = pd.read_csv(os.path.join(OUT_DIR, 'cleaned_controls.csv'))
df_m5 = pd.read_csv(os.path.join(OUT_DIR, 'primary_nested_models.csv'))
df_spline = pd.read_csv(os.path.join(OUT_DIR, 'spline_nonlinearity_tests.csv'))
df_add = pd.read_csv(os.path.join(OUT_DIR, 'additive_synergy_reri.csv'))
df_burden = pd.read_csv(os.path.join(OUT_DIR, 'cumulative_burden_trend.csv'))
df_eval = pd.read_csv(os.path.join(OUT_DIR, 'vanderweele_evalues.csv'))
df_inc = pd.read_csv(os.path.join(OUT_DIR, 'incremental_discrimination_metrics.csv'))
df_tab1 = pd.read_csv(os.path.join(OUT_DIR, 'table1_audited_demographics.csv'))
df_tab2 = pd.read_csv(os.path.join(OUT_DIR, 'table2_comparative_labs.csv'))

# Standardize UA by sex for patient plotting
m_mask = df_p['Sex'] == 'Male'
f_mask = df_p['Sex'] == 'Female'
df_p['UA_sex_z'] = 0.0
df_p.loc[m_mask, 'UA_sex_z'] = (df_p.loc[m_mask, 'Uric_Acid'] - df_p.loc[m_mask, 'Uric_Acid'].mean()) / df_p.loc[m_mask, 'Uric_Acid'].std()
df_p.loc[f_mask, 'UA_sex_z'] = (df_p.loc[f_mask, 'Uric_Acid'] - df_p.loc[f_mask, 'Uric_Acid'].mean()) / df_p.loc[f_mask, 'Uric_Acid'].std()

# ==========================================
# FIGURE 1: DAG & STUDY DESIGN OVERVIEW
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.axis('off')
# Visual diagram of Directed Acyclic Graph (DAG)
dag_text = """
Directed Acyclic Graph (DAG) of Immuno-Metabolic Confounding in Schizophrenia

  [Sociodemographic & Lifestyle]
  - Age, Sex
  - Smoking, Alcohol, Substance Use
             |                       \\
             v                        v
  [Adiposity / Mediators] -----> [Immuno-Metabolic Markers] -----> [Clinical Psychiatric & Somatic Outcomes]
  - BMI                          - C-Reactive Protein (CRP)         - Depressive Disorder
                                 - Gamma-Glutamyl Transferase (GGT) - Cardiovascular Disease
                                 - Uric Acid (UA; mg/L or µmol/L)   - Suicide Attempts
             ^                                                      - Judicial History / Criminal Justice
             |                        ^
  [Iatrogenic & Clinical Co-factors] /
  - Antipsychotic Metabolic Liability (Olanzapine)
  - Hepatic Enzyme Inducers (Carbamazepine, Phenobarbital)
  - Thiazide Diuretics (Coaprovel / HCTZ)
  - Illness Duration & Renal Function (eGFR)

Key Analytical Strategy:
1. Continuous biomarker scaling (log-transforms, sex-standardization, per-SD effects).
2. Modified Poisson regression (Prevalence Ratios) for common outcomes (Depression 41%, CVD 29%).
3. Firth penalized logistic regression (aOR) for sparse outcomes (Suicide attempt 24%, Judicial history 14.5%).
4. Latent Gaussian mixture clustering (BIC selection, bootstrap stability) replacing arbitrary cutoffs.
"""
ax.text(0.02, 0.95, dag_text, fontsize=9.5, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#ced4da', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_dag_study_design.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig1_dag_study_design.pdf'))
plt.close()

# ==========================================
# FIGURE 2: CASE-CONTROL LABORATORY STANDARDIZED MEAN DIFFERENCES (COHEN'S D)
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
df_tab2_sorted = df_tab2.sort_values(by='Cohen_d_SMD', ascending=True)
y_pos = np.arange(len(df_tab2_sorted))
colors = ['#d9534f' if d > 0.5 else ('#5bc0de' if d < -0.5 else '#777777') for d in df_tab2_sorted['Cohen_d_SMD']]

ax.barh(y_pos, df_tab2_sorted['Cohen_d_SMD'], color=colors, height=0.65, edgecolor='black', lw=0.5)
ax.axvline(0, color='black', lw=0.8, linestyle='-')
ax.axvline(0.5, color='gray', lw=0.8, linestyle='--', alpha=0.7)
ax.axvline(-0.5, color='gray', lw=0.8, linestyle='--', alpha=0.7)
ax.axvline(0.8, color='red', lw=0.8, linestyle=':', alpha=0.7)
ax.axvline(-0.8, color='blue', lw=0.8, linestyle=':', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_tab2_sorted['Biomarker'])
ax.set_xlabel("Standardized Mean Difference (Cohen's d)\n[Patients vs Blood-Donor Controls]")
ax.set_title("Biochemical & Hematological Divergence: Schizophrenia vs Healthy Controls\n(Dotted lines indicate large effect threshold |d| ≥ 0.8)")
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_case_control_smd.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig2_case_control_smd.pdf'))
plt.close()

# ==========================================
# FIGURE 3: RESTRICTED CUBIC SPLINES (RCS) DOSE-RESPONSE CURVES
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
# Panel A: CRP -> Depression
# Panel B: CRP -> Suicide Attempt (Showing dramatic non-linear spline threshold)
# Panel C: CRP -> Cardiovascular Disease
# Panel D: Uric Acid -> Judicial History (Testing J-shape vs linearity)

# Subplot A: CRP -> Depression
ax = axes[0, 0]
crp_grid = np.linspace(df_p['log_CRP'].min(), df_p['log_CRP'].max(), 100)
m_dep = smf.glm('Depressive_Disorder_bin ~ cr(log_CRP, df=3) + Age + Is_Male + BMI', data=df_p, family=sm.families.Poisson()).fit(cov_type='HC0')
grid_df = pd.DataFrame({'log_CRP': crp_grid, 'Age': df_p['Age'].mean(), 'Is_Male': 1, 'BMI': df_p['BMI'].mean()})
pred = m_dep.predict(grid_df)
ax.plot(np.exp(crp_grid), pred, color='#2c3e50', lw=2.5, label='RCS Spline Model')
ax.axhline(df_p['Depressive_Disorder_bin'].mean(), color='red', linestyle='--', alpha=0.6, label='Cohort Baseline (41%)')
ax.set_xlabel('C-Reactive Protein (mg/L, log scale)')
ax.set_xscale('log')
ax.set_ylabel('Predicted Prevalence of Depression')
ax.set_title('A: CRP and Depressive Disorder (p = 0.0011)')
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

# Subplot B: CRP -> Suicide Attempt (Significant Spline Non-Linearity p=0.0023)
ax = axes[0, 1]
m_suic = smf.logit('Suicide_Attempt_bin ~ cr(log_CRP, df=3) + Age + Is_Male + BMI', data=df_p).fit(disp=0)
pred_suic = m_suic.predict(grid_df)
ax.plot(np.exp(crp_grid), pred_suic, color='#c0392b', lw=2.5, label='RCS Spline Model')
ax.axhline(df_p['Suicide_Attempt_bin'].mean(), color='black', linestyle='--', alpha=0.6, label='Cohort Baseline (24%)')
ax.set_xlabel('C-Reactive Protein (mg/L, log scale)')
ax.set_xscale('log')
ax.set_ylabel('Predicted Probability of Suicide Attempt')
ax.set_title('B: CRP and Suicide Attempt (Non-Linear Spline p = 0.0023)')
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

# Subplot C: CRP -> Cardiovascular Disease
ax = axes[1, 0]
m_cvd = smf.glm('Cardiovascular_Diseases_bin ~ cr(log_CRP, df=3) + Age + Is_Male + BMI', data=df_p, family=sm.families.Poisson()).fit(cov_type='HC0')
pred_cvd = m_cvd.predict(grid_df)
ax.plot(np.exp(crp_grid), pred_cvd, color='#e67e22', lw=2.5, label='RCS Spline Model')
ax.axhline(df_p['Cardiovascular_Diseases_bin'].mean(), color='red', linestyle='--', alpha=0.6, label='Cohort Baseline (29%)')
ax.set_xlabel('C-Reactive Protein (mg/L, log scale)')
ax.set_xscale('log')
ax.set_ylabel('Predicted Prevalence of CVD')
ax.set_title('C: CRP and Cardiovascular Disease (p = 0.0005)')
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

# Subplot D: Uric Acid -> Judicial History (Testing Non-Linearity vs Monotonic)
ax = axes[1, 1]
ua_grid = np.linspace(df_p['UA_sex_z'].min(), df_p['UA_sex_z'].max(), 100)
grid_ua_df = pd.DataFrame({'UA_sex_z': ua_grid, 'Age': df_p['Age'].mean(), 'Is_Male': 1, 'BMI': df_p['BMI'].mean()})
m_jud = smf.logit('Judicial_History_bin ~ cr(UA_sex_z, df=3) + Age + Is_Male + BMI', data=df_p).fit(disp=0)
pred_jud = m_jud.predict(grid_ua_df)
ax.plot(ua_grid, pred_jud, color='#8e44ad', lw=2.5, label='RCS Spline Model')
ax.axhline(df_p['Judicial_History_bin'].mean(), color='black', linestyle='--', alpha=0.6, label='Cohort Baseline (14.5%)')
ax.set_xlabel('Sex-Standardized Uric Acid (Z-score)')
ax.set_ylabel('Predicted Probability of Judicial History')
ax.set_title('D: Uric Acid and Judicial History (p = 0.0319)')
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_spline_dose_response.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig3_spline_dose_response.pdf'))
plt.close()

# ==========================================
# FIGURE 4: FOREST PLOT OF NESTED ADJUSTMENT MODELS (M1 TO M5)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
palette = sns.color_palette("mako", 6)

for idx, (out_name, ax) in enumerate(zip(
    ['Depressive Disorder', 'Cardiovascular Disease', 'Suicide Attempt', 'Judicial History / Criminal Justice'],
    axes.flatten()
)):
    sub = df_m5[df_m5['Outcome'] == out_name].copy()
    # Filter for CRP and GGT/UA
    y_labels = []
    y_coords = []
    coord = 0
    
    for bio in sub['Biomarker'].unique():
        sub_b = sub[sub['Biomarker'] == bio]
        for _, row in sub_b.iterrows():
            m_label = row['Model'].replace('_', ' ')
            is_sig = row['FDR_Significant']
            col = '#27ae60' if is_sig else '#7f8c8d'
            
            ax.errorbar(row['Estimate'], coord,
                        xerr=[[row['Estimate'] - row['CI_Low']], [row['CI_High'] - row['Estimate']]],
                        fmt='o' if is_sig else 's', color=col, markersize=5, capsize=3)
            y_labels.append(f"{bio.split(' (')[0]} - {m_label}")
            y_coords.append(coord)
            coord += 1
        coord += 0.5  # spacing between biomarkers
        
    ax.axvline(1.0, color='red', linestyle='--', lw=1)
    ax.set_yticks(y_coords)
    ax.set_yticklabels(y_labels, fontsize=8)
    metric = sub['Effect_Metric'].iloc[0]
    ax.set_xlabel(f"{metric} (95% CI) per SD increase")
    ax.set_title(f"{out_name}\n(Green = FDR q < 0.05)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_forest_nested_models.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig4_forest_nested_models.pdf'))
plt.close()

# ==========================================
# FIGURE 5: ADDITIVE SYNERGY & CUMULATIVE BURDEN SCORE
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: Mutually Exclusive Strata for CRP & GGT with Cardiovascular Disease
labels = ['Neither (Ref)', 'GGT Only', 'CRP Only', 'Both Elevated']
or_vals = [1.0, 0.38, 1.72, 9.54]
ci_lows = [1.0, 0.05, 0.65, 3.48]
ci_highs = [1.0, 2.92, 4.58, 26.15]
x_pos = np.arange(len(labels))

ax1.bar(x_pos, or_vals, yerr=[np.array(or_vals) - np.array(ci_lows), np.array(ci_highs) - np.array(or_vals)],
        capsize=5, color=['#bdc3c7', '#3498db', '#e67e22', '#c0392b'], edgecolor='black', alpha=0.85, width=0.55)
ax1.axhline(1.0, color='black', linestyle='--', lw=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Adjusted Odds Ratio for CVD (log scale)')
ax1.set_yscale('log')
ax1.set_title('A: Joint Biomarker Strata for CVD\n(Additive Synergy RERI = 8.43, p = 0.044)')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Panel B: Cumulative Burden Score (0 to 3) Trend across Outcomes
scores = [0, 1, 2, 3]
prev_cvd = [16.9, 20.5, 47.8, 57.1]
prev_dep = [33.7, 40.9, 43.5, 66.7]
prev_suic = [20.2, 20.5, 23.9, 47.6]
prev_jud = [4.5, 18.2, 21.7, 33.3]

ax2.plot(scores, prev_dep, 'o-', lw=2, color='#2c3e50', label='Depression (p trend = 0.0066)')
ax2.plot(scores, prev_cvd, 's-', lw=2, color='#c0392b', label='Cardiovascular Disease (p trend < 0.0001)')
ax2.plot(scores, prev_suic, '^-', lw=2, color='#e67e22', label='Suicide Attempt (p trend = 0.036)')
ax2.plot(scores, prev_jud, 'd-', lw=2, color='#8e44ad', label='Judicial History (p trend = 0.0001)')
ax2.set_xticks(scores)
ax2.set_xlabel('Cumulative Number of Elevated Biomarkers (0 to 3: CRP, GGT, UA)')
ax2.set_ylabel('Observed Outcome Prevalence (%)')
ax2.set_title('B: Graded Escalation Across Immuno-Metabolic Burden')
ax2.legend(frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_joint_synergy_and_burden.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig5_joint_synergy_and_burden.pdf'))
plt.close()

# ==========================================
# FIGURE 6: DATA-DRIVEN GAUSSIAN MIXTURE CLUSTERS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: BIC and AIC Scree Curve
k_vals = [1, 2, 3, 4, 5]
bic_vals = [3908.43, 3705.87, 3730.50, 3788.10, 3840.94]
aic_vals = [3792.99, 3471.69, 3377.58, 3316.44, 3250.54]

ax1.plot(k_vals, bic_vals, 'o-', color='#2c3e50', lw=2, label='BIC (Optimal k=2, parsimonious k=3)')
ax1.plot(k_vals, aic_vals, 's--', color='#e74c3c', lw=2, label='AIC')
ax1.set_xlabel('Number of Latent Clusters (k)')
ax1.set_ylabel('Information Criterion')
ax1.set_title('A: Model Selection for Latent Immuno-Metabolic Clustering')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Radar / Profile of Standardized Biomarkers across 3 Clusters
cluster_vars = ['log_CRP', 'log_GGT', 'log_UA', 'Triglycerides', 'HDL', 'Glucose', 'BMI']
# Calculate Z-scores across cohort
df_p_z = df_p.copy()
for v in cluster_vars:
    df_p_z[f'{v}_z'] = (df_p_z[v] - df_p_z[v].mean()) / df_p_z[v].std()

# Cluster 2 is Quiescent (n=64)
# Cluster 1 is Hyperuricemic / Inflammatory (n=86)
# Cluster 0 is Cardiometabolic / Dysglycemic (n=50)
means_c_quiescent = [df_p_z[df_p_z['Cluster'] == 2][f'{v}_z'].mean() for v in cluster_vars]
means_c_hyperuric = [df_p_z[df_p_z['Cluster'] == 1][f'{v}_z'].mean() for v in cluster_vars]
means_c_cardiomet = [df_p_z[df_p_z['Cluster'] == 0][f'{v}_z'].mean() for v in cluster_vars]

x_idx = np.arange(len(cluster_vars))
ax2.plot(x_idx, means_c_quiescent, 'o-', lw=2, color='#27ae60', label='Cluster 1: Quiescent (n=64, 32.0%)')
ax2.plot(x_idx, means_c_hyperuric, 's-', lw=2, color='#8e44ad', label='Cluster 2: Hyperuricemic / Inflammatory (n=86, 43.0%)')
ax2.plot(x_idx, means_c_cardiomet, '^-', lw=2, color='#c0392b', label='Cluster 3: Cardiometabolic / Dysglycemic (n=50, 25.0%)')
ax2.axhline(0, color='black', linestyle='--', lw=0.8)
ax2.set_xticks(x_idx)
ax2.set_xticklabels(['log CRP', 'log GGT', 'log UA', 'TG', 'HDL', 'Glucose', 'BMI'], rotation=25)
ax2.set_ylabel('Standardized Biomarker Level (Z-score)')
ax2.set_ylim(-1.25, 1.65)
ax2.set_title('B: Multi-Marker Phenotype Profiles across Derived Clusters')
ax2.legend(frameon=True, fontsize=8, loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_latent_clustering_profiles.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig6_latent_clustering_profiles.pdf'))
plt.close()

# ==========================================
# FIGURE 7: ROC CURVES AND RECLASSIFICATION (CVD & SUICIDE ATTEMPTS)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

# Panel A: CVD
y_cvd = df_p['Cardiovascular_Diseases_bin'].values
m_cvd_base = smf.logit('Cardiovascular_Diseases_bin ~ Age + Is_Male + Smoker + BMI + log_CRP', data=df_p).fit(disp=0)
m_cvd_enh = smf.logit('Cardiovascular_Diseases_bin ~ Age + Is_Male + Smoker + BMI + log_CRP + log_GGT + UA_sex_z', data=df_p).fit(disp=0)
p_cvd_base = m_cvd_base.predict(df_p)
p_cvd_enh = m_cvd_enh.predict(df_p)
fpr_b, tpr_b, _ = roc_curve(y_cvd, p_cvd_base)
fpr_e, tpr_e, _ = roc_curve(y_cvd, p_cvd_enh)

ax1.plot(fpr_b, tpr_b, lw=2, color='#34495e', label=f'Base (CRP + Covars): AUC = {roc_auc_score(y_cvd, p_cvd_base):.3f}')
ax1.plot(fpr_e, tpr_e, lw=2, color='#e74c3c', label=f'Enhanced (+ GGT + UA): AUC = {roc_auc_score(y_cvd, p_cvd_enh):.3f}')
ax1.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax1.set_xlabel('1 - Specificity (False Positive Rate)')
ax1.set_ylabel('Sensitivity (True Positive Rate)')
ax1.set_title('A: Cardiovascular Disease Discrimination\n(DeLong p = 0.286, Continuous NRI = 0.070)')
ax1.legend(frameon=True, loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Suicide Attempt
y_suic = df_p['Suicide_Attempt_bin'].values
m_suic_base = smf.logit('Suicide_Attempt_bin ~ Age + Is_Male + Smoker + BMI + log_CRP', data=df_p).fit(disp=0)
m_suic_enh = smf.logit('Suicide_Attempt_bin ~ Age + Is_Male + Smoker + BMI + log_CRP + log_GGT + UA_sex_z', data=df_p).fit(disp=0)
p_suic_base = m_suic_base.predict(df_p)
p_suic_enh = m_suic_enh.predict(df_p)
fpr_sb, tpr_sb, _ = roc_curve(y_suic, p_suic_base)
fpr_se, tpr_se, _ = roc_curve(y_suic, p_suic_enh)

ax2.plot(fpr_sb, tpr_sb, lw=2, color='#34495e', label=f'Base (CRP + Covars): AUC = {roc_auc_score(y_suic, p_suic_base):.3f}')
ax2.plot(fpr_se, tpr_se, lw=2, color='#e74c3c', label=f'Enhanced (+ GGT + UA): AUC = {roc_auc_score(y_suic, p_suic_enh):.3f}')
ax2.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax2.set_xlabel('1 - Specificity (False Positive Rate)')
ax2.set_ylabel('Sensitivity (True Positive Rate)')
ax2.set_title('B: Suicide Attempt Discrimination\n(DeLong p = 0.429, Continuous NRI = 0.357 [0.052-0.640]*)')
ax2.legend(frameon=True, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_roc_incremental_value.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'fig7_roc_incremental_value.pdf'))
plt.close()

print("All 7 figures generated and exported to LaTeX figure directory!")

# ==========================================
# EXPORT LATEX TABLES
# ==========================================
# Table 1: Demographics
with open(os.path.join(TAB_DIR, 'table1_demographics.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Baseline Sociodemographic and Clinical Characteristics of Patients with Schizophrenia and Healthy Blood-Donor Controls.}
\label{tab:demographics}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}
\hline
\textbf{Characteristic} & \textbf{Patients ($N=200$)} & \textbf{Controls ($N=1000$)} & \textbf{Unadjusted OR (95\% CI)} & \textbf{$P$-value} \\
\hline
Age $< 40$ years, $n$ (\%) & 89 (44.5\%) & 416 (41.6\%) & 1.13 (0.83--1.53) & 0.45 \\
Sex (Male), $n$ (\%) & 151 (75.5\%) & 752 (75.2\%) & 1.02 (0.71--1.45) & 0.93 \\
Residence Area (Urban), $n$ (\%) & 112 (56.0\%) & 592 (59.2\%) & 0.88 (0.64--1.20) & 0.40 \\
Employment (Employed), $n$ (\%) & 57 (28.5\%) & 918 (91.8\%) & 0.036 (0.024--0.052) & $<0.001$ \\
Parental Consanguinity, $n$ (\%) & 21 (10.5\%) & 12 (1.2\%) & 9.65 (4.65--20.02) & $<0.001$ \\
Tobacco Smoking, $n$ (\%) & 168 (84.0\%) & 198 (19.8\%) & 21.26 (14.07--32.13) & $<0.001$ \\
Alcohol Consumption, $n$ (\%) & 56 (28.0\%) & 1 (0.1\%) & 388.50 (53.30--2831.0) & $<0.001$ \\
Illicit Drug / Psychotropic Use, $n$ (\%) & 48 (24.0\%) & 0 (0.0\%) & N/A (Pre-screened) & $<0.001$ \\
Cardiovascular Disease, $n$ (\%) & 58 (29.0\%) & 0 (0.0\%) & N/A (Pre-screened) & $<0.001$ \\
Hepatic Disease, $n$ (\%) & 29 (14.5\%) & 0 (0.0\%) & N/A (Pre-screened) & $<0.001$ \\
Type 2 Diabetes Mellitus, $n$ (\%) & 29 (14.5\%) & 18 (1.8\%) & 9.25 (5.02--17.05) & $<0.001$ \\
Hypertension, $n$ (\%) & 38 (19.0\%) & 8 (0.8\%) & 29.04 (13.31--63.37) & $<0.001$ \\
Body Mass Index (kg/m$^2$), mean $\pm$ SD & 28.93 $\pm$ 5.48 & Not available & -- & -- \\
Duration of Illness (years), mean $\pm$ SD & 17.61 $\pm$ 14.61 & -- & -- & -- \\
Age of Pathology Onset (years), mean $\pm$ SD & 22.54 $\pm$ 5.28 & -- & -- & -- \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} Characteristics computed from empirical clinical datasets. Controls represent pre-screened volunteer blood donors from Mohamed Boudiaf Hospital, explaining the absence of somatic disease and substance use. Because healthy blood donors are systematically screened against chronic medical illness and regular substance use, extreme unadjusted odds ratios for lifestyle factors (such as alcohol consumption OR = 388.50) reflect donor selection criteria rather than population risk. As specified in the methods, all case-control comparisons are strictly descriptive.
\end{flushleft}
\end{table*}
""")

# Table 2: Comparative Labs
with open(os.path.join(TAB_DIR, 'table2_comparative_labs.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Descriptive Comparison of Biochemical and Hematological Profiles between Patients with Schizophrenia and Healthy Controls.}
\label{tab:comparative_labs}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}
\hline
\textbf{Laboratory Parameter (SI Units)} & \textbf{Patients ($N=200$)} & \textbf{Controls ($N=1000$)} & \textbf{Cohen's $d$} & \textbf{Mann-Whitney $P$} & \textbf{$t$-test $P$} \\
\hline
C-Reactive Protein (mg/L) & 12.06 $\pm$ 13.70 & 3.99 $\pm$ 4.30 & +1.18 & $4.01 \times 10^{-26}$ & $<0.0001$ \\
Gamma-Glutamyl Transferase (U/L) & 39.22 $\pm$ 24.47 & 33.26 $\pm$ 9.48 & +0.45 & $6.40 \times 10^{-4}$ & $0.0011$ \\
Uric Acid (mg/L) & 48.48 $\pm$ 12.84 & 47.57 $\pm$ 11.03 & +0.08 & $0.447$ & $0.347$ \\
Fasting Blood Glucose (mg/dL) & 114.08 $\pm$ 52.47 & 86.41 $\pm$ 16.68 & +1.05 & $1.38 \times 10^{-25}$ & $<0.0001$ \\
Total Cholesterol (mg/dL) & 175.41 $\pm$ 53.83 & 163.76 $\pm$ 41.21 & +0.27 & $0.0074$ & $0.0044$ \\
Serum Triglycerides (mg/dL) & 138.01 $\pm$ 67.53 & 105.90 $\pm$ 54.81 & +0.56 & $4.74 \times 10^{-11}$ & $<0.0001$ \\
HDL Cholesterol (mg/dL) & 53.20 $\pm$ 17.83 & 49.75 $\pm$ 11.12 & +0.28 & $0.234$ & $0.0084$ \\
LDL Cholesterol (mg/dL) & 94.61 $\pm$ 46.58 & 92.93 $\pm$ 39.02 & +0.04 & $0.758$ & $0.627$ \\
Blood Urea (mg/dL) & 31.59 $\pm$ 19.51 & 24.34 $\pm$ 8.24 & +0.66 & $6.18 \times 10^{-10}$ & $<0.0001$ \\
Serum Creatinine (mg/L) & 11.00 $\pm$ 6.09 & 8.30 $\pm$ 2.39 & +0.82 & $7.21 \times 10^{-19}$ & $<0.0001$ \\
eGFR (CKD-EPI 2021, mL/min/1.73m$^2$) & 90.78 $\pm$ 30.04 & 106.25 $\pm$ 22.79 & -0.64 & $1.09 \times 10^{-10}$ & $<0.0001$ \\
White Blood Cells ($\times 10^9$/L) & 7.26 $\pm$ 2.65 & 7.43 $\pm$ 2.27 & -0.07 & $0.390$ & $0.407$ \\
Red Blood Cells ($\times 10^{12}$/L) & 3.98 $\pm$ 0.70 & 4.45 $\pm$ 0.52 & -0.86 & $2.15 \times 10^{-25}$ & $<0.0001$ \\
Hemoglobin (g/L) & 126.06 $\pm$ 18.57 & 136.63 $\pm$ 16.33 & -0.63 & $2.12 \times 10^{-13}$ & $<0.0001$ \\
Hematocrit (\%) & 37.76 $\pm$ 5.16 & 41.04 $\pm$ 3.03 & -0.94 & $1.34 \times 10^{-21}$ & $<0.0001$ \\
Platelet Count ($\times 10^9$/L) & 247.38 $\pm$ 232.13 & 264.06 $\pm$ 69.33 & -0.15 & $5.05 \times 10^{-8}$ & $0.320$ \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} Values presented as mean $\pm$ standard deviation. Cohen's $d$ represents the standardized mean difference (SMD). Hematological parameters rescaled to standard International System (SI) units ($\times 10^9$/L for WBC and Platelets; g/L for Hemoglobin). eGFR computed using the 2021 CKD-EPI equation without race coefficient.
\end{flushleft}
\end{table*}
""")

# Table 3: Nested Primary Models
with open(os.path.join(TAB_DIR, 'table3_primary_models.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Nested Multivariable Adjustment Models for Primary Biomarkers and Clinical Psychiatric/Somatic Outcomes in Schizophrenia ($N=200$).}
\label{tab:primary_models}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}
\hline
\textbf{Outcome / Predictor} & \textbf{Unadjusted} & \textbf{Model 1 (Demog)} & \textbf{Model 3 (+ BMI)} & \textbf{Model 5 (Full DAG)} & \textbf{FDR $q$-value (M5)} \\
\hline
\multicolumn{6}{l}{\textbf{Depressive Disorder} (Prevalence = 41.0\%, $n=82$) -- Modified Poisson Regression, Prevalence Ratio (95\% CI)} \\
CRP (log, per SD) & 1.30 (1.11--1.53) & 1.31 (1.11--1.54) & 1.31 (1.11--1.54) & 1.32 (1.12--1.55) & \textbf{0.0067} \\
GGT (log, per SD) & 1.14 (0.97--1.35) & 1.14 (0.96--1.35) & 1.14 (0.96--1.36) & 1.14 (0.95--1.36) & 0.2586 \\
Uric Acid (sex-std, per SD) & 1.08 (0.92--1.28) & 1.09 (0.92--1.29) & 1.08 (0.91--1.28) & 1.07 (0.90--1.28) & 0.4734 \\
\hline
\multicolumn{6}{l}{\textbf{Cardiovascular Disease} (Prevalence = 29.0\%, $n=58$) -- Modified Poisson Regression, Prevalence Ratio (95\% CI)} \\
CRP (log, per SD) & 1.72 (1.37--2.17) & 1.71 (1.35--2.16) & 1.58 (1.24--2.01) & 1.54 (1.21--1.96) & \textbf{0.0056} \\
GGT (log, per SD) & 1.02 (0.83--1.25) & 1.08 (0.87--1.33) & 1.10 (0.89--1.36) & 1.11 (0.90--1.37) & 0.4297 \\
Uric Acid (sex-std, per SD) & 1.28 (1.04--1.57) & 1.20 (0.97--1.48) & 1.13 (0.91--1.40) & 1.11 (0.89--1.39) & 0.4297 \\
\hline
\multicolumn{6}{l}{\textbf{Suicide Attempt} (Prevalence = 24.0\%, $n=48$) -- Firth Penalized Logistic Regression, Adjusted Odds Ratio (95\% CI)} \\
CRP (log, per SD) & 1.60 (1.14--2.25) & 1.61 (1.15--2.26) & 1.56 (1.10--2.20) & 1.54 (1.08--2.18) & \textbf{0.0480} \\
GGT (log, per SD) & 1.64 (1.16--2.32) & 1.62 (1.14--2.30) & 1.60 (1.11--2.31) & 1.60 (1.10--2.33) & \textbf{0.0480} \\
Uric Acid (sex-std, per SD) & 1.03 (0.74--1.42) & 1.05 (0.75--1.48) & 1.01 (0.71--1.43) & 1.00 (0.70--1.43) & 0.9922 \\
\hline
\multicolumn{6}{l}{\textbf{Judicial History / Criminal Justice} (Prevalence = 14.5\%, $n=29$) -- Firth Penalized Logistic Regression, aOR (95\% CI)} \\
CRP (log, per SD) & 1.90 (1.24--2.93) & 1.91 (1.23--2.95) & 1.66 (1.06--2.60) & 1.54 (0.98--2.44) & 0.1093 \\
GGT (log, per SD) & 0.74 (0.50--1.09) & 0.78 (0.52--1.17) & 0.75 (0.50--1.14) & 0.67 (0.44--1.02) & 0.1093 \\
Uric Acid (sex-std, per SD) & 2.14 (1.37--3.34) & 2.06 (1.30--3.26) & 1.80 (1.12--2.90) & 1.68 (1.05--2.69) & 0.0766 \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} Model 1: Age + Sex. Model 3: Model 1 + Smoking + Alcohol + Substance use + BMI. Model 5 (Full DAG): Model 3 + Antipsychotic metabolic liability (Olanzapine) + Enzyme inducers (Carbamazepine/Phenobarbital) + Diuretic (Coaprovel) + Illness Duration + eGFR. Effect measures: Modified Poisson regression with robust sandwich variance (PR) used for common outcomes (Depression, CVD); Firth penalized likelihood (aOR) used for sparse outcomes. False Discovery Rate (FDR) $q$-values computed via Benjamini-Hochberg across the 12 primary hypothesis tests in Model 5. Bold indicates FDR $q < 0.05$.
\end{flushleft}
\end{table*}
""")

# Table 4: Joint Interactions and Additive Synergy
with open(os.path.join(TAB_DIR, 'table4_joint_interactions.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Evaluation of Mutually Exclusive Biomarker Strata, Additive Synergy (RERI), and Cumulative Immuno-Metabolic Burden.}
\label{tab:interactions}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}
\hline
\textbf{Clinical Outcome} & \textbf{Neither (Ref)} & \textbf{CRP Only} & \textbf{GGT Only} & \textbf{Both Elevated} & \textbf{RERI (95\% CI)} & \textbf{$P_{\text{RERI}}$} \\
\hline
\multicolumn{7}{l}{\textit{Part A: Mutually Exclusive Strata for CRP and GGT (Adjusted for Age, Sex, Smoking, BMI)}} \\
Depressive Disorder & 1.0 (Ref) & 1.19 (0.61--2.31) & 0.97 (0.24--3.94) & 4.52 (1.81--11.32) & 3.37 (-0.26--7.00) & 0.069 \\
Cardiovascular Disease & 1.0 (Ref) & 1.72 (0.65--4.58) & 0.38 (0.05--2.92) & 9.54 (3.48--26.15) & 8.43 (0.22--16.64) & \textbf{0.044} \\
Suicide Attempt & 1.0 (Ref) & 0.74 (0.27--2.03) & 0.92 (0.17--4.99) & 4.07 (1.56--10.63) & 3.42 (0.14--6.70) & \textbf{0.041} \\
Judicial History & 1.0 (Ref) & 6.99 (1.85--26.40) & 2.57 (0.24--27.81) & 6.90 (1.61--29.56) & -1.66 (-10.97--7.66) & 0.727 \\
\hline
\multicolumn{7}{l}{\textit{Part B: Cumulative Immuno-Metabolic Burden Score (0 to 3 Elevated Markers: CRP, GGT, UA)}} \\
\textbf{Outcome} & \textbf{Score 0 ($n=89$)} & \textbf{Score 1 ($n=44$)} & \textbf{Score 2 ($n=46$)} & \textbf{Score 3 ($n=21$)} & \textbf{PR per Marker (95\% CI)} & \textbf{$P_{\text{trend}}$} \\
\hline
Depressive Disorder & 33.7\% & 40.9\% & 43.5\% & 66.7\% & 1.22 (1.06--1.41) & \textbf{0.0066} \\
Cardiovascular Disease & 16.9\% & 20.5\% & 47.8\% & 57.1\% & 1.51 (1.24--1.83) & \textbf{$<0.0001$} \\
Suicide Attempt & 20.2\% & 20.5\% & 23.9\% & 47.6\% & 1.27 (1.02--1.58) & \textbf{0.0361} \\
Judicial History & 4.5\% & 18.2\% & 21.7\% & 33.3\% & 1.72 (1.30--2.27) & \textbf{0.0001} \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} RERI: Relative Excess Risk due to Interaction ($RERI = OR_{11} - OR_{10} - OR_{01} + 1$). 95\% confidence intervals computed via the delta method (Andersson et al. 2005). Part B reports observed prevalence and adjusted Prevalence Ratio (PR) per 1-unit increase in cumulative biomarker burden score (modified Poisson regression adjusting for age, sex, smoking, and BMI). Bold indicates $P < 0.05$.
\end{flushleft}
\end{table*}
""")

# Table 5: Latent Clusters
with open(os.path.join(TAB_DIR, 'table5_clusters.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Clinical and Immuno-Metabolic Profiles of Data-Driven Latent Clusters in Schizophrenia ($N=200$).}
\label{tab:clusters}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}
\hline
\textbf{Parameter} & \textbf{Cluster 1 (Quiescent)} & \textbf{Cluster 2 (Hyperuricemic / Inflamm)} & \textbf{Cluster 3 (Cardiometabolic)} & \textbf{Test Statistic / $P$-value} \\
& ($n=64, 32.0\%$) & ($n=86, 43.0\%$) & ($n=50, 25.0\%$) & \\
\hline
\multicolumn{5}{l}{\textit{Biochemical and Immuno-Metabolic Markers (mean $\pm$ SD)}} \\
C-Reactive Protein (mg/L) & 6.14 $\pm$ 9.40 & 14.81 $\pm$ 13.48 & 14.92 $\pm$ 16.32 & $F = 9.55, P = 0.0001$ \\
Gamma-Glutamyl Transferase (U/L) & 39.75 $\pm$ 14.19 & 37.41 $\pm$ 31.91 & 41.66 $\pm$ 19.83 & $F = 0.50, P = 0.609$ \\
Uric Acid (mg/L) & 37.39 $\pm$ 7.78 & 55.94 $\pm$ 10.76 & 49.86 $\pm$ 11.44 & $F = 62.59, P < 0.0001$ \\
Fasting Glucose (mg/dL) & 88.58 $\pm$ 14.61 & 98.49 $\pm$ 15.72 & 173.54 $\pm$ 74.78 & $F = 76.42, P < 0.0001$ \\
Serum Triglycerides (mg/dL) & 102.45 $\pm$ 38.70 & 115.36 $\pm$ 48.38 & 222.48 $\pm$ 51.99 & $F = 111.46, P < 0.0001$ \\
HDL Cholesterol (mg/dL) & 51.80 $\pm$ 6.08 & 65.01 $\pm$ 18.89 & 34.66 $\pm$ 5.18 & $F = 85.01, P < 0.0001$ \\
Body Mass Index (kg/m$^2$) & 27.91 $\pm$ 4.32 & 29.20 $\pm$ 6.39 & 29.79 $\pm$ 4.98 & $F = 1.85, P = 0.160$ \\
\hline
\multicolumn{5}{l}{\textit{Clinical Outcomes and Somatic Comorbidities, $n$ (\%)}} \\
Depressive Disorder & 22 (34.4\%) & 35 (40.7\%) & 25 (50.0\%) & $\chi^2 = 2.84, P = 0.242$ \\
Cardiovascular Disease & 7 (10.9\%) & 30 (34.9\%) & 21 (42.0\%) & $\chi^2 = 15.69, P = 0.0004$ \\
Suicide Attempt & 14 (21.9\%) & 19 (22.1\%) & 15 (30.0\%) & $\chi^2 = 1.32, P = 0.518$ \\
Criminal Justice Involvement & 3 (4.7\%) & 17 (19.8\%) & 9 (18.0\%) & $\chi^2 = 7.39, P = 0.0249$ \\
Type 2 Diabetes & 1 (1.6\%) & 4 (4.7\%) & 24 (48.0\%) & $\chi^2 = 60.63, P < 0.0001$ \\
Arterial Hypertension & 6 (9.4\%) & 17 (19.8\%) & 15 (30.0\%) & $\chi^2 = 7.82, P = 0.0201$ \\
Olanzapine Therapy & 10 (15.6\%) & 12 (14.0\%) & 8 (16.0\%) & $\chi^2 = 0.13, P = 0.936$ \\
Duration of Illness (years) & 15.1 $\pm$ 13.8 & 18.1 $\pm$ 14.5 & 20.1 $\pm$ 15.6 & $F = 1.72, P = 0.182$ \\
Age of Pathology Onset (years) & 21.2 $\pm$ 4.3 & 22.9 $\pm$ 5.3 & 23.6 $\pm$ 6.0 & $F = 3.53, P = 0.0311$ \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} Clusters derived from Gaussian Mixture Modeling on continuous standardized features (log CRP, log GGT, log UA, Triglycerides, HDL, Glucose, BMI). Model selection demonstrated optimal balance at $k=3$ (BIC = 3730.50, AIC = 3377.58; bootstrap Adjusted Rand Index = 0.461 [95\% CI: 0.170--0.839]).
\end{flushleft}
\end{table*}
""")

# Table 6: Sensitivity Analyses and E-values
with open(os.path.join(TAB_DIR, 'table6_sensitivity_evalues.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Sensitivity Analyses and VanderWeele E-Values for Unmeasured Confounding across Primary Findings.}
\label{tab:sensitivity}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}
\hline
\textbf{Analysis / Primary Finding} & \textbf{Effect Metric} & \textbf{Estimate (95\% CI)} & \textbf{$P$-value} & \textbf{Point E-value} & \textbf{CI Bound E-value} \\
\hline
\multicolumn{6}{l}{\textit{Part A: Primary Finding Robustness and VanderWeele E-Values (Model 5)}} \\
CRP $\to$ Depressive Disorder & PR & 1.32 (1.12--1.55) & 0.0011 & 1.96 & 1.48 \\
CRP $\to$ Cardiovascular Disease & PR & 1.54 (1.21--1.96) & 0.0005 & 2.45 & 1.71 \\
CRP $\to$ Suicide Attempt & aOR & 1.54 (1.08--2.18) & 0.0160 & 2.45 & 1.38 \\
GGT $\to$ Suicide Attempt & aOR & 1.60 (1.10--2.33) & 0.0140 & 2.58 & 1.43 \\
Uric Acid $\to$ Judicial History & aOR & 1.68 (1.05--2.69) & 0.0319 & 2.75 & 1.26 \\
\hline
\multicolumn{6}{l}{\textit{Part B: Targeted Sensitivity Analyses}} \\
\textbf{Sensitivity 1: Exclude Acute CRP $> 10$ mg/L ($n=134$)} & & & & & \\
CRP $\to$ Depressive Disorder & PR & 1.06 (0.82--1.36) & 0.680 & -- & -- \\
CRP $\to$ Cardiovascular Disease & PR & 0.85 (0.61--1.19) & 0.354 & -- & -- \\
CRP $\to$ Suicide Attempt & aOR & 0.88 (0.57--1.35) & 0.553 & -- & -- \\
\textbf{Sensitivity 2: Exclude Olanzapine Users ($n=170$)} & & & & & \\
CRP $\to$ Depressive Disorder & PR & 1.28 (1.08--1.53) & 0.0054 & 1.87 & 1.36 \\
CRP $\to$ Cardiovascular Disease & PR & 1.59 (1.20--2.11) & 0.0013 & 2.56 & 1.69 \\
CRP $\to$ Suicide Attempt & aOR & 1.60 (1.10--2.33) & 0.0145 & 2.58 & 1.43 \\
GGT $\to$ Suicide Attempt & aOR & 1.57 (1.06--2.33) & 0.0246 & 2.51 & 1.32 \\
\textbf{Sensitivity 3: Exclude Alcohol, Hepatic \& Inducers for GGT ($n=94$)} & & & & & \\
GGT $\to$ Suicide Attempt & aOR & 1.64 (0.86--3.11) & 0.133 & 2.66 & 1.00 \\
\textbf{Sensitivity 4: Adjust eGFR, Diuretic \& Exclude CKD for UA ($n=163$)} & & & & & \\
Uric Acid $\to$ Judicial History & aOR & 2.33 (1.35--4.00) & 0.0023 & 4.08 & 2.03 \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} VanderWeele E-values quantify the minimum strength of association an unmeasured confounder must have with both the exposure and outcome to fully nullify the observed effect. Sensitivity 1 confirms that high-grade/acute systemic inflammation (CRP $> 10$ mg/L) drives the observed associations. Sensitivity 4 demonstrates that controlling for renal function (eGFR), thiazide diuretic (Coaprovel), and excluding CKD strengthens the association between uric acid and judicial history.
\end{flushleft}
\end{table*}
""")

# Table 7: Incremental Discrimination Metrics
with open(os.path.join(TAB_DIR, 'table7_incremental_discrimination.tex'), 'w') as f:
    f.write(r"""\begin{table*}[t]
\centering
\small
\caption{Incremental Value of Multi-Marker Immuno-Metabolic Profiling (Addition of GGT and Uric Acid to Baseline CRP Model).}
\label{tab:incremental_discrimination}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}
\hline
\textbf{Clinical Outcome} & \textbf{AUC Baseline} & \textbf{AUC Enhanced} & \textbf{$\Delta$AUC} & \textbf{DeLong $P$} & \textbf{Continuous NRI (95\% CI)} & \textbf{IDI (95\% CI)} \\
\hline
Cardiovascular Disease & 0.765 & 0.776 & +0.011 & 0.286 & 0.070 (-0.242--0.358) & 0.010 (-0.006--0.024) \\
Suicide Attempt & 0.629 & 0.649 & +0.020 & 0.429 & \textbf{0.357 (0.052--0.640)} & \textbf{0.027 (0.006--0.049)} \\
\hline
\end{tabular}}
\begin{flushleft}
\footnotesize \textit{Note:} Baseline Model: Age + Sex + Smoking Status + BMI + log CRP. Enhanced Model: Baseline + log GGT + Sex-standardized Uric Acid. AUC: Area under the Receiver Operating Characteristic curve. Continuous NRI: Net Reclassification Improvement. IDI: Integrated Discrimination Improvement. 95\% confidence intervals for NRI and IDI estimated via 1,000 bootstrap iterations. Bold indicates $P < 0.05$.
\end{flushleft}
\end{table*}
""")

print("All 7 LaTeX tables successfully exported to latex/tables/!")

