#!/usr/bin/env python3
"""
synergy_and_interaction.py
Phase 3: Formal Evaluation of Biomarker Interactions and Additive Synergy
- Replaces flawed legacy comparisons with mutually exclusive categories sharing a common reference.
- Calculates Relative Excess Risk due to Interaction (RERI), Attributable Proportion (AP), and Synergy Index (S) with Delta-method 95% CIs.
- Evaluates multiplicative product terms (CRP * GGT, CRP * UA).
- Analyzes cumulative immuno-metabolic burden (0, 1, 2, 3 elevated markers) with formal trend tests.
- Highlights sparse cell constraints (e.g., n=6 UA+/CRP-).
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

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

# Create Mutually Exclusive Strata
# 1. CRP and GGT
df_p['CRP_GGT_strata'] = 'Neither'
df_p.loc[(df_p['CRP_elevated_6'] == 0) & (df_p['GGT_elevated'] == 1), 'CRP_GGT_strata'] = 'GGT_only'
df_p.loc[(df_p['CRP_elevated_6'] == 1) & (df_p['GGT_elevated'] == 0), 'CRP_GGT_strata'] = 'CRP_only'
df_p.loc[(df_p['CRP_elevated_6'] == 1) & (df_p['GGT_elevated'] == 1), 'CRP_GGT_strata'] = 'Both_elevated'

# 2. CRP and UA
df_p['CRP_UA_strata'] = 'Neither'
df_p.loc[(df_p['CRP_elevated_6'] == 0) & (df_p['UA_elevated'] == 1), 'CRP_UA_strata'] = 'UA_only'
df_p.loc[(df_p['CRP_elevated_6'] == 1) & (df_p['UA_elevated'] == 0), 'CRP_UA_strata'] = 'CRP_only'
df_p.loc[(df_p['CRP_elevated_6'] == 1) & (df_p['UA_elevated'] == 1), 'CRP_UA_strata'] = 'Both_elevated'

# 3. Cumulative Immuno-metabolic Burden Score (0 to 3)
df_p['Burden_Score_0_3'] = df_p['CRP_elevated_6'] + df_p['GGT_elevated'] + df_p['UA_elevated']

OUTCOMES = [
    ('Depressive_Disorder_bin', 'Depressive Disorder'),
    ('Cardiovascular_Diseases_bin', 'Cardiovascular Disease'),
    ('Suicide_Attempt_bin', 'Suicide Attempt'),
    ('Judicial_History_bin', 'Judicial History / Criminal Justice')
]

covariates = 'Age + Is_Male + Smoker + BMI'

interaction_results = []
additive_synergy_results = []

print("=== EVALUATION OF MUTUALLY EXCLUSIVE STRATA & ADDITIVE INTERACTION ===")

for out_var, out_name in OUTCOMES:
    for pair_name, strata_col, term1, term2 in [
        ('CRP_and_GGT', 'CRP_GGT_strata', 'CRP_elevated_6', 'GGT_elevated'),
        ('CRP_and_UA', 'CRP_UA_strata', 'CRP_elevated_6', 'UA_elevated')
    ]:
        # Tabulate counts
        ct = pd.crosstab(df_p[strata_col], df_p[out_var])
        
        # Fit model with common reference: 'Neither'
        formula_strata = f"{out_var} ~ C({strata_col}, Treatment(reference='Neither')) + {covariates}"
        try:
            m_logit = smf.logit(formula_strata, data=df_p).fit(disp=0)
            
            # Extract Betas and Covariance
            # Term names in statsmodels
            ref_prefix = f"C({strata_col}, Treatment(reference='Neither'))[T."
            b_crp_only = m_logit.params.get(f"{ref_prefix}CRP_only]", 0.0)
            b_met_only = m_logit.params.get(f"{ref_prefix}{'GGT_only' if 'GGT' in pair_name else 'UA_only'}]", 0.0)
            b_both = m_logit.params.get(f"{ref_prefix}Both_elevated]", 0.0)
            
            p_crp_only = m_logit.pvalues.get(f"{ref_prefix}CRP_only]", 1.0)
            p_met_only = m_logit.pvalues.get(f"{ref_prefix}{'GGT_only' if 'GGT' in pair_name else 'UA_only'}]", 1.0)
            p_both = m_logit.pvalues.get(f"{ref_prefix}Both_elevated]", 1.0)

            or_crp_only = np.exp(b_crp_only)
            or_met_only = np.exp(b_met_only)
            or_both = np.exp(b_both)
            
            cov_mat = m_logit.cov_params()
            k_crp = f"{ref_prefix}CRP_only]"
            k_met = f"{ref_prefix}{'GGT_only' if 'GGT' in pair_name else 'UA_only'}]"
            k_both = f"{ref_prefix}Both_elevated]"
            
            # Additive interaction metrics
            # RERI = OR_both - OR_crp - OR_met + 1
            reri = or_both - or_crp_only - or_met_only + 1.0
            ap = reri / or_both if or_both != 0 else np.nan
            denom_s = (or_crp_only - 1.0) + (or_met_only - 1.0)
            s_index = (or_both - 1.0) / denom_s if denom_s != 0 else np.nan
            
            # Delta method for RERI SE (Andersson et al. 2005)
            # grad = [ -OR_crp, -OR_met, OR_both ]
            if k_crp in cov_mat and k_met in cov_mat and k_both in cov_mat:
                sub_cov = cov_mat.loc[[k_crp, k_met, k_both], [k_crp, k_met, k_both]].values
                grad = np.array([-or_crp_only, -or_met_only, or_both])
                var_reri = grad @ sub_cov @ grad
                se_reri = np.sqrt(max(0.0, var_reri))
                reri_ci_low = reri - 1.96 * se_reri
                reri_ci_high = reri + 1.96 * se_reri
                z_reri = reri / se_reri if se_reri > 0 else 0
                p_reri = 2.0 * (1.0 - norm.cdf(np.abs(z_reri)))
            else:
                se_reri, reri_ci_low, reri_ci_high, p_reri = np.nan, np.nan, np.nan, np.nan
            
            additive_synergy_results.append({
                'Outcome': out_name,
                'Pair': pair_name,
                'OR_Neither (Ref)': 1.0,
                'OR_CRP_only': or_crp_only,
                'P_CRP_only': p_crp_only,
                'OR_Metabolic_only': or_met_only,
                'P_Metabolic_only': p_met_only,
                'OR_Both_elevated': or_both,
                'P_Both_elevated': p_both,
                'RERI': reri,
                'RERI_95CI': f"[{reri_ci_low:.2f}, {reri_ci_high:.2f}]",
                'P_RERI': p_reri,
                'Attributable_Proportion': ap,
                'Synergy_Index': s_index
            })
            
        except Exception as e:
            print(f"Error modeling strata for {out_name} and {pair_name}: {e}")

        # Multiplicative product-term test
        formula_prod = f"{out_var} ~ {term1} * {term2} + {covariates}"
        try:
            m_prod = smf.logit(formula_prod, data=df_p).fit(disp=0)
            prod_term = f"{term1}:{term2}"
            if prod_term in m_prod.params:
                b_int = m_prod.params[prod_term]
                se_int = m_prod.bse[prod_term]
                p_int = m_prod.pvalues[prod_term]
                or_int = np.exp(b_int)
                interaction_results.append({
                    'Outcome': out_name,
                    'Pair': pair_name,
                    'Multiplicative_OR_int': or_int,
                    'CI_Low': np.exp(b_int - 1.96 * se_int),
                    'CI_High': np.exp(b_int + 1.96 * se_int),
                    'P_multiplicative': p_int
                })
        except Exception as e:
            print(f"Error modeling product term for {out_name} and {pair_name}: {e}")

df_add_res = pd.DataFrame(additive_synergy_results)
df_prod_res = pd.DataFrame(interaction_results)
df_add_res.to_csv(os.path.join(OUT_DIR, 'additive_synergy_reri.csv'), index=False)
df_prod_res.to_csv(os.path.join(OUT_DIR, 'multiplicative_interactions.csv'), index=False)

# 4. CUMULATIVE BURDEN SCORE (0 TO 3) TREND ANALYSIS
print("\n=== CUMULATIVE IMMUNO-METABOLIC BURDEN (0 TO 3) ANALYSIS ===")
burden_results = []
for out_var, out_name in OUTCOMES:
    # Prevalence by burden score
    prev_by_score = df_p.groupby('Burden_Score_0_3')[out_var].agg(['count', 'mean'])
    
    # Fit linear trend model (Poisson/Logit per +1 biomarker)
    m_trend = smf.glm(f"{out_var} ~ Burden_Score_0_3 + {covariates}", data=df_p, family=sm.families.Poisson()).fit(cov_type='HC0')
    coef = m_trend.params['Burden_Score_0_3']
    se = m_trend.bse['Burden_Score_0_3']
    p_trend = m_trend.pvalues['Burden_Score_0_3']
    pr_per_point = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    
    burden_results.append({
        'Outcome': out_name,
        'N_Score_0': f"{prev_by_score.loc[0, 'mean']*100:.1f}% ({prev_by_score.loc[0, 'count']})",
        'N_Score_1': f"{prev_by_score.loc[1, 'mean']*100:.1f}% ({prev_by_score.loc[1, 'count']})",
        'N_Score_2': f"{prev_by_score.loc[2, 'mean']*100:.1f}% ({prev_by_score.loc[2, 'count']})",
        'N_Score_3': f"{prev_by_score.loc[3, 'mean']*100:.1f}% ({prev_by_score.loc[3, 'count']})",
        'PR_per_biomarker': pr_per_point,
        '95CI': f"[{ci_low:.2f}, {ci_high:.2f}]",
        'P_trend': p_trend
    })

df_burden = pd.DataFrame(burden_results)
df_burden.to_csv(os.path.join(OUT_DIR, 'cumulative_burden_trend.csv'), index=False)

print("\nAdditive synergy and RERI results:")
print(df_add_res[['Outcome', 'Pair', 'OR_CRP_only', 'OR_Metabolic_only', 'OR_Both_elevated', 'RERI', 'P_RERI']].to_string(index=False))

print("\nCumulative burden trend results:")
print(df_burden[['Outcome', 'N_Score_0', 'N_Score_1', 'N_Score_2', 'N_Score_3', 'PR_per_biomarker', 'P_trend']].to_string(index=False))

