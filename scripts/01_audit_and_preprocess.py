#!/usr/bin/env python3
"""
audit_and_preprocess.py
Phase 1 and Phase 2 data auditing, validation, cleaning, and feature engineering.
"""

import os
import numpy as np
import pandas as pd

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

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Raw Data
df_c_raw = pd.read_csv(os.path.join(DATA_DIR, 'control_group_english.csv'))
df_p_raw = pd.read_csv(os.path.join(DATA_DIR, 'patients_english.csv'))

print("=== PHASE 1: DATA AUDIT & COUNT VERIFICATION ===")
print(f"Controls raw count: {len(df_c_raw)}")
print(f"Patients raw count: {len(df_p_raw)}")

# Audit Sex Counts
c_sex = df_c_raw['Sex'].value_counts()
print(f"Controls Sex: Male={c_sex.get('Male', 0)} ({c_sex.get('Male', 0)/len(df_c_raw):.1%}), Female={c_sex.get('Female', 0)} ({c_sex.get('Female', 0)/len(df_c_raw):.1%})")
p_sex = df_p_raw['Sex'].value_counts()
print(f"Patients Sex: Male={p_sex.get('Male', 0)} ({p_sex.get('Male', 0)/len(df_p_raw):.1%}), Female={p_sex.get('Female', 0)} ({p_sex.get('Female', 0)/len(df_p_raw):.1%})")

# Audit Age < 40
c_age_lt40 = (df_c_raw['Age'] < 40).sum()
p_age_lt40 = (df_p_raw['Age'] < 40).sum()
print(f"Controls Age < 40: {c_age_lt40} ({c_age_lt40/len(df_c_raw):.1%})")
print(f"Patients Age < 40: {p_age_lt40} ({p_age_lt40/len(df_p_raw):.1%})")

# Audit the identical cell count finding (Hepatic disease vs Judicial History)
p_hep = df_p_raw['Hepatic_Diseases'].value_counts()
p_jud = df_p_raw['Judicial_History'].value_counts()
print("\nHepatic disease counts in patients:\n", p_hep)
print("Judicial history counts in patients:\n", p_jud)
overlap_hep_jud = pd.crosstab(df_p_raw['Hepatic_Diseases'], df_p_raw['Judicial_History'])
print("Cross-tabulation of Hepatic Diseases vs Judicial History:\n", overlap_hep_jud)

# 2. FEATURE ENGINEERING (PATIENTS)
df_p = df_p_raw.copy()

# Fix units and calculate standard clinical variables
# Note: In raw data, White_Blood_Cells is in decacells/uL (740 = 7.40 x 10^9 / L)
df_p['WBC_standard'] = df_p['White_Blood_Cells'] / 100.0  # x 10^9/L
df_c_raw['WBC_standard'] = df_c_raw['White_Blood_Cells'] / 100.0

# Calculate BMI
df_p['BMI'] = df_p['Weight_kg'] / ((df_p['Height_cm'] / 100.0) ** 2)
df_p['Obesity'] = (df_p['BMI'] >= 30.0).astype(int)
df_p['Overweight'] = (df_p['BMI'] >= 25.0).astype(int)

# Calculate eGFR (CKD-EPI 2021 equation without race coefficient)
# Creatinine in raw data is in mg/L, convert to mg/dL by dividing by 10
def compute_ckd_epi(ser_cr_mg_l, age, sex):
    scr_mg_dl = ser_cr_mg_l / 10.0
    is_fem = (sex == 'Female').astype(int)
    kappa = np.where(is_fem == 1, 0.7, 0.9)
    alpha = np.where(is_fem == 1,
                     np.where(scr_mg_dl <= 0.7, -0.241, -1.200),
                     np.where(scr_mg_dl <= 0.9, -0.302, -1.200))
    gender_mult = np.where(is_fem == 1, 1.012, 1.0)
    egfr = 142.0 * ((scr_mg_dl / kappa) ** alpha) * (0.9938 ** age) * gender_mult
    return egfr

df_p['eGFR'] = compute_ckd_epi(df_p['Creatinine'], df_p['Age'], df_p['Sex'])
df_c = df_c_raw.copy()
df_c['eGFR'] = compute_ckd_epi(df_c['Creatinine'], df_c['Age'], df_c['Sex'])
df_p['CKD_stage3_plus'] = (df_p['eGFR'] < 60.0).astype(int)
df_c['CKD_stage3_plus'] = (df_c['eGFR'] < 60.0).astype(int)

# Binarize clinical binary variables
binary_cols_p = [
    'Depressive_Disorder', 'Suicide_Attempt', 'Judicial_History', 'Cardiovascular_Diseases',
    'Hepatic_Diseases', 'Diabetes', 'Hypertension', 'OCD', 'Anxiety_Disorder',
    'Family_Psychiatric_History', 'History_of_Violence'
]
for col in binary_cols_p:
    df_p[f'{col}_bin'] = (df_p[col] == 'Yes').astype(int)

# Sociodemographic and Lifestyle
df_p['Is_Male'] = (df_p['Sex'] == 'Male').astype(int)
df_p['Smoker'] = (df_p['Smoking_Status'] == 'Smoker').astype(int)
df_p['Alcohol'] = (df_p['Alcohol_Consumption'] == 'Yes').astype(int)
df_p['Drugs'] = (df_p['Drugs_and_Psychotropics'] == 'Yes').astype(int)
df_p['Consanguineous'] = (df_p['Consanguineous_Marriage'] == 'Yes').astype(int)
df_p['Urban'] = (df_p['Residence_Area'] == 'Urban').astype(int)
df_p['Unemployed'] = (df_p['Profession'] == 'Unemployed').astype(int)

# Controls processing
df_c['Is_Male'] = (df_c['Sex'] == 'Male').astype(int)
df_c['Smoker'] = (df_c['Smoking_Status'] == 'Smoker').astype(int)
df_c['Alcohol'] = (df_c['Alcohol_Consumption'] == 'Yes').astype(int)
df_c['Drugs'] = (df_c['Drugs'] == 'Yes').astype(int)
df_c['Consanguineous'] = (df_c['Consanguineous_Marriage'] == 'Yes').astype(int)
df_c['Urban'] = (df_c['Residence_Area'] == 'Urban').astype(int)
df_c['Unemployed'] = (df_c['Profession'] == 'Unemployed').astype(int)

# Medication feature engineering (Phase 2)
# Treat1 - Treat4 contain psychotropics
all_treats = df_p[['Treatment_1', 'Treatment_2', 'Treatment_3', 'Treatment_4', 'Treatment_5', 'Treatment_6']].fillna('')

# 1. Antipsychotics with High Metabolic Liability: Olanzapine
df_p['Med_Olanzapine'] = all_treats.apply(lambda r: int('Olanzapine' in r.values), axis=1)

# 2. Moderate Liability: Risperidone
df_p['Med_Risperidone'] = all_treats.apply(lambda r: int('Risperidon' in r.values), axis=1)

# 3. Low Liability: Aripiprazole (Abilify)
df_p['Med_Aripiprazole'] = all_treats.apply(lambda r: int('Abilify' in r.values), axis=1)

# 4. First generation / typicals: Nozinan, Largactil, Haldol, Modicate
df_p['Med_Typical_Only'] = all_treats.apply(
    lambda r: int(
        any(m in r.values for m in ['Nozinan', 'Largactil', 'Haldol', 'Modicate[injectable]']) and
        not any(m in r.values for m in ['Risperidon', 'Olanzapine', 'Abilify', 'Solian'])
    ), axis=1
)

# 5. Hepatic Enzyme Inducers (affecting GGT): Carbamazepine (Tegretol), Phenobarbital (Gardinal)
df_p['Med_Enzyme_Inducer'] = all_treats.apply(
    lambda r: int(any(m in r.values for m in ['Tegretol', 'Gardinal'])), axis=1
)

# 6. Antihypertensive / Thiazide Diuretic: Coaprovel (Irbesartan + Hydrochlorothiazide, raises Uric Acid!)
df_p['Med_Coaprovel_Diuretic'] = all_treats.apply(lambda r: int('Coaprovel' in r.values), axis=1)

# 7. Antidiabetic: Glucophage (Metformin)
df_p['Med_Metformin'] = all_treats.apply(lambda r: int('Glucophage' in r.values), axis=1)

# Polypharmacy count of psychotropics (Treatment 1 to 4)
def count_psychotropics(r):
    count = 0
    for val in r:
        if val != '' and val not in ['Glucophage', 'Coaprovel']:
            count += 1
    return count
df_p['Psychotropic_Count'] = all_treats[['Treatment_1', 'Treatment_2', 'Treatment_3', 'Treatment_4']].apply(count_psychotropics, axis=1)

# Continuous Biomarker Transformations
# CRP, GGT, Uric Acid
df_p['log_CRP'] = np.log(df_p['CRP'] + 0.1)
df_p['log_GGT'] = np.log(df_p['GGT'])
df_p['log_UA'] = np.log(df_p['Uric_Acid'])

df_c['log_CRP'] = np.log(df_c['CRP'] + 0.1)
df_c['log_GGT'] = np.log(df_c['GGT'])
df_c['log_UA'] = np.log(df_c['Uric_Acid'])

# Standardized Z-scores within cohort (per 1-SD increase)
for m in ['log_CRP', 'log_GGT', 'log_UA', 'CRP', 'GGT', 'Uric_Acid']:
    df_p[f'{m}_z'] = (df_p[m] - df_p[m].mean()) / df_p[m].std()

# Clinical Biomarker Cutoffs
# CRP:
# Elevated > 6.0 mg/L (as in original cohort: 95 positive, 105 negative)
# Clinical low-grade vs acute: < 3 mg/L normal/low, 3-10 mg/L low-grade, > 10 mg/L acute systemic inflammation
df_p['CRP_elevated_6'] = (df_p['CRP'] > 6.0).astype(int)
df_p['CRP_lowgrade_3_10'] = ((df_p['CRP'] >= 3.0) & (df_p['CRP'] <= 10.0)).astype(int)
df_p['CRP_acute_gt10'] = (df_p['CRP'] > 10.0).astype(int)

df_c['CRP_elevated_6'] = (df_c['CRP'] > 6.0).astype(int)
df_c['CRP_acute_gt10'] = (df_c['CRP'] > 10.0).astype(int)

# GGT:
# Clinical elevated >= 49 U/L
df_p['GGT_elevated'] = (df_p['GGT'] >= 49.0).astype(int)
df_c['GGT_elevated'] = (df_c['GGT'] >= 49.0).astype(int)

# Uric Acid: Sex-specific thresholds (Male > 60 mg/L [6.0 mg/dL], Female > 54 mg/L [5.4 mg/dL])
df_p['UA_elevated'] = np.where(
    df_p['Sex'] == 'Male',
    (df_p['Uric_Acid'] > 60.0).astype(int),
    (df_p['Uric_Acid'] > 54.0).astype(int)
)
df_c['UA_elevated'] = np.where(
    df_c['Sex'] == 'Male',
    (df_c['Uric_Acid'] > 60.0).astype(int),
    (df_c['Uric_Acid'] > 54.0).astype(int)
)

print(f"\nElevated Biomarker Counts in Patients (N=200):")
print(f"CRP >= 6 mg/L: {df_p['CRP_elevated_6'].sum()} ({df_p['CRP_elevated_6'].mean():.1%})")
print(f"CRP > 10 mg/L (acute): {df_p['CRP_acute_gt10'].sum()} ({df_p['CRP_acute_gt10'].mean():.1%})")
print(f"GGT >= 49 U/L: {df_p['GGT_elevated'].sum()} ({df_p['GGT_elevated'].mean():.1%})")
print(f"UA elevated (sex-specific): {df_p['UA_elevated'].sum()} ({df_p['UA_elevated'].mean():.1%})")

# Combined Case-Control Indicator
df_p['Group'] = 'Patient'
df_c['Group'] = 'Control'

# Save cleaned datasets
df_p.to_csv(os.path.join(OUT_DIR, 'cleaned_patients.csv'), index=False)
df_c.to_csv(os.path.join(OUT_DIR, 'cleaned_controls.csv'), index=False)

print("\nAudit and preprocessing successfully completed!")
