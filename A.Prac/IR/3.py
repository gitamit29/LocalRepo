# 3_fixed.py
import sys
import numpy as np
import pandas as pd

# ----------------- Safe imports for pgmpy -----------------
try:
    from pgmpy.models import BayesianModel
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    print("Imported pgmpy classes from pgmpy.models")
except Exception as e:
    # helpful diagnostic and re-raise
    print("Error importing pgmpy classes:", e)
    print("Check that pgmpy is installed in the same interpreter. Try:")
    print("    python -m pip install pgmpy")
    raise

# ----------------- Load data -----------------
data_path = r"C:\Users\amrut\OneDrive\Desktop\IR\heart.csv"
df = pd.read_csv(data_path, na_values=["?", "NA", ""])

print("\nRaw columns:", df.columns.tolist())
print("\nFirst rows:\n", df.head())

# ----------------- Rename/check target column -----------------
# Many heart datasets use 'target' as the label. If your file uses 'target', rename:
if 'target' in df.columns and 'heartdisease' not in df.columns:
    df = df.rename(columns={'target': 'heartdisease'})

expected_cols = ['age','trestbps','fbs','sex','exang','heartdisease','restecg','thalach','chol']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("\nWARNING: These expected columns are missing from the CSV:", missing)
    print("You should either rename your columns or adjust the model accordingly.")
    # continue but may fail later if required columns absent

# ----------------- Handle missing values -----------------
print(f"\nOriginal rows: {len(df)}")
df = df.dropna(subset=[c for c in expected_cols if c in df.columns])  # drop rows missing any expected column
print(f"Rows after dropping missing: {len(df)}")

# ----------------- Discretize continuous columns -----------------
# CPD learning in pgmpy expects discrete variables. We'll discretize age, trestbps, chol, thalach.
# Adjust number of bins as needed.
def discretize(series, bins=3, labels=None, prefix=None):
    if labels is None:
        labels = [f"{prefix}_bin{i}" for i in range(bins)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

# Only discretize if column present and numeric
if 'age' in df.columns:
    df['age_disc'] = discretize(df['age'], bins=4, prefix='age')
if 'trestbps' in df.columns:
    df['trestbps_disc'] = discretize(df['trestbps'], bins=3, prefix='trestbps')
if 'chol' in df.columns:
    df['chol_disc'] = discretize(df['chol'], bins=3, prefix='chol')
if 'thalach' in df.columns:
    df['thalach_disc'] = discretize(df['thalach'], bins=3, prefix='thalach')

# For binary or categorical columns like fbs, sex, exang, restecg, ensure they are strings (discrete)
for col in ['fbs','sex','exang','restecg','heartdisease']:
    if col in df.columns:
        df[col] = df[col].astype(str)

print("\nColumns after discretization:", [c for c in df.columns if 'disc' in c or c in ['fbs','sex','exang','restecg','heartdisease']])
print(df[[c for c in df.columns if 'disc' in c][:5]].head())

# ----------------- Build model using discretized variables -----------------
# Update model edges to use discretized names where required
model = BayesianModel([
    ('age_disc','trestbps_disc'),
    ('age_disc','fbs'),
    ('sex','trestbps_disc'),
    ('exang','trestbps_disc'),
    ('trestbps_disc','heartdisease'),
    ('fbs','heartdisease'),
    ('heartdisease','restecg'),
    ('heartdisease','thalach_disc'),
    ('heartdisease','chol_disc')
])

# Prepare dataframe for pgmpy: only include the variables present in the model
model_vars = set(sum(([u,v] for u,v in model.edges()), []))
available_vars = [v for v in model_vars if v in df.columns]
data_for_fit = df[available_vars].copy()

print("\nVariables used for fitting:", available_vars)
print("Sample of data_for_fit:\n", data_for_fit.head())

# ----------------- Fit CPDs -----------------
print("\nLearning CPDs using MaximumLikelihoodEstimator...")
model.fit(data_for_fit, estimator=MaximumLikelihoodEstimator)

# ----------------- Inference -----------------
infer = VariableElimination(model)

# Example queries: supply evidence using discretized labels
# Show available states for a variable (inspect its CPD)
try:
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is not None:
            print(f"\nCPD for {node}:\n{cpd}")
except Exception as e:
    print("Warning: could not print CPDs:", e)

# Query example 1: Age -> choose a discrete bin label that exists (inspect df['age_disc'].cat.categories)
if 'age_disc' in df.columns:
    print("\nAge bins:", df['age_disc'].cat.categories)
    # pick one example label from categories
    age_label = df['age_disc'].cat.categories[0]
    print(f"\n1) Probability of heartdisease given age_disc = {age_label}")
    q = infer.query(variables=['heartdisease'], evidence={'age_disc': age_label})
    print(q)
else:
    print("\nSkipping age query because 'age_disc' not present.")

# Query example 2: cholesterol -> use discretized label
if 'chol_disc' in df.columns:
    print("\nChol bins:", df['chol_disc'].cat.categories)
    chol_label = df['chol_disc'].cat.categories[-1]
    print(f"\n2) Probability of heartdisease given chol_disc = {chol_label}")
    q = infer.query(variables=['heartdisease'], evidence={'chol_disc': chol_label})
    print(q)
else:
    print("\nSkipping chol query because 'chol_disc' not present.")
