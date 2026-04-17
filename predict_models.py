import pandas as pd
import joblib
import torch
import numpy as np
from glob import glob
from bitirme import ANN1, ANN2, ANN3

# =========================
# 1) WIRESHARK CSV OKU
# =========================
all_files = glob(r"C:\Users\gamze\OneDrive\Masaüstü\CSVFILES\*.csv")

df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df.columns = df.columns.str.strip()

# sadece sayısal
df = df.select_dtypes(include=[np.number])
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df.values

print("Feature:", X.shape)

# =========================
# 2) SCALER
# =========================
scaler = joblib.load("scaler.pkl")
X = scaler.transform(X)

# =========================
# 3) SKLEARN TAHMİN
# =========================
attack_count = 0
normal_count = 0

for name in [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
    "KNN",
    "AdaBoost",
    "GradientBoosting"
]:

    model = joblib.load(f"{name}.joblib")
    pred = model.predict(X)

    attack_count += (pred != 0).sum()
    normal_count += (pred == 0).sum()

# =========================
# 4) SONUÇ
# =========================
print("\n--- SONUÇ ---")
print("Normal:", normal_count)
print("Attack:", attack_count)

if attack_count > normal_count:
    print("🚨 SALDIRI VAR")
else:
    print("🟢 NORMAL TRAFİK")