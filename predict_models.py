import pandas as pd
import joblib
import numpy as np
from glob import glob
from collections import Counter

# =========================
# 1) CSV OKU
# =========================
all_files = glob(r"C:\Users\gamze\OneDrive\Masaüstü\CSVFILES\*.csv")

df = pd.concat([pd.read_csv(f, encoding="latin1") for f in all_files], ignore_index=True)
df.columns = df.columns.str.strip()

# =========================
# 2) FEATURE FIX (EN ÖNEMLİ KISIM)
# =========================
feature_names = joblib.load("feature_names.pkl")

# eksik/fazla kolonları düzelt
df = df.reindex(columns=feature_names, fill_value=0)

X = df.values
print("Feature:", X.shape)

# =========================
# 3) SCALER
# =========================
scaler = joblib.load("scaler.pkl")
X = scaler.transform(X)

# =========================
# 4) MODEL
# =========================
model = joblib.load("RandomForest.joblib")
pred = model.predict(X)

# =========================
# 5) LABEL DECODE
# =========================
le = joblib.load("label_encoder.pkl")
labels = le.inverse_transform(pred)

# =========================
# 6) SAYIM
# =========================
counter = Counter(labels)

print("\n--- IDS ANALİZ SONUCU ---")

for label, count in counter.items():
    if label == "BENIGN":
        print(f"{label}: {count}")
    else:
        print(f" {label}: {count}")

# =========================
# 7) GENEL DURUM
# =========================
attack_total = sum(count for label, count in counter.items() if label != "BENIGN")
normal_total = counter.get("BENIGN", 0)

print("\n--- GENEL DURUM ---")

if attack_total > normal_total:
    print(" SALDIRI VAR")
else:
    print(" NORMAL TRAFİK")
