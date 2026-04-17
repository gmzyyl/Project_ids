# OFFLINE IDS - FULL + FIXED + PROFESSIONAL

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# DL MODELS
# =========================
class ANN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ANN2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ANN3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# MAIN TRAIN
# =========================
if __name__ == "__main__":

    print("Veriler okunuyor...")

    files = [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]

    df = pd.concat([pd.read_csv(f, encoding="latin1") for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()

    print("Original:", df.shape)
    print(df["Label"].value_counts())

    # =========================
    # BALANCE
    # =========================
    target = 10000
    balanced = []

    for label in df["Label"].unique():
        d = df[df["Label"] == label]
        d = resample(d, replace=len(d) < target, n_samples=target, random_state=42)
        balanced.append(d)

    df = pd.concat(balanced)
    print("Balanced:", df.shape)

    # =========================
    # ENCODE
    # =========================
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    joblib.dump(le, "label_encoder.pkl")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 🔥 FIX 1: featureleri NET sabitle
    X_df = df.drop("Label", axis=1)

    feature_names = X_df.columns
    joblib.dump(feature_names, "feature_names.pkl")

    X = X_df.values
    y = df["Label"].values

    # =========================
    # SCALER
    # =========================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # =========================
    # ML MODELS
    # =========================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=50),
        "SVM": SVC(kernel="linear"),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    }

    print("\n--- ML TRAIN ---")

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

print(f"\n{name}")

# =========================
# BASIC METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro")
rec_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

prec_weighted = precision_score(y_test, y_pred, average="weighted")
rec_weighted = recall_score(y_test, y_pred, average="weighted")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", acc)

print("Precision (macro):", prec_macro)
print("Recall (macro):", rec_macro)
print("F1 (macro):", f1_macro)

print("Precision (weighted):", prec_weighted)
print("Recall (weighted):", rec_weighted)
print("F1 (weighted):", f1_weighted)

# =========================
# IDS CRITICAL ANALYSIS
# =========================
print("\n--- CLASS REPORT (IDS DETAIL) ---")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX (OPTIONAL DEBUG)
# =========================
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

    # =========================
    # DL
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    def train(model, name):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        epochs = 3
        batch_size = 512

        for epoch in range(epochs):
            perm = torch.randperm(X_train_t.size(0))

            for i in range(0, X_train_t.size(0), batch_size):
                idx = perm[i:i+batch_size]

                x_batch = X_train_t[idx].to(device)
                y_batch = y_train_t[idx].to(device)

                optimizer.zero_grad()
                out = model(x_batch)
                loss = loss_fn(out, y_batch)
                loss.backward()
                optimizer.step()

            print(f"{name} Epoch {epoch+1}/{epochs}")

        torch.save(model.state_dict(), f"{name}.pth")

    print("\n--- DL TRAIN ---")

    train(ANN1(input_size, 64, num_classes), "ANN1")
    train(ANN2(input_size, 64, num_classes), "ANN2_Dropout")
    train(ANN3(input_size, 64, num_classes), "ANN3_BatchNormDeep")

    print("\n IDS READY")
