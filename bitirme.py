# bitirme.py

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


# =========================
# MODEL SINIFLARI
# =========================
class ANN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class ANN2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        return self.fc3(x)


class ANN3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.fc3(x))
        return self.fc4(x)


# =========================
# TRAIN SADECE BURADA ÇALIŞIR
# =========================
if __name__ == "__main__":

    print("Veriler okunuyor...")

    files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()

    # balance
    target = 10000
    dfs = []
    for label in df["Label"].unique():
        d = df[df["Label"] == label]
        d = resample(d, replace=len(d) < target, n_samples=target, random_state=42)
        dfs.append(d)

    df = pd.concat(dfs)

    # encode
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df.drop("Label", axis=1).values
    y = df["Label"].values

    # scaler → KAYDET
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # sklearn
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{name}.joblib")
        print(name, "kaydedildi")

    # torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(model, name):
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(3):
            opt.zero_grad()
            out = model(X_train.to(device))
            loss = loss_fn(out, y_train.to(device))
            loss.backward()
            opt.step()

        torch.save(model.state_dict(), f"{name}.pth")
        print(name, "kaydedildi")

    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    train(ANN1(input_size, 64, num_classes), "ANN1")
    train(ANN2(input_size, 64, num_classes), "ANN2_Dropout")
    train(ANN3(input_size, 64, num_classes), "ANN3_BatchNormDeep")