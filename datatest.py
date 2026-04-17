import pandas as pd
import glob
import numpy as np

all_files = glob.glob("*.csv")

df_list = []

for file in all_files:
    print("Okunuyor:", file)

    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()

    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# 🔥 LABEL AYIR
labels = df['Label']

# 🔥 SADECE FEATURELAR NUMERİK
features = df.drop('Label', axis=1)
features = features.apply(pd.to_numeric, errors='coerce')

# temizle
features = features.replace([np.inf, -np.inf], 0)
features = features.fillna(0)

# tekrar birleştir
df_clean = pd.concat([features, labels], axis=1)

print(df_clean['Label'].value_counts())
