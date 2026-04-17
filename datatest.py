import pandas as pd
import glob

all_files = glob.glob("*.csv")
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

print("Shape:", df.shape)
print(df['Label'].value_counts())

# KAYDET
df.to_csv("dataset_full.csv", index=False)