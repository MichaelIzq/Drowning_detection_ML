import pandas as pd

# Full file path, corrected
RAW = r"E:\thesis data\LOGS2\LOG005_capturestroke.CSV"
# Skip the metadata line
df = pd.read_csv(RAW, skiprows=1)

# Drop empty columns from extra commas
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Keep only the accel columns
df = df[['timestamp_ms', 'ax_g', 'ay_g', 'az_g']].copy()



df.to_csv("E:/documents/Thesis code/LOG005_capturestroke_clean", index=False)
print("Cleaned CSV saved!")
