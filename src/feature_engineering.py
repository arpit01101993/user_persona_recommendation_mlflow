import pandas as pd

df_clicks = pd.read_csv("data/raw/clickstream.csv")
df_profile = pd.read_csv("data/raw/user_profile.csv")

df = df_clicks.merge(df_profile, on="user_id")
df["session_length"] = df.groupby("session_id")["timestamp"].transform(lambda x: x.max() - x.min())

df.to_parquet("data/processed/features.parquet")
