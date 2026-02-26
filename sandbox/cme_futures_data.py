import pandas as pd

df = pd.read_csv("/Users/adithsrinivasan/Documents/GitHub/qts-cross-asset-fx/sandbox/glbx-mdp3-20260126.tbbo.csv", parse_dates=["ts_event"])

# Convert to pandas datetime (UTC-aware, keeps nanoseconds)
df["ts_utc"] = pd.to_datetime(df["ts_event"], utc=True)

# Convert to Eastern Time (handles DST automatically)
df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")

print(df["ts_et"])

