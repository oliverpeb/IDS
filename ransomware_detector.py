import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
import time
import hashlib
import json
from getpass import getpass  # Secure input for prod; use input() for testing

# -----------------------------
# Config
# -----------------------------
DATA_FILE = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'   # <-- your CIC-IDS2017 CSV
MODEL_FILE = 'ransomware_model.joblib'
SCALER_FILE = 'scaler.joblib'
LOG_FILE = 'cumulative_alerts.json'
CHUNK_SIZE = 200
CONTAMINATION = 0.01  # lower contamination to reduce FPs

# Features used by the model
features_cols = ['Time_numeric', 'Netflow_Bytes', 'Port']

# -----------------------------
# Load CIC-IDS2017 dataset (robust)
# -----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False, encoding_errors='ignore')

# 1) Strip leading/trailing spaces from headers (CIC often has them)
df.columns = df.columns.str.strip()

# 2) Helper to pick first matching column name
def pick(*candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None

# 3) Identify source columns (covering common variants)
time_src = pick('Timestamp', 'Time', 'StartTime', 'Start Time', 'Flow Start')
port_src = pick('Destination Port', 'Dst Port', 'Dst Port Number', 'Destination Port Number')
flow_bytes_rate_src = pick('Flow Bytes/s', 'Flow Bytes per s', 'Flow Bytes Per Second')
fwd_bytes_src = pick('Total Length of Fwd Packets', 'Total Fwd Bytes', 'Fwd Bytes', 'Fwd Bytes/s')
bwd_bytes_src = pick('Total Length of Bwd Packets', 'Total Bwd Bytes', 'Bwd Bytes', 'Bwd Bytes/s')
ip_src = pick('Destination IP', 'Dst IP', 'Destination Address', 'Dst Addr', 'Destination IP Address')
label_src = pick('Label', 'Attack', 'Category')

# 4) Build/rename required columns
# Time (create synthetic if not present)
if time_src:
    df = df.rename(columns={time_src: 'Time'})
else:
    # Create a synthetic monotonic timestamp so the pipeline works
    start = pd.Timestamp('2025-01-01 00:00:00')
    df['Time'] = start + pd.to_timedelta(np.arange(len(df)), unit='s')
    print("(!) No time column found. Using synthetic Time from row index.")

# Port
if not port_src:
    raise KeyError(f"Could not find a destination port column. Seen columns (first 25): {list(df.columns)[:25]}")
df = df.rename(columns={port_src: 'Port'})

# Netflow_Bytes: prefer Flow Bytes/s; else approximate via total bytes
if flow_bytes_rate_src:
    df = df.rename(columns={flow_bytes_rate_src: 'Netflow_Bytes'})
else:
    if fwd_bytes_src and bwd_bytes_src:
        df['Netflow_Bytes'] = pd.to_numeric(df[fwd_bytes_src], errors='coerce') + \
                               pd.to_numeric(df[bwd_bytes_src], errors='coerce')
        print("(!) 'Flow Bytes/s' missing. Using Total Fwd + Total Bwd bytes as Netflow_Bytes.")
    else:
        raise KeyError(
            "Could not find 'Flow Bytes/s' nor total fwd/bwd byte columns to build Netflow_Bytes.\n"
            f"Seen columns (first 25): {list(df.columns)[:25]}"
        )

# Optional IP column for de-hash
if ip_src and 'IPaddress' not in df.columns:
    df = df.rename(columns={ip_src: 'IPaddress'})

# Labels -> Prediction (1=Benign, 0=Malicious)
if label_src and 'Prediction' not in df.columns:
    def to_pred(x):
        s = str(x).strip().upper()
        return 1 if s == 'BENIGN' else 0
    df['Prediction'] = df[label_src].apply(to_pred)

print(f"Loaded {len(df)} rows from CIC-IDS2017 subset: {DATA_FILE}")

# Optional: subsample for speed on first runs (comment out if you want full data)
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    print("Subsampled to 50,000 rows for performance.")

# -----------------------------
# Ensure derived/expected columns exist
# -----------------------------
# Parse time and create numeric epoch seconds
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Time_numeric'] = (df['Time'].astype('int64') // 10**9)

# Coerce numeric fields and clean infinities
df['Netflow_Bytes'] = pd.to_numeric(df['Netflow_Bytes'], errors='coerce')
df['Port'] = pd.to_numeric(df['Port'], errors='coerce')
for col in ['Netflow_Bytes', 'Port']:
    df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

# Proxy entropy feature (monotonic transform)
df['file_entropy'] = np.log(df['Netflow_Bytes'].fillna(0) + 1)

# Expose destination IP as 'ip' (if present)
if 'IPaddress' in df.columns and 'ip' not in df.columns:
    df['ip'] = df['IPaddress']

# Drop rows missing required model features
needed_cols = features_cols + ['file_entropy']
df = df.dropna(subset=needed_cols).reset_index(drop=True)

# -----------------------------
# Hashing helpers (demo salt)
# -----------------------------
SALT = b"your_project_salt_2025"

def hash_field(field):
    return hashlib.sha256(SALT + str(field).encode()).hexdigest()[:8]

def de_hash(hashed_value, original_candidates):
    for orig in original_candidates:
        if hash_field(orig) == hashed_value:
            return orig
    return "Access Denied - No Match"

# -----------------------------
# Load or initialize cumulative log
# -----------------------------
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        cumulative_log = json.load(f)
    total_anomalies = cumulative_log.get('total_anomalies', 0)
    print(f"Loaded prior state: {total_anomalies} anomalies tracked so far.")
else:
    cumulative_log = {'total_anomalies': 0, 'alerts': []}
    total_anomalies = 0

# -----------------------------
# Train / load model + scaler
# -----------------------------
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Loaded existing model—no retraining needed.")
    start_idx = int(0.7 * len(df))  # treat last 30% as 'incoming'
else:
    historical_data = df.iloc[:int(0.7 * len(df))].copy()
    scaler = StandardScaler()
    scaled_historical = scaler.fit_transform(historical_data[features_cols + ['file_entropy']])
    model = IsolationForest(contamination=CONTAMINATION, random_state=42)
    model.fit(scaled_historical)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Trained and saved new model.")
    start_idx = int(0.7 * len(df))

# -----------------------------
# Streaming / chunked scoring
# -----------------------------
new_data = df.iloc[start_idx:].copy()
run_num = (start_idx // CHUNK_SIZE) + 1
new_anomalies = []

for i in range(0, len(new_data), CHUNK_SIZE):
    chunk = new_data.iloc[i:i + CHUNK_SIZE].copy()
    if len(chunk) == 0:
        break

    pred_features = chunk[features_cols + ['file_entropy']]
    scaled_chunk = scaler.transform(pred_features)
    predictions = model.predict(scaled_chunk)  # -1 anomaly, 1 normal

    chunk.loc[:, 'anomaly'] = predictions
    anomalies_chunk = chunk[chunk['anomaly'] == -1].copy()

    num_chunk_anoms = len(anomalies_chunk)
    total_anomalies += num_chunk_anoms

    if num_chunk_anoms > 0:
        # Put hashed IDs on anomalies (used later for de-hash)
        anomalies_chunk.loc[:, 'Time_hashed'] = anomalies_chunk['Time'].apply(hash_field)
        if 'ip' in anomalies_chunk.columns:
            anomalies_chunk.loc[:, 'ip_hashed'] = anomalies_chunk['ip'].apply(hash_field)
        elif 'IPaddress' in anomalies_chunk.columns:
            anomalies_chunk.loc[:, 'ip_hashed'] = anomalies_chunk['IPaddress'].apply(hash_field)

        # Simple reasoning flags (tune thresholds to your data scale)
        def get_reasoning(row):
            reasons = []
            if row['file_entropy'] > np.log(1e5):
                reasons.append("high entropy > log(1e5)")
            if row['Port'] > 10000:
                reasons.append("high dest port >10000")
            if row['Netflow_Bytes'] > 1e6:
                reasons.append("very high throughput >1e6 B/s")
            return "; ".join(reasons) if reasons else "outlier pattern"

        anomalies_chunk.loc[:, 'reasoning'] = anomalies_chunk.apply(get_reasoning, axis=1)

        # Append alerts (hashed) to cumulative log
        for _, row in anomalies_chunk.iterrows():
            alert = {
                'Time_hashed': row['Time_hashed'],
                'ip_hashed': row.get('ip_hashed', hash_field(row.get('ip', row.get('IPaddress', 'n/a')))),
                'Netflow_Bytes': float(row['Netflow_Bytes']),
                'Port': float(row['Port']),
                'file_entropy': float(row['file_entropy']),
                'reasoning': row['reasoning'],
                'run_num': run_num,
                'Prediction': int(row.get('Prediction')) if 'Prediction' in row else 'N/A'
            }
            cumulative_log['alerts'].append(alert)

        new_anomalies.append(anomalies_chunk)
        print(f"Run {run_num} (Chunk {start_idx + i}-{start_idx + i + len(chunk)}): Detected {num_chunk_anoms} new anomalies.")

        # Evaluate with labels if available
        if 'Prediction' in chunk.columns:
            y_true = chunk['Prediction'].fillna(1).astype(int).values
            y_pred = np.array([0 if p == -1 else 1 for p in predictions])
            print(classification_report(y_true, y_pred, target_names=['Malicious', 'Benign'], zero_division=0))

    run_num += 1
    time.sleep(0.05)

# -----------------------------
# Save audit log for new anomalies
# -----------------------------
if new_anomalies:
    all_new_anoms = pd.concat(new_anomalies, ignore_index=True)

    cols_to_save = [
        'Time', 'Time_numeric', 'Netflow_Bytes', 'Port', 'file_entropy',
        'Time_hashed', 'ip_hashed', 'reasoning'
    ]
    if 'Prediction' in all_new_anoms.columns:
        cols_to_save.append('Prediction')
    cols_to_save = [c for c in cols_to_save if c in all_new_anoms.columns]

    all_new_anoms.to_csv('gdpr_audit_log.csv', index=False, columns=cols_to_save)
    print("GDPR-compliant audit log saved with anonymized data.")

    # De-hash option for follow-up (interactive)
    try:
        investigate = input("Enter 'investigate' to de-hash anomalies? (y/n): ").strip().lower()
    except EOFError:
        investigate = 'n'
        print("(Skipped interactive prompt—non-terminal mode detected.)")

    if investigate == 'y':
        try:
            access_code = getpass("Enter access code: ")
        except EOFError:
            access_code = ''
        if access_code == "KEA2025":  # demo
            time_candidates = df['Time'].dropna().astype(str).tolist()
            ip_source = 'ip' if 'ip' in df.columns else ('IPaddress' if 'IPaddress' in df.columns else None)
            ip_candidates = df[ip_source].dropna().astype(str).tolist() if ip_source else []

            all_new_anoms.loc[:, 'Time_dehashed'] = all_new_anoms['Time_hashed'].apply(lambda h: de_hash(h, time_candidates))
            if 'ip_hashed' in all_new_anoms.columns and ip_candidates:
                all_new_anoms.loc[:, 'ip_dehashed'] = all_new_anoms['ip_hashed'].apply(lambda h: de_hash(h, ip_candidates))

            print("\nDe-hashed for investigation:")
            cols = ['Time_dehashed', 'ip_dehashed', 'reasoning']
            cols = [c for c in cols if c in all_new_anoms.columns]
            print(all_new_anoms[cols].to_string(index=False))
        else:
            print("Access denied—compliance enforced.")

# -----------------------------
# Save updated cumulative log
# -----------------------------
cumulative_log['total_anomalies'] = total_anomalies
with open(LOG_FILE, 'w') as f:
    json.dump(cumulative_log, f, default=str)
print(f"\nSession total: {total_anomalies} cumulative suspicious possible anomalies.")
print(f"State saved—next run will continue from here. Full log in {LOG_FILE}.")
