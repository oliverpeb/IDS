# detector.py
import os
import json
import time
import hashlib
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ========= Config (base) =========
SCALER_FILE = 'scaler.joblib'
LOG_FILE = 'cumulative_alerts.json'
AUDIT_FILE = 'gdpr_audit_log.csv'

CHUNK_SIZE = 200
DEFAULT_CONTAM = 0.02
FEATURES = ['Time_numeric', 'Netflow_Bytes', 'Port']
SALT = b"your_project_salt_2025"  # demo salt


# ---------- Helpers ----------
def _model_file_for(contam: float) -> str:
    return f"ransomware_model_c{contam:.4f}.joblib"

def _hash_field(field) -> str:
    return hashlib.sha256(SALT + str(field).encode()).hexdigest()[:8]

def _pick(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _load_dataset_robust(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_FILE not found: {path}")

    df = pd.read_csv(path, low_memory=False, encoding_errors='ignore')
    df.columns = df.columns.str.strip()

    time_src = _pick(df, 'Timestamp', 'Time', 'StartTime', 'Start Time', 'Flow Start')
    port_src = _pick(df, 'Destination Port', 'Dst Port', 'Dst Port Number', 'Destination Port Number')
    flow_bytes_rate_src = _pick(df, 'Flow Bytes/s', 'Flow Bytes per s', 'Flow Bytes Per Second')
    fwd_bytes_src = _pick(df, 'Total Length of Fwd Packets', 'Total Fwd Bytes', 'Fwd Bytes', 'Fwd Bytes/s')
    bwd_bytes_src = _pick(df, 'Total Length of Bwd Packets', 'Total Bwd Bytes', 'Bwd Bytes', 'Bwd Bytes/s')
    ip_src = _pick(df, 'Destination IP', 'Dst IP', 'Destination Address', 'Dst Addr', 'Destination IP Address')
    label_src = _pick(df, 'Label', 'Attack', 'Category')

    if time_src:
        df = df.rename(columns={time_src: 'Time'})
    else:
        start = pd.Timestamp('2025-01-01 00:00:00')
        df['Time'] = start + pd.to_timedelta(np.arange(len(df)), unit='s')
        print("(!) No time column found. Using synthetic Time from row index.")

    if not port_src:
        raise KeyError(f"Could not find destination port column. Columns: {list(df.columns)[:25]}")
    df = df.rename(columns={port_src: 'Port'})

    if flow_bytes_rate_src:
        df = df.rename(columns={flow_bytes_rate_src: 'Netflow_Bytes'})
    else:
        if fwd_bytes_src and bwd_bytes_src:
            df['Netflow_Bytes'] = pd.to_numeric(df[fwd_bytes_src], errors='coerce') + \
                                  pd.to_numeric(df[bwd_bytes_src], errors='coerce')
            print("(!) 'Flow Bytes/s' missing. Using Total Fwd + Total Bwd bytes for Netflow_Bytes.")
        else:
            raise KeyError(
                "Missing 'Flow Bytes/s' and total fwd/bwd byte columns; can't build Netflow_Bytes.\n"
                f"Columns: {list(df.columns)[:25]}"
            )

    if ip_src and 'IPaddress' not in df.columns:
        df = df.rename(columns={ip_src: 'IPaddress'})

    if label_src and 'Prediction' not in df.columns:
        df['Prediction'] = df[label_src].apply(lambda x: 1 if str(x).strip().upper() == 'BENIGN' else 0)

    return df

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Time_numeric'] = (df['Time'].astype('int64') // 10**9)

    df['Netflow_Bytes'] = pd.to_numeric(df['Netflow_Bytes'], errors='coerce')
    df['Port'] = pd.to_numeric(df['Port'], errors='coerce')

    # no chained assignment warnings
    for col in ['Netflow_Bytes', 'Port']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df['file_entropy'] = np.log(df['Netflow_Bytes'].fillna(0) + 1)

    if 'IPaddress' in df.columns and 'ip' not in df.columns:
        df['ip'] = df['IPaddress']

    needed = FEATURES + ['file_entropy']
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df

def _load_or_train(df_hist: pd.DataFrame, contamination: float) -> Tuple[IsolationForest, StandardScaler]:
    """
    Load scaler & a model that matches 'contamination'. If not found, train new.
    """
    model_file = _model_file_for(contamination)

    scaler = None
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)

    if scaler is None:
        scaler = StandardScaler()
        X_hist0 = scaler.fit_transform(df_hist[FEATURES + ['file_entropy']])
        joblib.dump(scaler, SCALER_FILE)
    else:
        X_hist0 = scaler.transform(df_hist[FEATURES + ['file_entropy']])

    if os.path.exists(model_file):
        model = joblib.load(model_file)
        return model, scaler

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_hist0)
    joblib.dump(model, model_file)
    return model, scaler

def _append_cumulative_alerts(alerts: List[dict], total_new: int) -> None:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            cumul = json.load(f)
    else:
        cumul = {'total_anomalies': 0, 'alerts': []}
    cumul['alerts'].extend(alerts)
    cumul['total_anomalies'] = cumul.get('total_anomalies', 0) + total_new
    with open(LOG_FILE, 'w') as f:
        json.dump(cumul, f, default=str)


# ---------- Public API ----------
def run_anomaly_detection(data_file: str, contamination: float = DEFAULT_CONTAM) -> pd.DataFrame:
    """
    Run the anomaly detector on a CIC-IDS2017-style CSV.
      - contamination: float in (0, 0.5); typical 0.001–0.05
      - Returns a DataFrame of anomalies with reasoning & hashed IDs
      - Saves gdpr_audit_log.csv and updates cumulative_alerts.json
    """
    if not (0 < contamination < 0.5):
        raise ValueError("contamination must be between 0 and 0.5")

    df_raw = _load_dataset_robust(data_file)
    if len(df_raw) > 50_000:  # sample for speed; remove to use full data
        df_raw = df_raw.sample(n=50_000, random_state=42).reset_index(drop=True)

    df = _prepare_features(df_raw)

    split = int(0.7 * len(df))
    df_hist = df.iloc[:split].copy()
    df_new = df.iloc[split:].copy()

    model, scaler = _load_or_train(df_hist, contamination)

    all_anoms = []
    cumul_alerts = []
    total_new = 0
    run_num = (split // CHUNK_SIZE) + 1

    for i in range(0, len(df_new), CHUNK_SIZE):
        chunk = df_new.iloc[i:i + CHUNK_SIZE].copy()
        if chunk.empty:
            break

        X = scaler.transform(chunk[FEATURES + ['file_entropy']])
        preds = model.predict(X)  # -1 anomaly, 1 normal
        chunk.loc[:, 'anomaly'] = preds

        anoms = chunk[chunk['anomaly'] == -1].copy()
        if not anoms.empty:
            total_new += len(anoms)

            anoms.loc[:, 'Time_hashed'] = anoms['Time'].apply(_hash_field)
            if 'ip' in anoms.columns:
                anoms.loc[:, 'ip_hashed'] = anoms['ip'].apply(_hash_field)
            elif 'IPaddress' in anoms.columns:
                anoms.loc[:, 'ip_hashed'] = anoms['IPaddress'].apply(_hash_field)

            def _reason(row):
                r = []
                if row['file_entropy'] > np.log(1e5): r.append("high entropy > log(1e5)")
                if row['Port'] > 10000: r.append("high dest port >10000")
                if row['Netflow_Bytes'] > 1e6: r.append("very high throughput >1e6 B/s")
                return "; ".join(r) if r else "outlier pattern"

            anoms.loc[:, 'reasoning'] = anoms.apply(_reason, axis=1)
            all_anoms.append(anoms)

            for _, row in anoms.iterrows():
                cumul_alerts.append({
                    'Time_hashed': row.get('Time_hashed'),
                    'ip_hashed': row.get('ip_hashed', _hash_field(row.get('ip', row.get('IPaddress', 'n/a')))),
                    'Netflow_Bytes': float(row['Netflow_Bytes']),
                    'Port': float(row['Port']),
                    'file_entropy': float(row['file_entropy']),
                    'reasoning': row['reasoning'],
                    'run_num': run_num,
                    'Prediction': int(row.get('Prediction')) if 'Prediction' in row else 'N/A'
                })

        run_num += 1
        time.sleep(0.01)

    anomalies_df = (pd.concat(all_anoms, ignore_index=True)
                    if all_anoms else
                    pd.DataFrame(columns=['Time','Time_numeric','Netflow_Bytes','Port','file_entropy',
                                          'Time_hashed','ip_hashed','reasoning','Prediction','anomaly']))

    if not anomalies_df.empty:
        cols = ['Time','Time_numeric','Netflow_Bytes','Port','file_entropy','Time_hashed','ip_hashed','reasoning']
        if 'Prediction' in anomalies_df.columns: cols.append('Prediction')
        cols = [c for c in cols if c in anomalies_df.columns]
        anomalies_df.to_csv(AUDIT_FILE, index=False, columns=cols)

    _append_cumulative_alerts(cumul_alerts, total_new)
    return anomalies_df


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    from getpass import getpass

    parser = argparse.ArgumentParser(description="Run GDPR-compliant anomaly detection on a CSV.")
    parser.add_argument("data_file", help="Path to CIC-IDS2017 CSV")
    parser.add_argument("--contam", type=float, default=DEFAULT_CONTAM,
                        help="IsolationForest contamination (e.g., 0.005, 0.01, 0.02)")
    parser.add_argument("--investigate", action="store_true", help="Prompt for access code and de-hash anomalies")
    args = parser.parse_args()

    anoms = run_anomaly_detection(args.data_file, contamination=args.contam)
    print(f"Anomalies detected: {len(anoms)}")
    if not anoms.empty:
        preview_cols = [c for c in ['Time','Port','Netflow_Bytes','reasoning'] if c in anoms.columns]
        print(anoms[preview_cols].head(20).to_string(index=False))
        print(f"\nSaved anonymized audit log → {AUDIT_FILE}")
        print(f"Updated cumulative log     → {LOG_FILE}")

        if args.investigate:
            access_code = getpass("Enter access code: ")
            if access_code == "KEA2025":
                time_candidates = anoms['Time'].unique().tolist()
                anoms.loc[:, 'Time_dehashed'] = anoms['Time_hashed'].apply(
                    lambda h: next((t for t in time_candidates if _hash_field(t) == h), "No match")
                )
                if 'ip_hashed' in anoms.columns:
                    ip_candidates: List[str] = []
                    if 'ip' in anoms.columns:
                        ip_candidates = anoms['ip'].dropna().astype(str).unique().tolist()
                    elif 'IPaddress' in anoms.columns:
                        ip_candidates = anoms['IPaddress'].dropna().astype(str).unique().tolist()
                    anoms.loc[:, 'ip_dehashed'] = anoms['ip_hashed'].apply(
                        lambda h: next((ip for ip in ip_candidates if _hash_field(ip) == h), "No match")
                    )
                    show_cols = ['Time_dehashed', 'ip_dehashed', 'reasoning']
                else:
                    show_cols = ['Time_dehashed', 'reasoning']

                print("\n🔎 De-hashed anomalies for investigation:")
                print(anoms[show_cols].head(20).to_string(index=False))
            else:
                print("Access denied — GDPR compliance enforced.")
