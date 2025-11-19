# detector.py
import os
import json
import time
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ========= Config (base) =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_FILE = os.path.join(BASE_DIR, 'scaler.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'cumulative_alerts.json')
AUDIT_FILE = os.path.join(BASE_DIR, 'gdpr_audit_log.csv')
BASELINE_FILE = os.path.join(BASE_DIR, 'baseline_history.csv')

CHUNK_SIZE = 200
DEFAULT_CONTAM = 0.02

# Vi bruger så mange relevante numeriske felter som muligt fra det nye datasæt
FEATURES = [
    # Afledt tid
    "Time_numeric",

    # Alias felter
    "Netflow_Bytes",  # = bidirectional_bytes
    "Port",           # = dst_port

    # Header / basis
    "src_port",
    "protocol",
    "ip_version",
    "vlan_id",

    # Flow durations / bytes / packets
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_first_seen_ms",
    "src2dst_last_seen_ms",
    "src2dst_duration_ms",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_first_seen_ms",
    "dst2src_last_seen_ms",
    "dst2src_duration_ms",
    "dst2src_packets",
    "dst2src_bytes",

    # Rate (per second)
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "src2dst_min_ps",
    "src2dst_mean_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",
    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",

    # Piat (inter-arrival time)
    "bidirectional_min_piat_ms",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms",
    "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms",
    "src2dst_max_piat_ms",
    "dst2src_min_piat_ms",
    "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms",
    "dst2src_max_piat_ms",

    # TCP-flag tællinger
    "bidirectional_syn_packets",
    "bidirectional_cwr_packets",
    "bidirectional_ece_packets",
    "bidirectional_urg_packets",
    "bidirectional_ack_packets",
    "bidirectional_psh_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "src2dst_syn_packets",
    "src2dst_cwr_packets",
    "src2dst_ece_packets",
    "src2dst_urg_packets",
    "src2dst_ack_packets",
    "src2dst_psh_packets",
    "src2dst_rst_packets",
    "src2dst_fin_packets",
    "dst2src_syn_packets",
    "dst2src_cwr_packets",
    "dst2src_ece_packets",
    "dst2src_urg_packets",
    "dst2src_ack_packets",
    "dst2src_psh_packets",
    "dst2src_rst_packets",
    "dst2src_fin_packets",
]

SALT = b"your_project_salt_2025"
MAX_BASELINE_ROWS = 100_000


# ---------- Helpers ----------
def _model_file_for(contam: float) -> str:
    return os.path.join(BASE_DIR, f"ransomware_model_c{contam:.4f}.joblib")


def _hash_field(field) -> str:
    return hashlib.sha256(SALT + str(field).encode()).hexdigest()[:8]


def _load_dataset_robust(path: str) -> pd.DataFrame:
    """
    Loader KUN det nye flow-datasæt med kolonner som:
    id, expiration_id, src_ip, src_mac, src_oui, src_port, dst_ip, dst_mac, dst_oui, dst_port,
    protocol, ip_version, vlan_id, bidirectional_first_seen_ms, ..., label
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_FILE not found: {path}")

    df = pd.read_csv(path, low_memory=False, encoding_errors='ignore')
    df.columns = df.columns.str.strip()

    required_cols = [
        "bidirectional_first_seen_ms",
        "dst_port",
        "bidirectional_bytes",
        "dst_ip",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for new dataset: {missing}")

    # ---- Time (datetime) ----
    df["Time"] = pd.to_datetime(
        df["bidirectional_first_seen_ms"], unit="ms", errors="coerce"
    )

    # ---- Alias kolonner til resten af koden ----
    # bytes / throughput
    df["Netflow_Bytes"] = pd.to_numeric(df["bidirectional_bytes"], errors="coerce")
    # destination port
    df["Port"] = pd.to_numeric(df["dst_port"], errors="coerce")
    # IP til hashing/anonymisering
    df = df.rename(columns={"dst_ip": "IPaddress"})

    # ---- Label → Prediction (optional) ----
    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"

    if label_col and "Prediction" not in df.columns:
        df = df.rename(columns={label_col: "Label"})
        df["Prediction"] = df["Label"].apply(
            lambda x: 1 if str(x).strip().upper() == "BENIGN" else 0
        )

    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tid → Time_numeric (sekunder siden epoch)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Time_numeric"] = (df["Time"].astype("int64") // 10**9)

    # Alle de features vi forventer, skal være numeriske
    numeric_cols = set(FEATURES) | {
        "Netflow_Bytes",
        "Port",
    }

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # "file_entropy" er en simpel proxy baseret på volumen
    df["file_entropy"] = np.log(df["Netflow_Bytes"].fillna(0) + 1)

    # IP alias (til hashing)
    if "IPaddress" in df.columns and "ip" not in df.columns:
        df["ip"] = df["IPaddress"]

    needed = FEATURES + ["file_entropy"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def _load_or_train(df_hist: pd.DataFrame, contamination: float) -> Tuple[IsolationForest, StandardScaler]:
    model_file = _model_file_for(contamination)

    # scaler
    if os.path.exists(SCALER_FILE):
        scaler: StandardScaler = joblib.load(SCALER_FILE)
        X_hist0 = scaler.transform(df_hist[FEATURES + ["file_entropy"]])
    else:
        scaler = StandardScaler()
        X_hist0 = scaler.fit_transform(df_hist[FEATURES + ["file_entropy"]])
        joblib.dump(scaler, SCALER_FILE)

    # model
    if os.path.exists(model_file):
        model: IsolationForest = joblib.load(model_file)
        print(f"[LOAD] Loaded existing model: {model_file}")
        return model, scaler

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_hist0)
    joblib.dump(model, model_file)
    print(f"[TRAIN] Trained new model on {len(df_hist)} rows → {model_file}")
    return model, scaler


def _append_cumulative_alerts(alerts: List[dict], total_new: int) -> None:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            cumul = json.load(f)
    else:
        cumul = {"total_anomalies": 0, "alerts": []}
    cumul["alerts"].extend(alerts)
    cumul["total_anomalies"] = cumul.get("total_anomalies", 0) + total_new
    with open(LOG_FILE, "w") as f:
        json.dump(cumul, f, default=str)


# ---------- Public API ----------
def run_anomaly_detection(data_file: str, contamination: float = DEFAULT_CONTAM) -> pd.DataFrame:
    if not (0 < contamination < 0.5):
        raise ValueError("contamination must be between 0 and 0.5")

    # 1) load raw CSV (nyt schema)
    df_raw = _load_dataset_robust(data_file)
    if len(df_raw) > 50_000:
        df_raw = df_raw.sample(n=50_000, random_state=42).reset_index(drop=True)

    # 2) load baseline if we have it
    if os.path.exists(BASELINE_FILE):
        df_hist = pd.read_csv(BASELINE_FILE, low_memory=False)
        print(f"[i] Loaded existing baseline from {BASELINE_FILE} with {len(df_hist)} rows.")
        df_new = _prepare_features(df_raw)
    else:
        # build baseline from this CSV
        if "Label" in df_raw.columns:
            benign_mask = df_raw["Label"].astype(str).str.upper() == "BENIGN"
            df_hist_raw = df_raw[benign_mask].copy()
            if df_hist_raw.empty:
                split = int(0.7 * len(df_raw))
                df_hist_raw = df_raw.iloc[:split].copy()
                df_new_raw = df_raw.iloc[split:].copy()
                print("[i] Label column found but no BENIGN rows; using 70/30 split.")
            else:
                df_new_raw = df_raw.drop(df_hist_raw.index, errors="ignore").reset_index(drop=True)
                print(f"[i] Using {len(df_hist_raw)} BENIGN rows for training baseline.")
        else:
            split = int(0.7 * len(df_raw))
            df_hist_raw = df_raw.iloc[:split].copy()
            df_new_raw = df_raw.iloc[split:].copy()
            print("[i] No Label column; using first 70% for training baseline.")

        df_hist = _prepare_features(df_hist_raw)
        df_new = _prepare_features(df_new_raw)

    if df_hist.empty:
        raise ValueError("No rows available for training the model (df_hist is empty).")

    # 3) load/train model
    model, scaler = _load_or_train(df_hist, contamination)

    # 4) detect anomalies on new data
    all_anoms = []
    cumul_alerts = []
    total_new = 0
    run_num = 1

    # vi samler normale rækker op → single retrain
    collected_normals = []

    for i in range(0, len(df_new), CHUNK_SIZE):
        chunk = df_new.iloc[i : i + CHUNK_SIZE].copy()
        if chunk.empty:
            break

        X = scaler.transform(chunk[FEATURES + ["file_entropy"]])
        preds = model.predict(X)
        chunk.loc[:, "anomaly"] = preds

        # anomalies
        anoms = chunk[chunk["anomaly"] == -1].copy()
        if not anoms.empty:
            total_new += len(anoms)

            anoms.loc[:, "Time_hashed"] = anoms["Time"].apply(_hash_field)
            if "ip" in anoms.columns:
                anoms.loc[:, "ip_hashed"] = anoms["ip"].apply(_hash_field)
            elif "IPaddress" in anoms.columns:
                anoms.loc[:, "ip_hashed"] = anoms["IPaddress"].apply(_hash_field)

            def _reason(row):
                r = []

                # Høj "entropi" baseret på volumen
                try:
                    if row["file_entropy"] > np.log(1e5):
                        r.append("high entropy > log(1e5)")
                except KeyError:
                    pass

                # Høj destination port (ephemeral/ukonventionel)
                try:
                    if row["Port"] > 50000:
                        r.append("high dest port >50000")
                except KeyError:
                    pass

                # Meget stor datamængde
                try:
                    if row["Netflow_Bytes"] > 5_000_000:
                        r.append("large data transfer >5MB")
                except KeyError:
                    pass

                # Mange pakker
                if "bidirectional_packets" in row.index and row["bidirectional_packets"] > 5000:
                    r.append("packet burst >5000 packets")

                # Lang session
                if "bidirectional_duration_ms" in row.index and row["bidirectional_duration_ms"] > 60_000:
                    r.append("long session >60s")

                # SYN-heavy (scan/flood)
                if (
                    "bidirectional_syn_packets" in row.index
                    and "bidirectional_ack_packets" in row.index
                    and row["bidirectional_syn_packets"] > 50
                    and row["bidirectional_ack_packets"] < 5
                ):
                    r.append("syn-heavy traffic (possible scan)")

                return "; ".join(r) if r else "outlier pattern"

            anoms.loc[:, "reasoning"] = anoms.apply(_reason, axis=1)
            all_anoms.append(anoms)

            for _, row in anoms.iterrows():
                cumul_alerts.append(
                    {
                        "Time_hashed": row.get("Time_hashed"),
                        "ip_hashed": row.get(
                            "ip_hashed",
                            _hash_field(row.get("ip", row.get("IPaddress", "n/a"))),
                        ),
                        "Netflow_Bytes": float(row["Netflow_Bytes"]),
                        "Port": float(row["Port"]),
                        "file_entropy": float(row["file_entropy"]),
                        "reasoning": row["reasoning"],
                        "run_num": run_num,
                        "Prediction": int(row.get("Prediction"))
                        if "Prediction" in row
                        else "N/A",
                    }
                )

        # collect normals
        normals = chunk[chunk["anomaly"] == 1].copy()
        if not normals.empty:
            collected_normals.append(normals)

        run_num += 1
        time.sleep(0.01)

    # 5) single retrain til sidst (kun hvis vi har nye normale data)
    if collected_normals:
        add_normals = pd.concat(collected_normals, ignore_index=True)
        df_hist = pd.concat([df_hist, add_normals], ignore_index=True)
        df_hist = df_hist.drop_duplicates().iloc[-MAX_BASELINE_ROWS:]
        X_hist_new = scaler.transform(df_hist[FEATURES + ["file_entropy"]])
        model.fit(X_hist_new)
        joblib.dump(model, _model_file_for(contamination))
        df_hist.to_csv(BASELINE_FILE, index=False)
        print(
            f"[RETRAIN] Final retrain with {len(add_normals)} new normal rows → baseline={len(df_hist)}"
        )
    else:
        df_hist.to_csv(BASELINE_FILE, index=False)

    # 6) output
    anomalies_df = (
        pd.concat(all_anoms, ignore_index=True)
        if all_anoms
        else pd.DataFrame(
            columns=[
                "Time",
                "Time_numeric",
                "Netflow_Bytes",
                "Port",
                "file_entropy",
                "Time_hashed",
                "ip_hashed",
                "reasoning",
                "Prediction",
                "anomaly",
            ]
        )
    )

    if not anomalies_df.empty:
        cols = [
            "Time",
            "Time_numeric",
            "Netflow_Bytes",
            "Port",
            "file_entropy",
            "Time_hashed",
            "ip_hashed",
            "reasoning",
        ]
        if "Prediction" in anomalies_df.columns:
            cols.append("Prediction")
        cols = [c for c in cols if c in anomalies_df.columns]
        anomalies_df.to_csv(AUDIT_FILE, index=False, columns=cols)

    _append_cumulative_alerts(cumul_alerts, total_new)
    return anomalies_df


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    from getpass import getpass

    parser = argparse.ArgumentParser(
        description="Run GDPR-compliant anomaly detection on new flow dataset."
    )
    parser.add_argument("data_file", help="Path to new flow CSV")
    parser.add_argument(
        "--contam",
        type=float,
        default=DEFAULT_CONTAM,
        help="IsolationForest contamination (e.g., 0.005, 0.01, 0.02)",
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help="Prompt for access code and de-hash anomalies",
    )
    args = parser.parse_args()

    anoms = run_anomaly_detection(args.data_file, contamination=args.contam)
    print(f"Anomalies detected: {len(anoms)}")
    if not anoms.empty:
        preview_cols = [
            c
            for c in ["Time", "Port", "Netflow_Bytes", "reasoning"]
            if c in anoms.columns
        ]
        print(anoms[preview_cols].head(20).to_string(index=False))
        print(f"\nSaved anonymized audit log → {AUDIT_FILE}")
        print(f"Updated cumulative log     → {LOG_FILE}")

        if args.investigate:
            access_code = getpass("Enter access code: ")
            if access_code == "KEA2025":
                time_candidates = anoms["Time"].unique().tolist()
                anoms.loc[:, "Time_dehashed"] = anoms["Time_hashed"].apply(
                    lambda h: next(
                        (t for t in time_candidates if _hash_field(t) == h), "No match"
                    )
                )
                if "ip_hashed" in anoms.columns:
                    ip_candidates: List[str] = []
                    if "ip" in anoms.columns:
                        ip_candidates = (
                            anoms["ip"].dropna().astype(str).unique().tolist()
                        )
                    elif "IPaddress" in anoms.columns:
                        ip_candidates = (
                            anoms["IPaddress"].dropna().astype(str).unique().tolist()
                        )
                    anoms.loc[:, "ip_dehashed"] = anoms["ip_hashed"].apply(
                        lambda h: next(
                            (ip for ip in ip_candidates if _hash_field(ip) == h),
                            "No match",
                        )
                    )
                    show_cols = ["Time_dehashed", "ip_dehashed", "reasoning"]
                else:
                    show_cols = ["Time_dehashed", "reasoning"]

                print("\n🔎 De-hashed anomalies for investigation:")
                print(anoms[show_cols].head(20).to_string(index=False))
            else:
                print("Access denied — GDPR compliance enforced.")
