#!/usr/bin/env python3
"""
port_ip_detector.py

- Used by Port/IP Detector GUI.
- Runs a lightweight IsolationForest on port + byte metrics to find anomalies.
- Can optionally query VirusTotal for IP reputation (requires VT_API_KEY env var).
- Saves results (restricted CSV + vt_results/ JSON).

Tilpasset til det nye flow-datasæt:
  - dst_ip      → IPaddress
  - dst_port    → Port
  - bidirectional_bytes → Netflow_Bytes
"""

import os
import time
import json
import ipaddress
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
import joblib  # for saving/loading model

# ------------------ Config ------------------
OUTPUT_ALERT_CSV = "port_ip_alerts.csv"
VT_OUTDIR = "vt_results"
VT_SLEEP_SEC = 15  # sleep between requests (for free VT tier)
MODEL_PATH = "isolation_model.pkl"  # saved IF + scaler

# ------------------ Helpers ------------------
def pick_ip_column(df: pd.DataFrame) -> str:
    """Return the ip-like column name if present."""
    for c in [
        "Destination IP",
        "Dst IP",
        "Destination Address",
        "Dst Addr",
        "Destination IP Address",
        "IPaddress",
        "ip",
        "dst_ip",  # new flow dataset
    ]:
        if c in df.columns:
            return c
    return None


def is_public_ip(ip: str) -> bool:
    """Return True if IP is globally routable (not private/reserved)."""
    try:
        ipobj = ipaddress.ip_address(ip)
        return not (
            ipobj.is_private
            or ipobj.is_reserved
            or ipobj.is_loopback
            or ipobj.is_multicast
        )
    except Exception:
        return False


# ------------------ Isolation Forest ------------------
def detect_port_ip_anomalies(df_raw: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Small IsolationForest on ['Port','Netflow_Bytes','file_entropy']
    to flag odd port/throughput combos.

    Nu tilpasset så den også spiller med det nye flow-datasæt:
      - dst_port           → Port
      - bidirectional_bytes → Netflow_Bytes
      - dst_ip             → IPaddress
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Port", "Netflow_Bytes", "file_entropy", "pi_anomaly"])

    df = df_raw.copy()

    # ---- Normalize column names for new + old schemas ----
    # New dataset: dst_port -> Port
    if "dst_port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"dst_port": "Port"})

    # Old CIC-style: Destination Port -> Port
    if "Destination Port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"Destination Port": "Port"})

    # New dataset: bidirectional_bytes -> Netflow_Bytes
    if "bidirectional_bytes" in df.columns and "Netflow_Bytes" not in df.columns:
        df = df.rename(columns={"bidirectional_bytes": "Netflow_Bytes"})

    # Old CIC-style: Flow Bytes/s -> Netflow_Bytes
    if "Flow Bytes/s" in df.columns and "Netflow_Bytes" not in df.columns:
        df = df.rename(columns={"Flow Bytes/s": "Netflow_Bytes"})

    # New dataset: dst_ip -> IPaddress (used for VT + display)
    if "dst_ip" in df.columns and "IPaddress" not in df.columns:
        df = df.rename(columns={"dst_ip": "IPaddress"})

    # ---- Derive Netflow_Bytes if still missing (old CIC flows) ----
    if "Netflow_Bytes" not in df.columns:
        if {"Total Length of Fwd Packets", "Total Length of Bwd Packets"}.issubset(df.columns):
            df["Netflow_Bytes"] = (
                pd.to_numeric(df["Total Length of Fwd Packets"], errors="coerce")
                + pd.to_numeric(df["Total Length of Bwd Packets"], errors="coerce")
            )
        else:
            return pd.DataFrame(columns=["Port", "Netflow_Bytes", "file_entropy", "pi_anomaly"])

    # ---- Ensure numerics ----
    for c in ["Port", "Netflow_Bytes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["file_entropy"] = np.log(df["Netflow_Bytes"].fillna(0) + 1)

    # ---- Build feature frame (keep index to merge back) ----
    feats = df[["Port", "Netflow_Bytes", "file_entropy"]].dropna().reset_index()
    if feats.empty:
        return pd.DataFrame(columns=["Port", "Netflow_Bytes", "file_entropy", "pi_anomaly"])

    X_raw = feats[["Port", "Netflow_Bytes", "file_entropy"]].values

    # ---- Try to load existing model ----
    loaded = False
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            scaler = bundle["scaler"]
            if_model = bundle["model"]
            X = scaler.transform(X_raw)
            preds = if_model.predict(X)
            loaded = True
        except Exception as e:
            print(f"[ML] Failed to load existing model – retraining. Reason: {e}")
            loaded = False

    if not loaded:
        # Train new
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        if_model = IsolationForest(contamination=contamination, random_state=42)
        if_model.fit(X)
        # Save both scaler and model
        joblib.dump({"scaler": scaler, "model": if_model}, MODEL_PATH)
        print(f"[ML] Trained new IsolationForest and saved to {MODEL_PATH}")
        preds = if_model.predict(X)

    # ---- Attach predictions ----
    feats["pi_anomaly"] = preds  # -1 anomaly, 1 normal

    # Merge back to original df (to keep other columns, incl. IPs)
    out = feats.merge(df.reset_index(), on="index", how="left", suffixes=("", "_raw"))

    # Return only anomalies; include IP if we have it
    cols = ["Port", "Netflow_Bytes", "file_entropy", "pi_anomaly"]
    for extra in ["IPaddress", "src_ip", "dst_ip"]:
        if extra in out.columns and extra not in cols:
            cols.append(extra)

    return out[out["pi_anomaly"] == -1][cols]


# ------------------ VirusTotal (v3 API) ------------------
def vt_lookup_ips(ips: List[str], api_key: str, outdir: str = VT_OUTDIR, sleep_sec: int = VT_SLEEP_SEC) -> Dict[str, dict]:
    """
    Query VirusTotal v3 /ip_addresses/<ip>
    Returns dict ip -> result, saves to vt_results/ (JSON files + cumulative log).
    """
    os.makedirs(outdir, exist_ok=True)
    headers = {"x-apikey": api_key}
    base = "https://www.virustotal.com/api/v3/ip_addresses/{}"
    results = {}
    outfile = os.path.join(outdir, "vt_batch_results.json")

    # resume existing
    if os.path.exists(outfile):
        try:
            with open(outfile, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for i, ip in enumerate(ips):
        if ip in results:
            print(f"[VT] Skipping {ip} (cached)")
            continue
        if not is_public_ip(ip):
            print(f"[VT] Skipping {ip} (non-public)")
            results[ip] = {"skipped": True}
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            continue
        print(f"[VT] {i+1}/{len(ips)} -> {ip}")
        try:
            r = requests.get(base.format(ip), headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                results[ip] = {"ok": True, "data": data}
                with open(os.path.join(outdir, f"{ip.replace(':','_')}.json"), "w") as f2:
                    json.dump(data, f2)
            else:
                results[ip] = {"ok": False, "status": r.status_code, "text": r.text}
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            results[ip] = {"ok": False, "error": str(e)}
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
        print(f"  - sleeping {sleep_sec}s to respect VT limits")
        time.sleep(sleep_sec)
    return results


# ------------------ CLI / standalone mode ------------------
def main(csv_path: str, contamination: float = 0.02, watch_ports: List[int] = None,
         malicious_ips: List[str] = None, vt_lookup: bool = False):
    """
    Command-line entry: loads CSV, runs IsolationForest and optional VirusTotal lookups.
    Tilpasset til nyt flow-datasæt, men kan stadig håndtere ældre feltnavne.
    """
    print("Loading CSV...")
    df = pd.read_csv(csv_path, low_memory=False, encoding_errors="ignore")

    # Normalize IP column
    ip_col = pick_ip_column(df)
    if ip_col:
        df = df.rename(columns={ip_col: "IPaddress"})
        print(f"Detected IP column: {ip_col}")
    else:
        print("No IP column detected.")

    # Normalize port column (new + old)
    if "dst_port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"dst_port": "Port"})
    if "Destination Port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"Destination Port": "Port"})

    df["Port"] = pd.to_numeric(df.get("Port", pd.Series([-1] * len(df))), errors="coerce").fillna(-1).astype(int)

    # Default watch ports
    if watch_ports is None:
        watch_ports = [443, 53, 80]

    print(f"Running IsolationForest on ports/IP metrics (contamination={contamination})...")
    anomalies = detect_port_ip_anomalies(df, contamination)
    print(f"Detected {len(anomalies)} anomalies.")

    # Watchlist filter
    df_matches = df[df["Port"].isin(watch_ports)].copy()
    if malicious_ips and "IPaddress" in df.columns:
        df_matches = pd.concat(
            [df_matches, df[df["IPaddress"].isin(malicious_ips)]],
            ignore_index=True
        )
    df_matches.drop_duplicates(inplace=True)
    print(f"Found {len(df_matches)} matches (watch ports or malicious IPs).")

    # Save restricted CSV
    if not df_matches.empty:
        df_matches.to_csv(OUTPUT_ALERT_CSV, index=False)
        print(f"Saved {OUTPUT_ALERT_CSV} (restricted raw data).")

    # Optional VT enrichment
    if vt_lookup:
        api_key = os.getenv("VT_API_KEY")
        if not api_key:
            print("VT_API_KEY not set; skipping VirusTotal lookup.")
        else:
            ips = sorted(set(df_matches.get("IPaddress", []))) if "IPaddress" in df_matches.columns else []
            if malicious_ips:
                ips.extend(malicious_ips)
            ips = list(set(ips))
            if ips:
                print(f"Querying VT for {len(ips)} IPs...")
                vt_lookup_ips(ips, api_key=api_key, outdir=VT_OUTDIR, sleep_sec=VT_SLEEP_SEC)
            else:
                print("No public IPs to query in VT.")
    print("Done.")


# ------------------ CLI usage ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Port/IP Detector + optional VirusTotal enrichment")
    parser.add_argument("csv", help="Path to flow CSV file")
    parser.add_argument("--contam", type=float, default=0.02, help="IsolationForest contamination level")
    parser.add_argument("--watch-ports", type=str, default="443,53,80",
                        help="Comma-separated ports to watch (e.g., 443,22,53)")
    parser.add_argument("--malicious-ips", type=str, default="", help="Comma-separated known malicious IPs (demo)")
    parser.add_argument("--vt", action="store_true", help="Run VirusTotal lookups (requires VT_API_KEY env var)")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.watch_ports.split(",") if p.strip()]
    malicious = [ip.strip() for ip in args.malicious_ips.split(",") if ip.strip()]
    main(args.csv, contamination=args.contam, watch_ports=ports, malicious_ips=malicious, vt_lookup=args.vt)
