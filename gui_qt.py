# gui_qt.py
import sys
import os
import webbrowser
import hashlib
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QTabWidget,
    QLineEdit, QSplitter, QGroupBox, QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Make sure your detector exposes run_anomaly_detection(path, contamination=...)
# If your detector module is named differently, adjust the import below.
from detector import run_anomaly_detection

# helper: map label to friendly category (keeps previous mapping)
def map_label_to_category(label: str) -> str:
    if not isinstance(label, str):
        return "Other"
    s = label.strip().lower()
    if "ransom" in s:
        return "Ransomware"
    if "ddos" in s or s.startswith("dos ") or "hulk" in s or "goldeneye" in s or "slowloris" in s:
        return "DDoS/DoS"
    if "portscan" in s or "port scan" in s:
        return "PortScan"
    if "web attack" in s or "xss" in s or "sql" in s:
        return "Web Attack"
    if "bruteforce" in s or "ftp-patator" in s or "ssh-patator" in s:
        return "Brute Force"
    if "infiltration" in s:
        return "Infiltration"
    if "bot" in s or "botnet" in s:
        return "Bot"
    if "heartbleed" in s:
        return "Heartbleed"
    if s == "benign":
        return "BENIGN"
    return "Other"

def detect_label_column(df: pd.DataFrame):
    candidates = [
        "label", "Label", "Label_raw", "attack", "Attack", "Category", "category",
        "trafficLabel", "TrafficLabel"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: check last column if stringy and has distinct tokens (like BENIGN,DDoS,...)
    last = df.columns[-1]
    try:
        if df[last].dtype == object and df[last].nunique() > 1:
            return last
    except Exception:
        pass
    return None

class AnomalyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDPR-Compliant Anomaly Detector (All threats)")
        self.setGeometry(50, 50, 1400, 900)

        self.csv_path = None
        self.df_full = None
        self.anoms = pd.DataFrame()
        self.label_col = None

        # Top bar
        top = QWidget()
        topl = QHBoxLayout(top)
        self.info = QLabel("Load a CIC CSV (or any flow CSV with Label column).")
        self.contam_edit = QLineEdit("0.02")
        self.contam_edit.setFixedWidth(80)
        btn_load = QPushButton("Load CSV…")
        btn_load.clicked.connect(self.load_csv)
        self.run_btn = QPushButton("Run Anomaly Detection")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_detector)
        topl.addWidget(self.info, 1)
        topl.addWidget(QLabel("Contamination:"))
        topl.addWidget(self.contam_edit)
        topl.addWidget(btn_load)
        topl.addWidget(self.run_btn)

        # Tabs
        self.tabs = QTabWidget()
        # Results tab
        results = QWidget()
        rlay = QVBoxLayout(results)
        self.summary = QLabel("")
        rlay.addWidget(self.summary)
        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget()
        splitter.addWidget(self.table)
        fig, self.ax_hist = plt.subplots()
        self.canvas_hist = FigureCanvas(fig)
        splitter.addWidget(self.canvas_hist)
        splitter.setSizes([500, 300])
        rlay.addWidget(splitter)
        actions = QHBoxLayout()
        self.btn_export = QPushButton("Export anomalies (CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_anomalies)
        self.btn_open_audit = QPushButton("Open audit log (CSV)")
        self.btn_open_audit.setEnabled(False)
        self.btn_open_audit.clicked.connect(self.open_audit)
        actions.addStretch(1)
        actions.addWidget(self.btn_export)
        actions.addWidget(self.btn_open_audit)
        rlay.addLayout(actions)

        # Analytics tab (minimal - reuse previous)
        analytics = QWidget()
        al = QVBoxLayout(analytics)
        gp_ports = QGroupBox("Top Destination Ports (anomalies)")
        gp_ports.setLayout(QVBoxLayout())
        self.tbl_ports = QTableWidget()
        gp_ports.layout().addWidget(self.tbl_ports)
        fig2, self.ax_ports = plt.subplots()
        self.canvas_ports = FigureCanvas(fig2)
        gp_ports.layout().addWidget(self.canvas_ports)
        al.addWidget(gp_ports)
        gp_reason = QGroupBox("Reasoning Breakdown")
        gp_reason.setLayout(QVBoxLayout())
        fig3, self.ax_reason = plt.subplots()
        self.canvas_reason = FigureCanvas(fig3)
        gp_reason.layout().addWidget(self.canvas_reason)
        al.addWidget(gp_reason)
        gp_ts = QGroupBox("Anomaly Throughput over Time")
        gp_ts.setLayout(QVBoxLayout())
        fig4, self.ax_ts = plt.subplots()
        self.canvas_ts = FigureCanvas(fig4)
        gp_ts.layout().addWidget(self.canvas_ts)
        al.addWidget(gp_ts)

        # Preview tab
        preview = QWidget()
        pl = QVBoxLayout(preview)
        self.preview_lbl = QLabel("No file loaded.")
        self.preview_tbl = QTableWidget()
        pl.addWidget(self.preview_lbl)
        pl.addWidget(self.preview_tbl)

        # Threats tab: left = dataset labeled malicious rows (all attack types), middle = categories list, right = suspicious anomalies
        threats = QWidget()
        tl = QVBoxLayout(threats)
        toprow = QHBoxLayout()
        self.threats_info = QLabel("Threats: load & run detection to populate.")
        toprow.addWidget(self.threats_info, 1)
        self.btn_export_threats = QPushButton("Export malicious rows (CSV)")
        self.btn_export_threats.setEnabled(False)
        self.btn_export_threats.clicked.connect(self.export_threats)
        toprow.addWidget(self.btn_export_threats)
        tl.addLayout(toprow)

        split = QSplitter(Qt.Horizontal)

        # Left: malicious rows (all)
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.addWidget(QLabel("Dataset-labeled malicious rows (non-BENIGN)"))
        self.tbl_label_mal = QTableWidget()
        left_l.addWidget(self.tbl_label_mal)
        split.addWidget(left_w)

        # Middle: categories list
        mid_w = QWidget()
        mid_l = QVBoxLayout(mid_w)
        mid_l.addWidget(QLabel("Attack types (click one to filter)"))
        self.list_cats = QListWidget()
        self.list_cats.itemSelectionChanged.connect(self.on_category_selected)
        mid_l.addWidget(self.list_cats)
        split.addWidget(mid_w)

        # Right: suspicious anomalies (heuristic)
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.addWidget(QLabel("Suspicious anomalies (heuristic)"))
        self.tbl_suspicious = QTableWidget()
        right_l.addWidget(self.tbl_suspicious)
        split.addWidget(right_w)

        split.setSizes([500, 200, 500])
        tl.addWidget(split)

        # Add tabs
        self.tabs.addTab(results, "Results")
        self.tabs.addTab(analytics, "Analytics")
        self.tabs.addTab(preview, "Preview")
        self.tabs.addTab(threats, "Threats")

        root = QWidget()
        main = QVBoxLayout(root)
        main.addWidget(top)
        main.addWidget(self.tabs)
        self.setCentralWidget(root)

    # ---------------- UI actions ----------------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open flow CSV", "", "CSV Files (*.csv);;All files (*.*)")
        if not path:
            return
        self.csv_path = path
        self.info.setText(f"Selected: {path}")
        self.run_btn.setEnabled(True)
        try:
            self.df_full = pd.read_csv(path, low_memory=False, encoding_errors="ignore")
            # preview
            self.preview_lbl.setText(f"Preview of {os.path.basename(path)} (first 200 rows)")
            self.fill_table(self.preview_tbl, self.df_full.head(200))
            # detect label column
            self.label_col = detect_label_column(self.df_full)
            if not self.label_col:
                QMessageBox.information(
                    self,
                    "No label column",
                    "This CSV has no obvious Label column. The Threats left pane will remain empty until you load a CSV that contains attack labels.",
                )
            else:
                QMessageBox.information(self, "Label column detected", f"Detected label column: {self.label_col}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.df_full = None
            self.label_col = None

    def run_detector(self):
        if not self.csv_path:
            QMessageBox.warning(self, "No file", "Please load a CSV first.")
            return
        try:
            contam = float(self.contam_edit.text())
        except Exception:
            QMessageBox.warning(self, "Contamination", "Enter a valid contamination like 0.01")
            return

        self.info.setText("Running anomaly detection…")
        try:
            # run your detection function - ensure detector.run_anomaly_detection returns a DataFrame of anomalies
            self.anoms = run_anomaly_detection(self.csv_path, contamination=contam)
            n = len(self.anoms)
            self.info.setText(f"Detected {n} anomalies. Saved gdpr_audit_log.csv & cumulative_alerts.json")
            self.btn_export.setEnabled(n > 0)
            self.btn_open_audit.setEnabled(True)
            # show top columns in results table
            cols = [c for c in ["Time", "Port", "Netflow_Bytes", "reasoning", "Time_hashed", "ip_hashed"] if c in self.anoms.columns]
            self.fill_table(self.table, self.anoms[cols] if cols else self.anoms)
            # histogram
            self.ax_hist.clear()
            if n and "Netflow_Bytes" in self.anoms.columns:
                self.anoms["Netflow_Bytes"].astype(float).plot.hist(bins=50, ax=self.ax_hist)
                self.ax_hist.set_title("Netflow_Bytes (anomalies)")
            self.canvas_hist.draw()
            self.summary.setText(f"File: {os.path.basename(self.csv_path)} | Anomalies: {n} | Contamination: {contam}")

            # populate analytics and threats
            self.populate_analytics()
            self.populate_threats_lists()
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def populate_analytics(self):
        df = self.anoms
        # ports
        self.ax_ports.clear()
        if "Port" in df.columns and not df.empty:
            top_ports = df["Port"].value_counts().head(12).reset_index()
            top_ports.columns = ["Port", "Count"]
            self.fill_table(self.tbl_ports, top_ports)
            top_ports.plot(kind="bar", x="Port", y="Count", ax=self.ax_ports, legend=False)
            self.ax_ports.set_title("Top Destination Ports (anomalies)")
        self.canvas_ports.draw()
        # reasoning
        self.ax_reason.clear()
        if "reasoning" in df.columns and not df.empty:
            counts = {}
            for r in df["reasoning"].fillna(""):
                for token in [t.strip() for t in r.split(";") if t.strip()]:
                    counts[token] = counts.get(token, 0) + 1
            if counts:
                pd.Series(counts).sort_values(ascending=False).plot(kind="bar", ax=self.ax_reason)
                self.ax_reason.set_title("Reasoning Breakdown")
        self.canvas_reason.draw()
        # ts
        self.ax_ts.clear()
        if {"Time", "Netflow_Bytes"}.issubset(df.columns) and not df.empty:
            ts = df[["Time", "Netflow_Bytes"]].assign(Time=pd.to_datetime(df["Time"], errors="coerce")).dropna().sort_values("Time")
            if not ts.empty:
                self.ax_ts.plot(ts["Time"], ts["Netflow_Bytes"])
                self.ax_ts.set_title("Anomaly Throughput over Time")
                self.ax_ts.set_xlabel("Time")
                self.ax_ts.set_ylabel("bytes")
        self.canvas_ts.draw()

    # ---------------- Threats: label list and tables ----------------
    def populate_threats_lists(self):
        # Build left table: all non-BENIGN labeled rows
        if self.df_full is None or self.label_col is None:
            self.fill_table(self.tbl_label_mal, pd.DataFrame())
            self.list_cats.clear()
            self.threats_info.setText("Threats: load & run detection to populate.")
            return

        df = self.df_full.copy()
        lab = df[self.label_col].astype(str)
        non_benign_mask = lab.str.upper().str.strip() != "BENIGN"
        mal = df[non_benign_mask].copy()
        if mal.empty:
            self.fill_table(self.tbl_label_mal, pd.DataFrame())
            self.list_cats.clear()
            self.threats_info.setText(
                "Threats: dataset-labeled malicious rows = 0 | suspicious anomalies (heuristic) = " + str(len(self.anoms))
            )
            return

        # Add AttackCategory column for grouping
        mal["AttackCategory"] = mal[self.label_col].astype(str).map(map_label_to_category)
        # Fill left table with all malicious rows
        self.fill_table(self.tbl_label_mal, mal.head(1000))

        # Fill category list with counts per label (use the raw label values)
        label_counts = mal[self.label_col].value_counts()
        self.list_cats.clear()
        # Always add "All" item
        all_item = QListWidgetItem(f"All ({len(mal)})")
        all_item.setData(Qt.UserRole, "ALL")
        self.list_cats.addItem(all_item)
        for lbl, cnt in label_counts.items():
            it = QListWidgetItem(f"{lbl} ({cnt})")
            it.setData(Qt.UserRole, lbl)
            self.list_cats.addItem(it)

        self.threats_info.setText(f"Threats: dataset-labeled malicious rows = {len(mal)} | suspicious anomalies (heuristic) = {len(self.anoms)}")
        # also enable export
        self.btn_export_threats.setEnabled(True)

    def on_category_selected(self):
        """User clicked an attack label in the middle list; filter left and right panes accordingly."""
        sel = self.list_cats.selectedItems()
        if not sel:
            return
        tag = sel[0].data(Qt.UserRole)
        # left pane: show only rows with this label (or All)
        if self.df_full is None or self.label_col is None:
            return
        df = self.df_full.copy()
        lab = df[self.label_col].astype(str)
        non_benign_mask = lab.str.upper().str.strip() != "BENIGN"
        mal = df[non_benign_mask].copy()
        if tag != "ALL":
            mal_sel = mal[mal[self.label_col].astype(str) == tag]
        else:
            mal_sel = mal
        self.fill_table(self.tbl_label_mal, mal_sel.head(1000))

        # right pane: filter anomalies by matching label if possible via time-join OR by category heuristics
        right_df = pd.DataFrame()
        if not self.anoms.empty:
            # if we can time-join, do it: convert times to string keys
            if "Time" in self.anoms.columns and "Time" in self.df_full.columns:
                dfR = self.df_full.copy()
                dfR["_t"] = pd.to_datetime(dfR["Time"], errors="coerce").astype(str)
                an = self.anoms.copy()
                an["_t"] = pd.to_datetime(an["Time"], errors="coerce").astype(str)
                joined = pd.merge(an, dfR[["_t", self.label_col]], left_on="_t", right_on="_t", how="left")
                if tag != "ALL":
                    right_df = joined[joined[self.label_col].astype(str) == tag].copy()
                else:
                    right_df = joined.copy()
            else:
                # fallback: show heuristic anomalies but try to filter by map_label_to_category(tag) if tag is specific
                a = self.anoms.copy()
                nb = pd.to_numeric(a.get("Netflow_Bytes", pd.Series(0, index=a.index)), errors="coerce").fillna(0)
                high_entropy = (nb + 1).apply(lambda x: 0 if x <= 0 else np.log(x)) > np.log(1e5)
                high_bytes = nb > 1e7
                high_port = pd.to_numeric(a.get("Port", pd.Series(0, index=a.index)), errors="coerce").fillna(0) > 10000
                cand = a[high_entropy & (high_bytes | high_port)].copy()
                if tag != "ALL":
                    # attempt category mapping
                    cat = map_label_to_category(tag)
                    # if the tag indicates DDoS, PortScan etc we just show the candidate set (best-effort)
                    right_df = cand
                else:
                    right_df = cand

        if right_df.empty:
            self.tbl_suspicious.clear()
            self.tbl_suspicious.setRowCount(0)
            self.tbl_suspicious.setColumnCount(0)
        else:
            show_cols = [c for c in ["Time", "Port", "Netflow_Bytes", "reasoning", "Time_hashed"] if c in right_df.columns]
            self.fill_table(self.tbl_suspicious, right_df[show_cols] if show_cols else right_df)

    # ---------------- exports & helpers ----------------
    def export_anomalies(self):
        if self.anoms.empty:
            QMessageBox.information(self, "No anomalies", "No anomalies to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save anomalies", "anomalies.csv", "CSV Files (*.csv)")
        if not path:
            return
        self.anoms.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Anomalies saved to: {path}")

    def export_threats(self):
        if self.df_full is None or self.label_col is None:
            QMessageBox.information(self, "No threats", "No labeled malicious rows to export.")
            return
        # determine selected category
        sel = self.list_cats.selectedItems()
        if sel and sel[0].data(Qt.UserRole) != "ALL":
            tag = sel[0].data(Qt.UserRole)
            df = self.df_full[self.df_full[self.label_col].astype(str) == tag].copy()
            default_name = f"malicious_{tag}.csv"
        else:
            df = self.df_full[self.df_full[self.label_col].astype(str).str.upper().str.strip() != "BENIGN"].copy()
            default_name = "malicious_all.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save malicious rows", default_name, "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Malicious rows exported to: {path}")

    def open_audit(self):
        audit = os.path.join(os.getcwd(), "gdpr_audit_log.csv")
        if os.path.exists(audit):
            webbrowser.open(audit)
        else:
            QMessageBox.information(self, "Not found", "gdpr_audit_log.csv not found. Run detection first.")

    def fill_table(self, table: QTableWidget, df: pd.DataFrame):
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        rows = min(len(df), 2000)
        cols = list(df.columns)
        table.setRowCount(rows)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        for i, (_, row) in enumerate(df.head(rows).iterrows()):
            for j, c in enumerate(cols):
                item = QTableWidgetItem(str(row[c]) if pd.notna(row[c]) else "")
                table.setItem(i, j, item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AnomalyApp()
    win.show()
    sys.exit(app.exec_())
