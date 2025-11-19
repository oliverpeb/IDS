import sys
import os
import json
import threading
from typing import List

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QTabWidget,
    QLineEdit, QSplitter, QCheckBox, QMenu, QInputDialog
)
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from detector import run_anomaly_detection

try:
    from port_ip_detector import detect_port_ip_anomalies, vt_lookup_ips
except Exception:
    detect_port_ip_anomalies = None
    vt_lookup_ips = None

from vt_helpers import _detect_ip_col, _vt_verdict

# default forslag til VT lookups
MAX_VT_IPS_DEFAULT = 30


def _table_from_df(tbl: QTableWidget, df: pd.DataFrame, max_rows: int = 1500):
    tbl.clear()
    if df is None or df.empty:
        tbl.setRowCount(0)
        tbl.setColumnCount(0)
        return

    df_show = df.head(max_rows).copy()
    cols = list(df_show.columns)
    tbl.setRowCount(len(df_show))
    tbl.setColumnCount(len(cols))
    tbl.setHorizontalHeaderLabels(cols)

    for i, (_, row) in enumerate(df_show.iterrows()):
        highlight = None
        if "verdict" in row:
            v = str(row["verdict"]).lower()
            if v == "high":
                highlight = QColor(255, 130, 130)
            elif v == "medium":
                highlight = QColor(255, 210, 160)
            elif v == "low":
                highlight = QColor(200, 255, 200)
        elif "vt_flag" in row and str(row["vt_flag"]).strip().lower() in ("1", "true", "yes"):
            highlight = QColor(255, 160, 160)
        elif "Label" in row and str(row["Label"]).strip().upper() == "MALICIOUS":
            highlight = QColor(255, 210, 160)

        for j, c in enumerate(cols):
            # Brug positionsbaseret adgang for at undgå problemer med duplikerede kolonnenavne
            cell_value = row.iloc[j]
            val = "" if pd.isna(cell_value) else str(cell_value)
            item = QTableWidgetItem(val)
            if highlight:
                item.setBackground(highlight)
            tbl.setItem(i, j, item)


def _rename_port_and_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tilpas preview/match-data til at have de kolonner,
    som GUI'en forventer: Port, Netflow_Bytes, IPaddress.
    Virker både på det nye flow-datasæt og evt. gamle CSV'er.
    """
    df = df.copy()

    # Destination port → Port (gammelt schema)
    if "Destination Port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"Destination Port": "Port"})

    # Ny type: dst_port → Port
    if "dst_port" in df.columns and "Port" not in df.columns:
        df = df.rename(columns={"dst_port": "Port"})

    # Flow bytes/s → Netflow_Bytes (gammelt schema)
    if "Flow Bytes/s" in df.columns and "Netflow_Bytes" not in df.columns:
        df = df.rename(columns={"Flow Bytes/s": "Netflow_Bytes"})

    # Ny type: bidirectional_bytes → Netflow_Bytes
    if "bidirectional_bytes" in df.columns and "Netflow_Bytes" not in df.columns:
        df = df.rename(columns={"bidirectional_bytes": "Netflow_Bytes"})

    # Ny type: dst_ip → IPaddress (til VT + context menu)
    if "dst_ip" in df.columns and "IPaddress" not in df.columns:
        df = df.rename(columns={"dst_ip": "IPaddress"})

    return df


class WorkerSignals(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)


class VTWorker(threading.Thread):
    def __init__(self, ips: List[str], sleep_sec: int = 15):
        super().__init__(daemon=True)
        self.ips = ips
        self.sleep_sec = sleep_sec
        self.signals = WorkerSignals()

    def run(self):
        if vt_lookup_ips is None:
            self.signals.error.emit("port_ip_detector.vt_lookup_ips not available")
            return
        api_key = os.getenv("VT_API_KEY")
        if not api_key:
            self.signals.error.emit("VT_API_KEY not set in environment")
            return
        try:
            results = vt_lookup_ips(self.ips, api_key, sleep_sec=self.sleep_sec)
            self.signals.done.emit(results)
        except Exception as e:
            self.signals.error.emit(str(e))


class PortIPApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Port/IP Detector GUI (IsolationForest + VT)")
        self.setGeometry(60, 60, 1500, 950)

        self.csv_path = None
        self.df_full = pd.DataFrame()
        self.anoms_main = pd.DataFrame()
        self.anoms_portip = pd.DataFrame()
        self.df_matches = pd.DataFrame()

        # --- Top bar ---
        top = QWidget()
        tlay = QHBoxLayout(top)
        self.info = QLabel("Load a flow CSV (new flow dataset).")
        self.edit_contam = QLineEdit("0.02")
        self.edit_contam.setFixedWidth(80)
        self.edit_watch_ports = QLineEdit("443,53,80")
        self.edit_watch_ports.setFixedWidth(180)
        self.edit_bad_ips = QLineEdit("")
        self.edit_bad_ips.setPlaceholderText("malicious IPs (comma-separated)")
        self.edit_bad_ips.setFixedWidth(260)
        self.chk_vt = QCheckBox("VirusTotal lookups")
        self.chk_vt.setToolTip("Requires VT_API_KEY in environment. Skips private/reserved IPs.")

        btn_load = QPushButton("Load CSV…")
        btn_load.clicked.connect(self.on_load)
        self.btn_run = QPushButton("Run Port/IP Detection")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.on_run)

        tlay.addWidget(self.info, 1)
        tlay.addWidget(QLabel("Contam:"))
        tlay.addWidget(self.edit_contam)
        tlay.addWidget(QLabel("Watch ports:"))
        tlay.addWidget(self.edit_watch_ports)
        tlay.addWidget(self.edit_bad_ips)
        tlay.addWidget(self.chk_vt)
        tlay.addWidget(btn_load)
        tlay.addWidget(self.btn_run)

        # --- Tabs ---
        tabs = QTabWidget()

        # Results tab
        tab_res = QWidget()
        rlay = QVBoxLayout(tab_res)
        self.lbl_summary = QLabel("")
        rlay.addWidget(self.lbl_summary)
        split = QSplitter(Qt.Vertical)
        self.tbl_anoms = QTableWidget()
        split.addWidget(self.tbl_anoms)
        fig, self.ax_hist = plt.subplots()
        self.canvas_hist = FigureCanvas(fig)
        split.addWidget(self.canvas_hist)
        split.setSizes([600, 300])
        rlay.addWidget(split)
        actions = QHBoxLayout()
        self.btn_open_audit = QPushButton("Open gdpr_audit_log.csv")
        self.btn_open_audit.clicked.connect(self.open_audit)
        actions.addStretch(1)
        actions.addWidget(self.btn_open_audit)
        rlay.addLayout(actions)

        # Port/IP IF tab
        tab_pi = QWidget()
        pilay = QVBoxLayout(tab_pi)
        self.lbl_pi = QLabel("Port/IP-focused IsolationForest anomalies: 0")
        pilay.addWidget(self.lbl_pi)
        split2 = QSplitter(Qt.Vertical)
        self.tbl_pi = QTableWidget()
        split2.addWidget(self.tbl_pi)
        fig2, self.ax_ports = plt.subplots()
        self.canvas_ports = FigureCanvas(fig2)
        split2.addWidget(self.canvas_ports)
        split2.setSizes([600, 300])
        pilay.addWidget(split2)

        # Matches & VT tab
        tab_matches = QWidget()
        mlay = QVBoxLayout(tab_matches)
        self.lbl_matches = QLabel("Matches: 0")
        mlay.addWidget(self.lbl_matches)
        self.tbl_matches = QTableWidget()
        self.tbl_matches.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_matches.customContextMenuRequested.connect(self.on_matches_context_menu)
        mlay.addWidget(self.tbl_matches)
        hl = QHBoxLayout()
        self.btn_export_matches = QPushButton("Export matches → port_ip_alerts.csv")
        self.btn_export_matches.clicked.connect(self.export_matches)
        self.btn_vt = QPushButton("Run VT on matches")
        self.btn_vt.clicked.connect(self.run_vt)
        hl.addStretch(1)
        hl.addWidget(self.btn_export_matches)
        hl.addWidget(self.btn_vt)
        mlay.addLayout(hl)

        # Preview tab
        tab_prev = QWidget()
        pv = QVBoxLayout(tab_prev)
        self.lbl_prev = QLabel("No file loaded.")
        self.tbl_prev = QTableWidget()
        pv.addWidget(self.lbl_prev)
        pv.addWidget(self.tbl_prev)

        tabs.addTab(tab_res, "Results (Main IF)")
        tabs.addTab(tab_pi, "Port/IP IF")
        tabs.addTab(tab_matches, "Matches & VT")
        tabs.addTab(tab_prev, "Preview")

        root = QWidget()
        main = QVBoxLayout(root)
        main.addWidget(top)
        main.addWidget(tabs)
        self.setCentralWidget(root)

    # ---------- Slots ----------
    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open flow CSV",
            "",
            "CSV Files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        self.csv_path = path
        self.info.setText(f"Selected: {path}")
        self.btn_run.setEnabled(True)
        try:
            self.df_full = pd.read_csv(path, low_memory=False, encoding_errors="ignore")
            # Tilpas nye felter → Port / Netflow_Bytes / IPaddress
            self.df_full = _rename_port_and_bytes(self.df_full)
            self.lbl_prev.setText(f"Preview of {os.path.basename(path)} (first 200 rows)")
            _table_from_df(self.tbl_prev, self.df_full.head(200))
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.df_full = pd.DataFrame()

    def on_run(self):
        if not self.csv_path:
            QMessageBox.warning(self, "No file", "Please load a CSV first.")
            return
        try:
            contam = float(self.edit_contam.text())
        except Exception:
            QMessageBox.warning(self, "Contamination", "Enter a valid number like 0.02")
            return
        try:
            watch_ports = [int(x.strip()) for x in self.edit_watch_ports.text().split(",") if x.strip()]
        except Exception:
            QMessageBox.warning(self, "Watch ports", "Enter comma-separated integers, e.g., 443,53,80")
            return
        bad_ips = [x.strip() for x in self.edit_bad_ips.text().split(",") if x.strip()]

        self.info.setText("Running detections…")
        QApplication.processEvents()

        # 1) Main IsolationForest
        try:
            self.anoms_main = run_anomaly_detection(self.csv_path, contamination=contam)
        except Exception as e:
            QMessageBox.critical(self, "Detector error", str(e))
            return

        cols = [c for c in ["Time", "Port", "Netflow_Bytes", "reasoning", "Time_hashed", "ip_hashed"]
                if c in self.anoms_main.columns]
        _table_from_df(self.tbl_anoms, self.anoms_main[cols] if cols else self.anoms_main)
        self.lbl_summary.setText(
            f"File: {os.path.basename(self.csv_path)} | Main anomalies: {len(self.anoms_main)} | Contamination: {contam}"
        )

        # Histogram
        self.ax_hist.clear()
        if len(self.anoms_main) and "Netflow_Bytes" in self.anoms_main.columns:
            pd.to_numeric(self.anoms_main["Netflow_Bytes"], errors="coerce") \
                .dropna().astype(float).plot.hist(bins=50, ax=self.ax_hist)
            self.ax_hist.set_title("Netflow_Bytes (anomalies)")
        self.canvas_hist.draw()

        # 2) Port/IP-focused IF
        if detect_port_ip_anomalies is None:
            self.anoms_portip = pd.DataFrame()
            self.lbl_pi.setText("Port/IP-focused IsolationForest anomalies: 0 (module not found)")
        else:
            try:
                # Her bruger vi df_full (som har Port/Netflow_Bytes/IPaddress efter _rename_port_and_bytes)
                self.anoms_portip = detect_port_ip_anomalies(self.df_full, contamination=contam)
            except Exception as e:
                QMessageBox.warning(self, "Port/IP IF", f"Port/IP IF error: {e}")
                self.anoms_portip = pd.DataFrame()
            _table_from_df(self.tbl_pi, self.anoms_portip)
            self.lbl_pi.setText(f"Port/IP-focused IsolationForest anomalies: {len(self.anoms_portip)}")

            # Info om port/ip-model (port_ip_detector)
            if os.path.exists("isolation_model.pkl"):
                self.info.setText("Reused existing Port/IP IsolationForest model ✅")
            else:
                self.info.setText("Trained a new Port/IP IsolationForest model 🧠")

            # ports chart
            self.ax_ports.clear()
            if not self.anoms_portip.empty and "Port" in self.anoms_portip.columns:
                top_ports = self.anoms_portip["Port"].value_counts().head(15).sort_values(ascending=False)
                top_ports.plot(kind="bar", ax=self.ax_ports, legend=False)
                self.ax_ports.set_title("Top Ports among Port/IP anomalies")
            self.canvas_ports.draw()

        # 3) Matches (watch ports / malicious IPs)
        dfm = self.df_full.copy()
        if "Port" in dfm.columns:
            dfm["Port"] = pd.to_numeric(dfm["Port"], errors="coerce").fillna(-1).astype(int)
            port_mask = dfm["Port"].isin(watch_ports)
        else:
            port_mask = pd.Series([False] * len(dfm))

        ip_col = _detect_ip_col(dfm)
        if bad_ips and ip_col:
            ip_mask = dfm[ip_col].astype(str).isin(bad_ips)
        else:
            ip_mask = pd.Series([False] * len(dfm))

        self.df_matches = dfm[port_mask | ip_mask].copy()
        self.lbl_matches.setText(f"Matches: {len(self.df_matches)} (watch ports or malicious IPs)")
        self.lbl_matches.setStyleSheet("")
        _table_from_df(self.tbl_matches, self.df_matches.head(500))

        if self.chk_vt.isChecked():
            self.run_vt()

    def export_matches(self):
        if self.df_matches is None or self.df_matches.empty:
            QMessageBox.information(self, "No data", "No matches to export.")
            return
        out = os.path.join(os.getcwd(), "port_ip_alerts.csv")
        try:
            self.df_matches.to_csv(out, index=False)
            QMessageBox.information(self, "Saved", f"Saved: {out}\n(Contains raw IPs — handle with care)")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def run_vt(self):
        if vt_lookup_ips is None:
            QMessageBox.warning(self, "VT", "VirusTotal module not available in port_ip_detector.")
            return
        if self.df_matches is None or self.df_matches.empty:
            QMessageBox.information(self, "VT", "No matches to query.")
            return

        ip_col = _detect_ip_col(self.df_matches)
        if not ip_col:
            QMessageBox.information(self, "VT", "No IP column found in matches.")
            return

        ip_series = self.df_matches[ip_col].dropna().astype(str)
        if ip_series.empty:
            QMessageBox.information(self, "VT", "No IPs to query.")
            return

        ip_counts = ip_series.value_counts()
        total_unique = len(ip_counts)

        # Dialog: lad brugeren vælge hvor mange IP'er vi vil slå op
        default_n = min(MAX_VT_IPS_DEFAULT, total_unique)
        n_to_query, ok = QInputDialog.getInt(
            self,
            "VirusTotal lookups",
            f"Der er {total_unique} unikke IP'er i matches.\n"
            f"Hvor mange vil du slå op? (max {total_unique})",
            value=default_n,
            min=1,
            max=total_unique
        )
        if not ok:
            # bruger trykkede Cancel
            return

        # Vælg de mest hyppige IP'er
        top_ips = ip_counts.head(n_to_query).index.tolist()

        self.btn_vt.setEnabled(False)
        self.btn_vt.setText(f"Running VT for {n_to_query} IPs…")
        worker = VTWorker(top_ips)
        worker.signals.done.connect(self.vt_done)
        worker.signals.error.connect(self.vt_error)
        worker.start()
        self._vt_worker = worker  # keep reference

    def apply_vt_results_to_matches(self, results: dict):
        if self.df_matches is None or self.df_matches.empty:
            return
        ip_col = _detect_ip_col(self.df_matches)
        if not ip_col:
            return

        # Byg VT-resultat-rows
        vt_rows = []
        for _, row in self.df_matches.iterrows():
            ip = str(row.get(ip_col, "")) if ip_col in row else ""
            vt_rows.append(_vt_verdict(results.get(ip, {})))
        vt_df = pd.DataFrame(vt_rows, index=self.df_matches.index)

        # Fjern gamle VT-kolonner, så vi undgår duplikater
        vt_cols = list(vt_df.columns)
        for col in vt_cols:
            if col in self.df_matches.columns:
                del self.df_matches[col]

        # Merge nye VT-felter ind
        self.df_matches = pd.concat([self.df_matches, vt_df], axis=1)
        _table_from_df(self.tbl_matches, self.df_matches.head(500))

        # Robust beregning af "flagged": håndter bool, 0/1, str, osv.
        vt_flag_obj = self.df_matches.get("vt_flag")
        flagged = 0
        if vt_flag_obj is not None:
            # Hvis det mod forventning er en DataFrame (duplikerede navne), tag første kolonne
            if isinstance(vt_flag_obj, pd.DataFrame):
                vt_flag_series = vt_flag_obj.iloc[:, 0]
            else:
                vt_flag_series = vt_flag_obj

            # Konverter til 0/1
            vt_flag_norm = (
                vt_flag_series
                .astype(str)
                .str.lower()
                .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
                .fillna(0)
            )
            try:
                flagged = int(vt_flag_norm.sum())
            except Exception:
                flagged = 0

        self.lbl_matches.setStyleSheet("color: red; font-weight: bold;" if flagged > 0 else "")
        self.lbl_matches.setText(
            f"Matches: {len(self.df_matches)} (watch ports or malicious IPs) | VT flagged: {flagged}"
        )

    def vt_done(self, results: dict):
        self.btn_vt.setEnabled(True)
        self.btn_vt.setText("Run VT on matches")
        ok = sum(1 for v in results.values() if isinstance(v, dict) and v.get("ok"))
        skipped = sum(1 for v in results.values() if isinstance(v, dict) and v.get("skipped"))
        self.apply_vt_results_to_matches(results)
        QMessageBox.information(
            self, "VT",
            f"Done. OK: {ok}, skipped: {skipped}.\nResults saved under vt_results/ and merged into table."
        )

    def vt_error(self, msg: str):
        self.btn_vt.setEnabled(True)
        self.btn_vt.setText("Run VT on matches")
        QMessageBox.critical(self, "VT error", msg)

    def on_matches_context_menu(self, pos):
        sel = self.tbl_matches.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        cols = [self.tbl_matches.horizontalHeaderItem(i).text() for i in range(self.tbl_matches.columnCount())]
        ip_col_name = None
        # Udvid med dst_ip (hvis preview ikke har renamet)
        for cand in ["Destination IP", "IPaddress", "Dst IP", "ip", "dst_ip"]:
            if cand in cols:
                ip_col_name = cand
                break

        ip = ""
        if ip_col_name:
            idx = cols.index(ip_col_name)
            cell = self.tbl_matches.item(row, idx)
            if cell:
                ip = cell.text().strip()

        menu = QMenu(self)
        act_vt = menu.addAction("Open in VirusTotal")
        act_copy = menu.addAction("Copy IP")
        action = menu.exec_(self.tbl_matches.viewport().mapToGlobal(pos))
        if action == act_vt and ip:
            url = f"https://www.virustotal.com/gui/ip-address/{ip}"
            try:
                if sys.platform.startswith("win"):
                    os.startfile(url)
                elif sys.platform == "darwin":
                    os.system(f"open '{url}'")
                else:
                    os.system(f"xdg-open '{url}'")
            except Exception:
                pass
        elif action == act_copy and ip:
            QApplication.clipboard().setText(ip)

    def open_audit(self):
        path = os.path.join(os.getcwd(), "gdpr_audit_log.csv")
        if os.path.exists(path):
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")
        else:
            QMessageBox.information(self, "Not found", "gdpr_audit_log.csv not found. Run detection first.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PortIPApp()
    win.show()
    sys.exit(app.exec_())
