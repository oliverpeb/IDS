# vt_helpers.py
import pandas as pd

def _detect_ip_col(df: pd.DataFrame):
    """
    Return the name of an IP column in df if present, else None.
    Tries several common column names.
    """
    for c in [
        "IPaddress",
        "Destination IP",
        "Dst IP",
        "Destination Address",
        "Dst Addr",
        "Destination IP Address",
        "ip",
    ]:
        if c in df.columns:
            return c
    return None


def _vt_verdict(res: dict) -> dict:
    """
    Parse a single VirusTotal v3 IP response into a flat summary.

    Input formats expected:
      - {"ok": True,  "data": <VT JSON>}
      - {"skipped": True}   # non-public/reserved IPs
      - {"ok": False, "status": <int>, "text": "..."} or {"error": "..."}  # errors

    Returns (all keys always present except vt_error/vt_note):
      {
        "vt_flag": bool,                # True if we consider it a hit
        "vt_malicious": int,            # last_analysis_stats.malicious
        "vt_suspicious": int,           # last_analysis_stats.suspicious
        "vt_harmless": int,             # last_analysis_stats.harmless
        "vt_reputation": int,           # attributes.reputation
        "vt_categories": "cat1;cat2",   # attributes.categories (flattened)
        "vt_error": "...",              # present if parsing/error/etc.
        "vt_note":  "...",              # present if skipped or non-critical note
      }
    """
    # Default structure
    out = {
        "vt_flag": False,
        "vt_malicious": 0,
        "vt_suspicious": 0,
        "vt_harmless": 0,
        "vt_reputation": 0,
        "vt_categories": "",
    }

    if not isinstance(res, dict):
        out["vt_error"] = "bad_result"
        return out

    if res.get("skipped"):
        out["vt_note"] = "skipped_non_public"
        return out

    if not res.get("ok"):
        out["vt_error"] = str(
            res.get("error") or res.get("text") or res.get("status") or "error"
        )
        return out

    try:
        data = res["data"]
        attrs = data["data"]["attributes"]
        stats = attrs.get("last_analysis_stats", {}) or {}
        mal = int(stats.get("malicious", 0))
        susp = int(stats.get("suspicious", 0))
        harmless = int(stats.get("harmless", 0))
        rep = int(attrs.get("reputation", 0))
        cats_dict = attrs.get("categories") or {}
        cats = ";".join(sorted({str(v) for v in cats_dict.values()}))

        # Simple, explainable flag rule
        vt_flag = (mal >= 1) or (rep < 0 and susp >= 1)

        out.update(
            {
                "vt_flag": vt_flag,
                "vt_malicious": mal,
                "vt_suspicious": susp,
                "vt_harmless": harmless,
                "vt_reputation": rep,
                "vt_categories": cats,
            }
        )
        return out
    except Exception as e:
        out["vt_error"] = f"parse_error:{e}"
        return out
