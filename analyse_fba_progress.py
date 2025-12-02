#!/usr/bin/env python
# analyse_fba_progress.py
#
# Dashboard analysis for FBA (Feature-Based Attention) training sessions.
#
# Reads all *_summary.json files from ./data produced by:
#   - cb_fba_training_psychopy.py
#   - cb_fba_training_psychopy_DR.py
#
# Builds a single "dashboard" figure:
#   - One row per experimental condition: (task, H_deg, V_internal, angle_set)
#   - Left  column : direction-range threshold (deg) over sessions
#                   (mean threshold + the 3 individual staircases)
#   - Right column : accuracy (%) over sessions
#   - Adds:
#       - 3-session moving average (mean threshold and accuracy)
#       - Linear regression lines (slope + R² printed to console)
#       - Recent trend over the last sessions (for plateau detection)
#   - Saves the dashboard as PNG in ./analysis_outputs/

import os
import json
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "analysis_outputs"


def load_summaries(data_dir=DATA_DIR):
    """Load all *_summary.json files inside /data and return them as dicts."""
    if not os.path.isdir(data_dir):
        print(f"No '{data_dir}' directory found. Nothing to analyse.")
        return []

    summaries = []
    for fname in os.listdir(data_dir):
        if not fname.endswith("_summary.json"):
            continue
        full_path = os.path.join(data_dir, fname)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                js = json.load(f)
            js["_filename"] = fname
            summaries.append(js)
        except Exception as e:
            print(f"Could not read {fname}: {e}")

    if not summaries:
        print("No summary files found in /data.")
    return summaries


def parse_timestamp(js):
    """Parse the timestamp field into a Python datetime object if possible."""
    ts = js.get("timestamp", None)
    if ts is None:
        return None

    # Common timestamp formats
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None


def group_by_condition(summaries):
    """
    Group sessions by experimental condition:
        key = (task, H_deg, V_internal, angle_set)
    This allows separate lines in the dashboard for different
    stimulus locations or task variants.
    """
    groups = {}
    for js in summaries:
        task = js.get("task", "UNKNOWN")
        loc = js.get("location_deg_internal", {})
        H_deg = loc.get("H", None)
        V_internal = loc.get("V_internal", None)
        angle_set = js.get("angle_set", None)

        key = (task, H_deg, V_internal, angle_set)
        groups.setdefault(key, []).append(js)

    return groups


def sort_sessions(sessions):
    """Sort sessions chronologically using timestamp or file modification time."""
    def sort_key(js):
        ts = parse_timestamp(js)
        if ts is not None:
            return ts

        # Fallback: use file modification date
        fname = js.get("_filename", "")
        full = os.path.join(DATA_DIR, fname)
        try:
            mtime = os.path.getmtime(full)
        except Exception:
            mtime = 0
        return dt.datetime.fromtimestamp(mtime)

    return sorted(sessions, key=sort_key)


def moving_average(values, window=3):
    """Compute simple moving average. Returns (x_indices, ma_values) or (None, None)."""
    v = np.array(values, dtype=float)
    if len(v) < window:
        return None, None
    ma = np.convolve(v, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(v) + 1)
    return x, ma


def linear_regression(x, y):
    """
    Simple linear regression y = a*x + b.
    Returns slope a, intercept b, R^2.
    If fewer than 2 points or NaN, returns (None, None, None).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return None, None, None

    x_m = x[mask]
    y_m = y[mask]
    a, b = np.polyfit(x_m, y_m, 1)
    y_pred = a * x_m + b
    ss_res = np.sum((y_m - y_pred) ** 2)
    ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return a, b, r2


def trend_last_segment(values, last_n=5):
    """
    Compute simple slope over the last N sessions:
    (last - first) / (N-1). Returns (slope, n_used).
    """
    v = np.array(values, dtype=float)
    n = len(v)
    if n < 2:
        return None, n
    n_seg = min(last_n, n)
    first = v[-n_seg]
    last = v[-1]
    slope = (last - first) / (n_seg - 1) if n_seg > 1 else None
    return slope, n_seg


def build_dashboard(groups):
    """
    Build a single dashboard figure:
      - one row per condition
      - left : threshold vs session (mean + 3 staircases)
      - right: accuracy vs session
      - raw data + moving average + regression line
    Also prints text summary for each condition and saves the figure.
    """
    n_cond = len(groups)
    if n_cond == 0:
        print("No conditions to display.")
        return

    fig, axes = plt.subplots(
        nrows=n_cond,
        ncols=2,
        figsize=(12, 4 * n_cond),
        sharex="col"
    )

    if n_cond == 1:
        axes = [axes]

    fig.suptitle("FBA training dashboard", fontsize=14)

    for row_idx, (key, sessions) in enumerate(sorted(groups.items(), key=lambda x: x[0])):
        task, H_deg, V_internal, angle_set = key
        V_field = -V_internal if V_internal is not None else None

        sess_sorted = sort_sessions(sessions)

        thresholds_mean = []
        thresholds_s1 = []
        thresholds_s2 = []
        thresholds_s3 = []
        accuracies = []
        labels = []

        for js in sess_sorted:
            # accuracy
            acc = js.get("accuracy_percent", None)
            accuracies.append(acc)

            # label from timestamp
            ts = parse_timestamp(js)
            label = ts.strftime("%m-%d") if ts is not None else "?"
            labels.append(label)

            # staircase details
            angle_range = js.get("angle_range", [])
            stairs = js.get("stair_levels", {})
            s1_idx = stairs.get("stair1", None)
            s2_idx = stairs.get("stair2", None)
            s3_idx = stairs.get("stair3", None)

            def idx_to_deg(idx):
                if idx is None or idx < 1 or idx > len(angle_range):
                    return np.nan
                return float(angle_range[idx - 1])

            thr1 = idx_to_deg(s1_idx)
            thr2 = idx_to_deg(s2_idx)
            thr3 = idx_to_deg(s3_idx)

            thresholds_s1.append(thr1)
            thresholds_s2.append(thr2)
            thresholds_s3.append(thr3)

            # mean threshold (Rochester convention)
            thr_mean = np.nanmean([thr1, thr2, thr3])
            thresholds_mean.append(thr_mean)

        x = np.arange(1, len(sess_sorted) + 1)
        cond_label = f"{task}, H={H_deg}°, V={V_field}°, angle_set={angle_set}"

        # --- Left column: threshold ---
        ax_thr = axes[row_idx][0]

        # three staircases (thin dotted lines)
        ax_thr.plot(x, thresholds_s1, marker="o", linestyle=":", alpha=0.5, label="Staircase 1")
        ax_thr.plot(x, thresholds_s2, marker="o", linestyle=":", alpha=0.5, label="Staircase 2")
        ax_thr.plot(x, thresholds_s3, marker="o", linestyle=":", alpha=0.5, label="Staircase 3")

        # mean threshold (thicker)
        ax_thr.plot(x, thresholds_mean, marker="o", linewidth=2, label="Mean threshold")

        ax_thr.set_ylabel("Threshold (deg)")
        ax_thr.set_title(cond_label, fontsize=9)
        ax_thr.grid(True, linestyle="--", alpha=0.3)
        ax_thr.set_xticks(x)
        ax_thr.set_xticklabels(labels, rotation=45)

        # Moving average (mean threshold)
        x_ma_thr, ma_thr = moving_average(thresholds_mean, window=3)
        if x_ma_thr is not None:
            ax_thr.plot(x_ma_thr, ma_thr, linestyle="--", label="3-session MA (mean)")

        # Regression (mean threshold)
        slope_thr, intercept_thr, r2_thr = linear_regression(x, thresholds_mean)
        if slope_thr is not None:
            y_pred_thr = slope_thr * x + intercept_thr
            ax_thr.plot(x, y_pred_thr, alpha=0.6, label=f"Linear fit (slope={slope_thr:.2f})")

        ax_thr.legend(fontsize=7)

        # --- Right column: accuracy ---
        ax_acc = axes[row_idx][1]
        ax_acc.plot(x, accuracies, marker="o", label="Accuracy")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title(cond_label, fontsize=9)
        ax_acc.grid(True, linestyle="--", alpha=0.3)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels, rotation=45)

        # Moving average (accuracy)
        x_ma_acc, ma_acc = moving_average(accuracies, window=3)
        if x_ma_acc is not None:
            ax_acc.plot(x_ma_acc, ma_acc, linestyle="--", label="3-session MA")

        # Regression (accuracy)
        slope_acc, intercept_acc, r2_acc = linear_regression(x, accuracies)
        if slope_acc is not None:
            y_pred_acc = slope_acc * x + intercept_acc
            ax_acc.plot(x, y_pred_acc, alpha=0.6, label=f"Linear fit (slope={slope_acc:.2f})")

        ax_acc.legend(fontsize=7)

        # ---- Console summary ----
        print("\n==============================")
        print("Condition:", cond_label)
        print(f"  Number of sessions        : {len(sess_sorted)}")
        print(f"  Mean threshold start/last : {thresholds_mean[0]} / {thresholds_mean[-1]}")
        print(f"  Stair1 thresholds         : {thresholds_s1}")
        print(f"  Stair2 thresholds         : {thresholds_s2}")
        print(f"  Stair3 thresholds         : {thresholds_s3}")
        print(f"  Accuracy start/last       : {accuracies[0]} / {accuracies[-1]}")

        # Prepare R² strings safely
        if slope_thr is not None:
            if r2_thr is not None:
                r2_thr_str = f"{r2_thr:.3f}"
            else:
                r2_thr_str = "n/a"
            print(
                f"  Threshold linear slope     : {slope_thr:.3f} deg/session "
                f"(R^2={r2_thr_str})"
            )

        if slope_acc is not None:
            if r2_acc is not None:
                r2_acc_str = f"{r2_acc:.3f}"
            else:
                r2_acc_str = "n/a"
            print(
                f"  Accuracy  linear slope     : {slope_acc:.3f} %/session "
                f"(R^2={r2_acc_str})"
            )

        # Recent trend (last N sessions) on mean threshold and accuracy
        thr_recent_slope, n_thr_seg = trend_last_segment(thresholds_mean, last_n=5)
        acc_recent_slope, n_acc_seg = trend_last_segment(accuracies, last_n=5)

        if thr_recent_slope is not None:
            print(
                f"  Recent threshold trend     : {thr_recent_slope:.3f} deg/session "
                f"(over last {n_thr_seg} sessions)"
            )
        if acc_recent_slope is not None:
            print(
                f"  Recent accuracy trend      : {acc_recent_slope:.3f} %/session "
                f"(over last {n_acc_seg} sessions)"
            )

        # Simple qualitative heuristic for decision-making
        if thr_recent_slope is not None:
            if abs(thr_recent_slope) < 1.0 and len(sess_sorted) >= 10:
                print("  NOTE: Threshold trend is flat (<1 deg/session) over the last sessions.")
                print("        If the mean threshold remains high, this may be a point to consider")
                print("        changing or slightly shifting the stimulus location.")
            elif thr_recent_slope < -1.0:
                print("  NOTE: Threshold is still clearly improving (decreasing).")
            elif thr_recent_slope > 1.0:
                print("  NOTE: Threshold is increasing; consider checking fatigue or setup.")

        print("  Files in chronological order:")
        for js in sess_sorted:
            print("   -", js.get("_filename", "?"))

    # Common x-label at the bottom
    fig.text(0.5, 0.04, "Session (date)", ha="center")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    # Save figure to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fba_dashboard.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nDashboard figure saved to: {out_path}")

    plt.show()


def main():
    summaries = load_summaries()
    if not summaries:
        return

    groups = group_by_condition(summaries)
    build_dashboard(groups)


if __name__ == "__main__":
    main()
