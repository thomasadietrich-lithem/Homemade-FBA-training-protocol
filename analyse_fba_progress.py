#!/usr/bin/env python
# analyse_fba_progress.py
#
# Dashboard analysis for FBA (Feature-Based Attention) training sessions.
#
# Reads all *_summary.json files from ./data (produced by:
#   - cb_fba_training_psychopy.py
#   - cb_fba_training_psychopy_DR.py)
#
# Builds a single "dashboard" figure:
#   - Left column  : direction threshold (deg) over sessions
#   - Right column : accuracy (%) over sessions
#   - One row per experimental condition:
#         (task, H_deg, V_internal, angle_set)
#
# This is meant to resemble the way progress is visualized in
# Huxlin Lab publications: threshold curves + performance curves.

import os
import json
import datetime as dt
import matplotlib.pyplot as plt

DATA_DIR = "data"


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


def build_dashboard(groups):
    """
    Build a single dashboard figure:
      - one row per condition
      - left: threshold vs session
      - right: accuracy vs session
    """
    n_cond = len(groups)
    if n_cond == 0:
        print("No conditions to display.")
        return

    # Prepare figure and axes grid
    fig, axes = plt.subplots(
        nrows=n_cond,
        ncols=2,
        figsize=(10, 4 * n_cond),
        sharex="col"
    )

    # If only one condition, axes is 1D; make access uniform
    if n_cond == 1:
        axes = [axes]

    fig.suptitle("FBA training dashboard", fontsize=14)

    # Iterate over each condition and plot
    for row_idx, (key, sessions) in enumerate(sorted(groups.items(), key=lambda x: x[0])):
        task, H_deg, V_internal, angle_set = key
        V_field = -V_internal if V_internal is not None else None

        sess_sorted = sort_sessions(sessions)

        thresholds = []
        accuracies = []
        labels = []

        for js in sess_sorted:
            thr = js.get("final_threshold_deg", None)
            acc = js.get("accuracy_percent", None)
            ts = parse_timestamp(js)
            label = ts.strftime("%m-%d") if ts is not None else "?"
            thresholds.append(thr)
            accuracies.append(acc)
            labels.append(label)

        x = list(range(1, len(sess_sorted) + 1))

        # Left column: threshold
        ax_thr = axes[row_idx][0]
        ax_thr.plot(x, thresholds, marker="o")
        ax_thr.set_ylabel("Threshold (deg)")
        cond_label = f"{task}, H={H_deg}°, V={V_field}°, angle_set={angle_set}"
        ax_thr.set_title(cond_label, fontsize=9)
        ax_thr.grid(True, linestyle="--", alpha=0.3)
        ax_thr.set_xticks(x)
        ax_thr.set_xticklabels(labels, rotation=45)

        # Right column: accuracy
        ax_acc = axes[row_idx][1]
        ax_acc.plot(x, accuracies, marker="o")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title(cond_label, fontsize=9)
        ax_acc.grid(True, linestyle="--", alpha=0.3)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels, rotation=45)

        # Console summary for this condition
        print("\n==============================")
        print("Condition:", cond_label)
        print(f"  Number of sessions : {len(sess_sorted)}")
        print(f"  Threshold (start)  : {thresholds[0]}")
        print(f"  Threshold (last)   : {thresholds[-1]}")
        print(f"  Accuracy  (start)  : {accuracies[0]}")
        print(f"  Accuracy  (last)   : {accuracies[-1]}")
        print("  Files in order:")
        for js in sess_sorted:
            print("   -", js.get("_filename", "?"))

    # Common x-label at the bottom
    fig.text(0.5, 0.04, "Session (date)", ha="center")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.show()


def main():
    summaries = load_summaries()
    if not summaries:
        return

    groups = group_by_condition(summaries)
    build_dashboard(groups)


if __name__ == "__main__":
    main()
