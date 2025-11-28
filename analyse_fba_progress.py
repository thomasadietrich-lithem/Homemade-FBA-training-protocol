#!/usr/bin/env python
# analyse_fba_progress.py
#
# Analysis script for FBA (Feature-Based Attention) training sessions
# using summary JSON files produced by:
#   - cb_fba_training_psychopy.py
#   - cb_fba_training_psychopy_DR.py
#
# This script generates figures similar to those seen in the
# University of Rochester / Huxlin Lab publications:
#   - Direction threshold (deg) over training sessions
#   - Accuracy (%) over training sessions
#
# It also groups sessions by condition:
#   (task, H_deg, V_internal, angle_set)
# so you can train at multiple visual field locations independently.

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
        (task, H_deg, V_internal, angle_set)
    This allows separate curves for different visual-field positions.
    """
    groups = {}
    for js in summaries:
        task = js.get("task", "UNKNOWN")
        loc = js.get("location_deg_internal", {})
        H = loc.get("H", None)
        Vint = loc.get("V_internal", None)
        angle_set = js.get("angle_set", None)

        key = (task, H, Vint, angle_set)
        groups.setdefault(key, []).append(js)

    return groups


def sort_sessions(sessions):
    """Sort sessions chronologically using timestamp or file modification time."""
    def sort_key(js):
        ts = parse_timestamp(js)
        if ts is not None:
            return ts

        # fallback: use file modification date
        fname = js.get("_filename", "")
        full = os.path.join(DATA_DIR, fname)
        try:
            mtime = os.path.getmtime(full)
        except Exception:
            mtime = 0
        return dt.datetime.fromtimestamp(mtime)

    return sorted(sessions, key=sort_key)


def plot_condition(key, sessions_sorted):
    """
    Create threshold and accuracy plots for one experimental condition.
    key = (task, H_deg, V_internal, angle_set)
    """
    task, H_deg, V_internal, angle_set = key
    V_field = -V_internal if V_internal is not None else None  # field convention

    thresholds = []
    accuracies = []
    labels = []

    for js in sessions_sorted:
        thr = js.get("final_threshold_deg", None)
        acc = js.get("accuracy_percent", None)
        ts = parse_timestamp(js)

        label = ts.strftime("%m-%d") if ts is not None else "?"
        thresholds.append(thr)
        accuracies.append(acc)
        labels.append(label)

    x = list(range(1, len(sessions_sorted) + 1))

    # --- Figure 1: Direction threshold ---
    plt.figure()
    plt.plot(x, thresholds, marker="o")
    plt.xlabel("Session")
    plt.ylabel("Direction threshold (deg)")
    title = (
        f"FBA training – Direction threshold\n"
        f"Task={task}, H={H_deg}°, V={V_field}°, angle_set={angle_set}"
    )
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(x, labels, rotation=45)

    # --- Figure 2: Accuracy ---
    plt.figure()
    plt.plot(x, accuracies, marker="o")
    plt.xlabel("Session")
    plt.ylabel("Accuracy (%)")
    title2 = (
        f"FBA training – Accuracy\n"
        f"Task={task}, H={H_deg}°, V={V_field}°, angle_set={angle_set}"
    )
    plt.title(title2)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(x, labels, rotation=45)

    # Print a summary block in console
    print("\n==============================")
    print("Condition:", key)
    print(f"  Number of sessions : {len(sessions_sorted)}")
    print(f"  Threshold (start)  : {thresholds[0]}")
    print(f"  Threshold (last)   : {thresholds[-1]}")
    print(f"  Accuracy  (start)  : {accuracies[0]}")
    print(f"  Accuracy  (last)   : {accuracies[-1]}")
    print("  Files in order:")
    for js in sessions_sorted:
        print("   -", js.get("_filename", "?"))


def main():
    summaries = load_summaries()
    if not summaries:
        return

    groups = group_by_condition(summaries)

    for key, sess in groups.items():
        sess_sorted = sort_sessions(sess)
        plot_condition(key, sess_sorted)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
