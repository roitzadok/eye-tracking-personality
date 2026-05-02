#!/usr/bin/env python
"""
Batch-detect saccades/fixations in raw EyeLink CSV trials
using Engbert & Kliegl’s velocity-based detector.

Requires:
    • engbert_detector_corrected.py (same folder or on PYTHONPATH)
    • pandas, numpy, matplotlib, glob
"""

import os
import glob
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engbert_func_fixed_deg import EngbertDetector   # ← new import

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INPUT_CSV_DIR   = r".\inputs"
OUTPUT_EVENT_DIR = "output_events_10_test_deg"

SAMPLING_RATE   = 1000        # Hz
TIMESTAMP_UNIT  = "ms"        # column 0 is in ms or s

TIMESTAMP_COL = "0"
L_EYE_X_COL   = "1"
L_EYE_Y_COL   = "2"
R_EYE_X_COL   = "4"
R_EYE_Y_COL   = "5"

MIN_EVENT_DURATION = 50       # ms  (not used by detector, only by post-filter)
PAD_BLINKS_MS      = 0
LAMBDA_PARAM       = 6
DERIV_WINDOW_SIZE  = 5

VIEWER_DISTANCE_CM = 96
PIXEL_SIZE_CM      = 0.0277    # cm/px

# ------------------------------------------------------------------
# Event detection helper
# ------------------------------------------------------------------

def detect_events_engbert(
    df_trial: pd.DataFrame,
    sampling_rate: int = SAMPLING_RATE,
    missing_value: float = np.nan,
    min_event_duration: int = MIN_EVENT_DURATION,
    pad_blinks_ms: int = PAD_BLINKS_MS,
    lambda_param: float = LAMBDA_PARAM,
    deriv_window_size: int = DERIV_WINDOW_SIZE,
    viewer_distance_cm: float = VIEWER_DISTANCE_CM,
    pixel_size_cm: float = PIXEL_SIZE_CM,
):
    """
    Run Engbert’s detector on one trial (already averaged across eyes).
    Returns two DataFrames: saccades and fixations.
    """
    t = df_trial["time_sec"].to_numpy()
    x = df_trial["x_avg_pix"].to_numpy()
    y = df_trial["y_avg_pix"].to_numpy()

    # detector ------------------------------------------------------
    detector = EngbertDetector(
        missing_value      = missing_value,
        min_event_duration = min_event_duration,
        pad_blinks_ms      = pad_blinks_ms,
        lambda_param       = lambda_param,
        deriv_window_size  = deriv_window_size,
    )
    detector._sr = sampling_rate

    labels, metadata = detector.detect(
        t, x, y,
        viewer_distance_cm = viewer_distance_cm,
        pixel_size_cm      = pixel_size_cm,
    )

    # quick diagnostics --------------------------------------------
    print("DEBUG  unique labels:", np.unique(labels))
    print("DEBUG  metadata:", metadata)

    dt = 1.0 / sampling_rate
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    velocity = np.sqrt(vx**2 + vy**2)
    print(
        "DEBUG  velocity px/s  min {:.1f}  max {:.1f}  mean {:.1f}".format(
            np.nanmin(velocity), np.nanmax(velocity), np.nanmean(velocity)
        )
    )

    # group contiguous samples --------------------------------------
    MIN_SACCADE_DURATION = 10   # ms
    MIN_FIXATION_DURATION = 50  # ms

    sacc, fix = [], []
    current = labels[0]
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != current:
            end = i - 1
            dur = (end - start + 1) * (1000 / sampling_rate)
            ev = {
                "onset": start,
                "offset": end,
                "duration_ms": dur,
                "start_time_sec": t[start],
                "end_time_sec": t[end],
                "x_start": x[start],
                "y_start": y[start],
                "x_end": x[end],
                "y_end": y[end],
            }
            if current == "saccade"  and dur >= MIN_SACCADE_DURATION: sacc.append(ev)
            if current == "fixation" and dur >= MIN_FIXATION_DURATION: fix.append(ev)
            current = labels[i]
            start = i

    # last segment
    end = len(labels) - 1
    dur = (end - start + 1) * (1000 / sampling_rate)
    ev = {
        "onset": start, "offset": end, "duration_ms": dur,
        "start_time_sec": t[start], "end_time_sec": t[end],
        "x_start": x[start], "y_start": y[start],
        "x_end": x[end], "y_end": y[end],
    }
    if current == "saccade"  and dur >= MIN_SACCADE_DURATION: sacc.append(ev)
    if current == "fixation" and dur >= MIN_FIXATION_DURATION: fix.append(ev)

    print("DEBUG  saccades:", len(sacc), "  fixations:", len(fix))
    return pd.DataFrame(sacc), pd.DataFrame(fix)

# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Engbert detection …")
    os.makedirs(OUTPUT_EVENT_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(INPUT_CSV_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    processed, errors, all_events = 0, 0, []

    for path in csv_files:
        fname = os.path.basename(path)
        print(f"\nProcessing {fname}")
        try:
            df = pd.read_csv(path)
            if df.empty:
                print("  empty → skip")
                continue

            # numeric & time ---------------------------------------
            for c in [TIMESTAMP_COL, L_EYE_X_COL, L_EYE_Y_COL, R_EYE_X_COL, R_EYE_Y_COL]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=[TIMESTAMP_COL], inplace=True)
            df["time_sec"] = df[TIMESTAMP_COL] / (1000 if TIMESTAMP_UNIT == "ms" else 1.0)

            # average eyes -----------------------------------------
            df["x_avg_pix"] = df[[L_EYE_X_COL, R_EYE_X_COL]].mean(axis=1)
            df["y_avg_pix"] = df[[L_EYE_Y_COL, R_EYE_Y_COL]].mean(axis=1)
            print("  NaNs in x_avg_pix:", df["x_avg_pix"].isna().sum())

            # detect events ----------------------------------------
            sacc_df, fix_df = detect_events_engbert(df)

            # save per-trial CSVs ----------------------------------
            base = os.path.splitext(fname)[0]
            sacc_df.to_csv(os.path.join(OUTPUT_EVENT_DIR, f"{base}_saccades.csv"), index=False)
            fix_df.to_csv(os.path.join(OUTPUT_EVENT_DIR, f"{base}_fixations.csv"), index=False)
            print(f"  saved {len(sacc_df)} sacc & {len(fix_df)} fix")

            # accumulate for combined file -------------------------
            for df_ev, tag in ((sacc_df, "saccade"), (fix_df, "fixation")):
                if not df_ev.empty:
                    df_ev["trial"] = base
                    df_ev["event_type"] = tag
                    all_events.append(df_ev)

            # quick plot -------------------------------------------
            t = df["time_sec"].to_numpy()
            x = df["x_avg_pix"].to_numpy()
            y = df["y_avg_pix"].to_numpy()
            dt = 1.0 / SAMPLING_RATE
            vel = np.sqrt(np.gradient(x, dt)**2 + np.gradient(y, dt)**2)

            fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            ax[0].plot(t, x, label="x_avg_pix")
            ax[0].plot(t, y, label="y_avg_pix")
            ax[0].set_ylabel("px")
            ax[0].set_title("Raw gaze")
            ax[0].legend()

            ax[1].plot(t, vel, color="magenta", label="Velocity (px/s)")
            ax[1].set_ylabel("px/s")
            ax[1].set_title("Velocity")
            ax[1].legend()

            for ev in sacc_df.itertuples():
                ax[2].axvspan(ev.start_time_sec, ev.end_time_sec, color="red", alpha=0.3)
            for ev in fix_df.itertuples():
                ax[2].axvspan(ev.start_time_sec, ev.end_time_sec, color="blue", alpha=0.3)
            ax[2].set_ylim(0, 1)
            ax[2].set_title("Events (red = saccade, blue = fixation)")
            ax[2].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_EVENT_DIR, f"{base}_combined.png"))
            plt.close()

            processed += 1

        except Exception as e:
            print("  ERROR:", e)
            traceback.print_exc()
            errors += 1

    # combined file -----------------------------------------------
    if all_events:
        pd.concat(all_events).to_csv(
            os.path.join(OUTPUT_EVENT_DIR, "combined_events.csv"), index=False
        )
        print("Combined events saved")
    print(f"\nDone. OK {processed}  Errors {errors}")
