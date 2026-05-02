from EyelinkSample import *
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

INPUTS_DIR = r'.\inputs'
OUTPUTS_DIR = r'.\outputs'
INTRO_INPUTS_DIR = os.path.join(INPUTS_DIR, 'introverts')
EXTRO_INPUTS_DIR = os.path.join(INPUTS_DIR, 'extroverts')
INTRO_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, 'introverts')
EXTRO_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, 'extroverts')
SUMMARY_FILE = os.path.join(OUTPUTS_DIR, "summary.csv")
AQ_SCORES_FILE = os.path.join(INPUTS_DIR, "AQ scores.csv")
JASP_FORMATTED_SUMMARY = os.path.join(OUTPUTS_DIR, "summary_jasp_format.csv")


def analyze_folder(input_folder: str, output_folder: str) -> None:
    for f in os.listdir(input_folder):
        complete_path = os.path.join(input_folder, f)
        if os.path.isfile(complete_path):
            eye = AscEyelinkSample(complete_path)
            eye.analyze_trial(os.path.join(output_folder, f))


def load_summary_messages(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)

    # Find all CSV files ending with _summary.csv (recursive)
    csv_files = folder.rglob("*_summary.csv")

    dfs = []

    for f in csv_files:
        # Read CSV
        df = pd.read_csv(f)

        # Filter for msg 1 or 2
        df = df[df["msg"].isin([1, 2])].copy()

        # Add participant column = file name without extension
        df.insert(0, "participant", f.stem.split('_')[0])

        dfs.append(df)

    # Combine all files into one dataframe
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # return empty if no matching files found
    
    # Calculate per-second rates for all count columns
    # total_duration is in milliseconds, convert to seconds
    combined["total_duration_seconds"] = combined["total_duration"] / 1000.0
    
    count_cols = [col for col in combined.columns if col.endswith("_count")]
    for col in count_cols:
        combined[f"{col}_per_second"] = combined[col] / combined["total_duration_seconds"]
    
    return combined


def merge_summary_files():
    introverts = load_summary_messages(INTRO_OUTPUTS_DIR)
    introverts.insert(1, "is_introvert", 1)
    extroverts = load_summary_messages(EXTRO_OUTPUTS_DIR)
    extroverts.insert(1, "is_introvert", 0)
    pd.concat([introverts, extroverts], ignore_index=True).to_csv(os.path.join(OUTPUTS_DIR, "summary.csv"), index=False)


def clean_summary_participant() -> None:
    """
    Convert participant IDs like 's00012' -> 12
    and overwrite summary.csv.
    """

    df = pd.read_csv(SUMMARY_FILE)

    df["participant"] = (
        df["participant"]
            .astype(str)
            .str.replace(r"^s0*", "", regex=True)  # remove 's' and leading zeros
            .astype(int)
    )

    df.to_csv(SUMMARY_FILE, index=False)


def create_brief_summary() -> None:
    """
    Create a CSV with average and median center_dwell_percentage
    grouped by is_introvert and msg.
    """
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "summary.csv"))

    summary = (
        df
            .groupby(["is_introvert", "msg"])["center_dwell_percentage"]
            .agg(
            center_dwell_percentage_mean="mean",
            center_dwell_percentage_median="median"
        )
            .reset_index()
            .sort_values(["is_introvert", "msg"])
    )

    summary.to_csv(os.path.join(OUTPUTS_DIR, "summary_brief.csv"), index=False)


def add_aq_scores_to_summary() -> None:
    """
    Merge 'A Score' from AQ scores.csv into summary.csv using participant number.
    Invalid or missing A Score values will be replaced with 'missing'.
    """

    summary_df = pd.read_csv(SUMMARY_FILE)
    aq_df = pd.read_csv(AQ_SCORES_FILE)

    # Convert participant numbers safely
    aq_df["Participant Number"] = pd.to_numeric(
        aq_df["Participant Number"], errors="coerce"
    )

    # Convert A Score safely
    aq_df["A Score"] = pd.to_numeric(
        aq_df["A Score"], errors="coerce"
    )

    # Keep only rows with valid participant numbers
    aq_df = aq_df.dropna(subset=["Participant Number"])

    aq_df["Participant Number"] = aq_df["Participant Number"].astype(int)

    merged = summary_df.merge(
        aq_df[["Participant Number", "A Score"]],
        left_on="participant",
        right_on="Participant Number",
        how="left"
    )

    merged = merged.drop(columns=["Participant Number"])

    # Replace missing or invalid AQ scores
    merged["A Score"] = merged["A Score"].fillna("missing")

    merged.to_csv(SUMMARY_FILE, index=False)


def plot_center_dwell_by_msg() -> None:
    df = pd.read_csv(SUMMARY_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, msg_value in zip(axes, [1, 2]):
        subset = df[df["msg"] == msg_value]

        for is_introvert, color, label in [
            (0, "blue", "Extrovert"),
            (1, "red", "Introvert"),
        ]:
            group = subset[subset["is_introvert"] == is_introvert]

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.set_xlim(0, 40)
            ax.scatter(
                group["participant"],
                group["center_dwell_percentage"],
                label=label,
                color=color
            )

        ax.set_title(f"msg = {msg_value}")
        ax.set_xlabel("Participant")
        ax.legend()

    axes[0].set_ylabel("Center Dwell Percentage")

    plt.tight_layout()
    plt.show()


def plot_center_dwell_bars() -> None:

    df = pd.read_csv(SUMMARY_FILE)

    results = []

    for msg in [1, 2]:
        for intro in [0, 1]:

            group = df[
                (df["msg"] == msg) &
                (df["is_introvert"] == intro)
            ]["center_dwell_percentage"]

            mean = group.mean()
            sem = group.std(ddof=1) / np.sqrt(len(group))

            results.append({
                "msg": msg,
                "is_introvert": intro,
                "mean": mean,
                "sem": sem
            })

    results = pd.DataFrame(results)

    width = 0.35
    x = np.array([0, 1])

    extro = results[results["is_introvert"] == 0]
    intro = results[results["is_introvert"] == 1]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        x - width/2,
        extro["mean"],
        width,
        yerr=extro["sem"],
        label="Extrovert",
        capsize=5
    )

    ax.bar(
        x + width/2,
        intro["mean"],
        width,
        yerr=intro["sem"],
        label="Introvert",
        capsize=5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Part 1", "Part 2"])
    ax.set_ylabel("Center Dwell (%)")
    ax.set_xlabel("Experiment Part")

    ax.legend()

    plt.tight_layout()
    plt.show()


def run_mixed_model():

    df = pd.read_csv(SUMMARY_FILE)

    # Remove rows with missing AQ score
    df = df[df["A Score"] != "missing"].copy()

    df["A Score"] = df["A Score"].astype(float)

    model = smf.mixedlm(
        "center_dwell_percentage ~ is_introvert * msg + Q('A Score')",
        data=df,
        groups=df["participant"]
    )

    result = model.fit()

    print(result.summary())


def plot_participant_trajectories() -> None:
    df = pd.read_csv(SUMMARY_FILE)

    # Ensure correct ordering
    df = df.sort_values(["participant", "msg"])

    fig, ax = plt.subplots(figsize=(7,5))

    for participant, group in df.groupby("participant"):

        introvert = group["is_introvert"].iloc[0]

        color = "red" if introvert == 1 else "blue"

        ax.plot(
            group["msg"],
            group["center_dwell_percentage"],
            marker="o",
            alpha=0.4,
            color=color
        )

    # Plot group means
    means = (
        df.groupby(["msg", "is_introvert"])["center_dwell_percentage"]
        .mean()
        .reset_index()
    )

    for introvert, color, label in [
        (0, "blue", "Extrovert mean"),
        (1, "red", "Introvert mean")
    ]:
        group = means[means["is_introvert"] == introvert]

        ax.plot(
            group["msg"],
            group["center_dwell_percentage"],
            marker="o",
            linewidth=3,
            color=color,
            label=label
        )

    ax.set_xticks([1,2])
    ax.set_xticklabels(["Part 1","Part 2"])

    ax.set_xlabel("Experiment Part")
    ax.set_ylabel("Center Dwell (%)")

    ax.legend()

    plt.tight_layout()
    plt.show()


def transform_csv_to_jasp_format():
    # Load the original CSV
    df = pd.read_csv(SUMMARY_FILE)

    cols_to_split = [
        "center_dwell_percentage",
        "center_dwell_count",
        "center_dwell_count_per_second",
        "center_dispersion_fixations_percentage",
        "center_dispersion_fixations_count",
        "center_dispersion_fixations_count_per_second",
        "center_dispersion_fixations_duration",
        "saccades_count",
        "saccades_count_per_second",
        "saccades_avg_velocity",
        "saccades_median_velocity",
        "blinks_percentage",
        "blinks_count",
        "blinks_count_per_second",
        "dispersion_fixations_percentage",
        "dispersion_fixations_count",
        "dispersion_fixations_count_per_second",
        "dispersion_fixations_duration"
    ]

    participant_cols = [
        "A Score",
        "is_introvert"
    ]

    # Ensure there is exactly one row per participant/msg combination
    duplicate_rows = df[df.duplicated(subset=["participant", "msg"], keep=False)]
    if not duplicate_rows.empty:
        raise ValueError(
            "Duplicate participant/msg combinations found in summary.csv. "
            "JASP export expects exactly one row per participant and msg. "
            "Please clean summary.csv before transforming."
        )

    # Pivot repeated-measure variables without altering values
    pivot_df = df.pivot(
        index="participant",
        columns="msg",
        values=cols_to_split
    )

    # Flatten multiindex columns
    pivot_df.columns = [f"{col}_{msg}" for col, msg in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Extract participant-level variables
    participant_info = (
        df[["participant"] + participant_cols]
        .drop_duplicates(subset="participant")
    )

    # Merge
    result = pivot_df.merge(participant_info, on="participant", how="left")

    # Reorder columns so is_introvert is second
    cols = list(result.columns)
    cols.remove("is_introvert")
    cols.insert(1, "is_introvert")
    result = result[cols]

    # Save
    result.to_csv(JASP_FORMATTED_SUMMARY, index=False)


def main():
    # analyze_folder(INTRO_INPUTS_DIR, INTRO_OUTPUTS_DIR)
    # analyze_folder(EXTRO_INPUTS_DIR, EXTRO_OUTPUTS_DIR)
    merge_summary_files()
    clean_summary_participant()
    add_aq_scores_to_summary()
    create_brief_summary()
    # plot_center_dwell_by_msg()
    # plot_center_dwell_bars()
    # run_mixed_model()
    # plot_participant_trajectories()
    transform_csv_to_jasp_format()


if __name__ == '__main__':
    main()
