from pathlib import Path
import math

import numpy as np
import pandas as pd


RESULTS_PATH = Path(
    "cfg_strong_perturbation_outputs/strong_perturbation_results.csv"
)


def table(df):
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.2f}%"


def signed(x):
    return "NA" if pd.isna(x) else f"{x:+.4f}"


def number(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find: {RESULTS_PATH.resolve()}"
    )


df = pd.read_csv(RESULTS_PATH)

# Derived metrics
df["prediction_flipped"] = (
    df["clean_prediction"]
    != df["perturbed_prediction"]
)

df["clean_correct"] = (
    df["clean_prediction"]
    == df["true_label"]
)

df["perturbed_correct"] = (
    df["perturbed_prediction"]
    == df["true_label"]
)

df["clean_correct_to_wrong"] = (
    df["clean_correct"]
    & ~df["perturbed_correct"]
)

df["malware_to_benign"] = (
    (df["true_label"] == 1)
    & (df["clean_prediction"] == 1)
    & (df["perturbed_prediction"] == 0)
)

df["malicious_probability_change"] = (
    df["perturbed_probability_malicious"]
    - df["clean_probability_malicious"]
)

df["confidence_change"] = (
    df["perturbed_confidence"]
    - df["clean_confidence"]
)

if "edge_retention_fraction" not in df.columns:
    df["edge_retention_fraction"] = np.nan

if "perturbed_num_components" not in df.columns:
    df["perturbed_num_components"] = np.nan

if "perturbed_largest_component_fraction" not in df.columns:
    df["perturbed_largest_component_fraction"] = np.nan


# Clean baseline: one row per graph
clean = (
    df.sort_values(
        ["test_index", "fraction", "repeat"]
    )
    .drop_duplicates("test_index")
)

clean_accuracy = clean["clean_correct"].mean()

benign = clean["true_label"] == 0
malware = clean["true_label"] == 1

clean_benign_accuracy = (
    clean.loc[benign, "clean_correct"].mean()
)

clean_malware_accuracy = (
    clean.loc[malware, "clean_correct"].mean()
)


print("\n" + "=" * 80)
print("CFG PERTURBATION EXPERIMENT SUMMARY")
print("=" * 80)

print("\nEXPERIMENT OVERVIEW")
print("-" * 80)
print(f"Total result rows:       {len(df)}")
print(f"Unique test graphs:      {df['test_index'].nunique()}")
print(f"Perturbation levels:     {df['fraction'].nunique()}")
print(f"Repeats per level:       {df['repeat'].nunique()}")
print(f"Clean accuracy:          {pct(clean_accuracy)}")
print(f"Clean benign accuracy:   {pct(clean_benign_accuracy)}")
print(f"Clean malware accuracy:  {pct(clean_malware_accuracy)}")
print(f"Strongest perturbation:  {df['fraction'].max():.0%}")


summary_rows = []

for fraction, group in df.groupby(
    "fraction",
    sort=True,
):
    malware_clean = (
        (group["true_label"] == 1)
        & (group["clean_prediction"] == 1)
    )

    per_graph_flip_rate = (
        group.groupby("test_index")[
            "prediction_flipped"
        ].mean()
    )

    # Prediction entropy across repeated perturbations.
    # Higher values mean repeated perturbations disagree more.
    p_malicious = (
        group.groupby("test_index")[
            "perturbed_prediction"
        ]
        .mean()
        .clip(1e-9, 1 - 1e-9)
    )

    entropy = -(
        p_malicious * np.log(p_malicious)
        + (1 - p_malicious)
        * np.log(1 - p_malicious)
    ).mean()

    summary_rows.append({
        "Removed edges": f"{fraction:.0%}",
        "N": len(group),
        "Edge retained": pct(
            group["edge_retention_fraction"].mean()
        ),
        "Prediction flips": pct(
            group["prediction_flipped"].mean()
        ),
        "Correct -> wrong": pct(
            group["clean_correct_to_wrong"].mean()
        ),
        "Malware -> benign": pct(
            group["malware_to_benign"].sum()
            / max(1, malware_clean.sum())
        ),
        "Delta P(malware)": signed(
            group[
                "malicious_probability_change"
            ].mean()
        ),
        "Delta confidence": signed(
            group["confidence_change"].mean()
        ),
        "Mean components": number(
            group["perturbed_num_components"].mean()
        ),
        "Largest component": pct(
            group[
                "perturbed_largest_component_fraction"
            ].mean()
        ),
        "Mean graph flip rate": pct(
            per_graph_flip_rate.mean()
        ),
        "Repeat entropy": number(entropy),
    })


summary = pd.DataFrame(summary_rows)

print("\nRESULTS BY PERTURBATION LEVEL")
print("-" * 80)
table(summary)


# Find the most sensitive graphs at the strongest perturbation.
max_fraction = df["fraction"].max()
strongest = df[df["fraction"] == max_fraction]

sensitive_graphs = (
    strongest.groupby(
        ["test_index", "path", "true_label"],
        as_index=False,
    )
    .agg(
        flip_rate=(
            "prediction_flipped",
            "mean",
        ),
        probability_change=(
            "malicious_probability_change",
            "mean",
        ),
        confidence_change=(
            "confidence_change",
            "mean",
        ),
        mean_components=(
            "perturbed_num_components",
            "mean",
        ),
    )
    .sort_values(
        "flip_rate",
        ascending=False,
    )
    .head(15)
)

sensitive_graphs["flip_rate"] = (
    sensitive_graphs["flip_rate"].map(pct)
)

sensitive_graphs["probability_change"] = (
    sensitive_graphs["probability_change"].map(signed)
)

sensitive_graphs["confidence_change"] = (
    sensitive_graphs["confidence_change"].map(signed)
)

sensitive_graphs["mean_components"] = (
    sensitive_graphs["mean_components"].map(number)
)

sensitive_graphs = sensitive_graphs.rename(columns={
    "test_index": "Test index",
    "true_label": "True label",
    "flip_rate": "Flip rate",
    "probability_change": "Delta P(malware)",
    "confidence_change": "Delta confidence",
    "mean_components": "Mean components",
})

print(
    f"\nMOST SENSITIVE GRAPHS AT {max_fraction:.0%} EDGE REMOVAL"
)
print("-" * 80)

table(
    sensitive_graphs[
        [
            "Test index",
            "True label",
            "Flip rate",
            "Delta P(malware)",
            "Delta confidence",
            "Mean components",
        ]
    ]
)


# Print the raw flip counts as an additional sanity check.
print("\nRAW FLIP COUNTS")
print("-" * 80)

flip_counts = (
    df.groupby("fraction")
    .agg(
        flipped=(
            "prediction_flipped",
            "sum",
        ),
        total=(
            "prediction_flipped",
            "count",
        ),
    )
    .reset_index()
)

flip_counts["fraction"] = (
    flip_counts["fraction"].map(
        lambda x: f"{x:.0%}"
    )
)

table(flip_counts)
