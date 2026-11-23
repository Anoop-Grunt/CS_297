from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------

STRONG_DIR = Path("cfg_strong_perturbation_outputs")
EXPLANATION_DIR = Path("cfg_explanation_outputs")
REPORT_PATH = Path("cfg_perturbation_explanation_report.md")

STRONG_SUMMARY_PATH = STRONG_DIR / "strong_perturbation_summary.csv"
STRONG_RESULTS_PATH = STRONG_DIR / "strong_perturbation_results.csv"
CLEAN_RESULTS_PATH = STRONG_DIR / "clean_test_predictions.csv"
EXPLANATION_SUMMARY_PATH = (
    EXPLANATION_DIR / "explanation_comparison_summary.csv"
)
EXPLANATION_RESULTS_PATH = (
    EXPLANATION_DIR / "clean_perturbed_explanation_results.csv"
)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path.resolve()}"
        )


for required_path in [
    STRONG_SUMMARY_PATH,
    STRONG_RESULTS_PATH,
    CLEAN_RESULTS_PATH,
    EXPLANATION_SUMMARY_PATH,
]:
    require(required_path)


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def pct(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{100 * float(value):.{digits}f}%"


def signed(value, digits=4):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):+.{digits}f}"


def decimal(value, digits=4):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        # Fallback if tabulate is not installed.
        return "```text\n" + frame.to_string(index=False) + "\n```"


def first_or_closest(frame, fraction):
    index = (frame["fraction"] - fraction).abs().idxmin()
    return frame.loc[index]


def image_link(path: Path, caption: str) -> str:
    if path.exists():
        return f"![{caption}]({path.as_posix()})"
    return f"*Plot unavailable: `{path.as_posix()}`*"


# -----------------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------------

strong_summary = pd.read_csv(STRONG_SUMMARY_PATH)
strong_results = pd.read_csv(STRONG_RESULTS_PATH)
clean_results = pd.read_csv(CLEAN_RESULTS_PATH)
explanation_summary = pd.read_csv(EXPLANATION_SUMMARY_PATH)


# -----------------------------------------------------------------------------
# Baseline statistics
# -----------------------------------------------------------------------------

clean_results["correct"] = (
    clean_results["prediction"]
    == clean_results["true_label"]
)

clean_accuracy = clean_results["correct"].mean()
benign_mask = clean_results["true_label"].eq(0)
malware_mask = clean_results["true_label"].eq(1)

benign_accuracy = clean_results.loc[
    benign_mask,
    "correct",
].mean()

malware_accuracy = clean_results.loc[
    malware_mask,
    "correct",
].mean()

num_test_graphs = clean_results["test_index"].nunique()
num_benign = int(benign_mask.sum())
num_malware = int(malware_mask.sum())


# -----------------------------------------------------------------------------
# Tables for the report
# -----------------------------------------------------------------------------

strong_table = strong_summary.copy()
strong_table["fraction"] = strong_table["fraction"].map(
    lambda value: f"{100 * value:.0f}%"
)

strong_table = strong_table.rename(columns={
    "fraction": "Removed edge pairs",
    "prediction_flip_rate": "Prediction flips",
    "clean_correct_to_wrong_rate": "Correct to wrong",
    "malware_evasion_rate": "Malware to benign",
    "mean_malicious_probability_change": "Delta P(malware)",
    "mean_confidence_change": "Delta confidence",
    "mean_edge_retention_fraction": "Edges retained",
    "mean_num_components": "Mean components",
    "mean_largest_component_fraction": "Largest component",
    "mean_per_graph_flip_rate": "Mean graph flip rate",
    "prediction_entropy_across_repeats": "Prediction entropy",
})

for column in [
    "Prediction flips",
    "Correct to wrong",
    "Malware to benign",
    "Edges retained",
    "Largest component",
    "Mean graph flip rate",
]:
    if column in strong_table:
        strong_table[column] = strong_table[column].map(pct)

for column in [
    "Delta P(malware)",
    "Delta confidence",
]:
    if column in strong_table:
        strong_table[column] = strong_table[column].map(signed)

for column in [
    "Mean components",
    "Prediction entropy",
]:
    if column in strong_table:
        strong_table[column] = strong_table[column].map(decimal)

strong_table = strong_table[
    [
        "Removed edge pairs",
        "Prediction flips",
        "Correct to wrong",
        "Malware to benign",
        "Edges retained",
        "Mean components",
        "Largest component",
        "Delta confidence",
    ]
]


explanation_table = explanation_summary.copy()
explanation_table["fraction"] = explanation_table["fraction"].map(
    lambda value: f"{100 * value:.0f}%"
)

explanation_table = explanation_table.rename(columns={
    "fraction": "Removed edge pairs",
    "prediction_flip_rate": "Prediction flips",
    "top_k_overlap": "Top-k overlap",
    "spearman_rank_correlation": "Rank correlation",
    "cosine_similarity": "Cosine similarity",
    "attribution_l1_distance": "Attribution distance",
    "clean_top_k_removed_fraction": "Clean top-k removed",
})

for column in [
    "Prediction flips",
    "Top-k overlap",
    "Rank correlation",
    "Cosine similarity",
    "Attribution distance",
    "Clean top-k removed",
]:
    if column in explanation_table:
        explanation_table[column] = explanation_table[column].map(pct)

explanation_table = explanation_table[
    [
        "Removed edge pairs",
        "Prediction flips",
        "Top-k overlap",
        "Rank correlation",
        "Cosine similarity",
        "Attribution distance",
        "Clean top-k removed",
    ]
]


# -----------------------------------------------------------------------------
# Automatic result interpretation
# -----------------------------------------------------------------------------

strongest_fraction = strong_summary["fraction"].max()
strongest_row = first_or_closest(
    strong_summary,
    strongest_fraction,
)

explanation_20 = first_or_closest(
    explanation_summary,
    0.20,
)
explanation_50 = first_or_closest(
    explanation_summary,
    0.50,
)

explanation_10 = first_or_closest(
    explanation_summary,
    0.10,
)


# -----------------------------------------------------------------------------
# Report text
# -----------------------------------------------------------------------------

report = []

report.extend([
    "# CFG Perturbation and Explanation-Stability Study",
    "",
    "**Status:** Preliminary results for discussion and review",
    "",
    "## Executive summary",
    "",
    (
        "I used the already-trained CFG-GCN model to see how much its predictions "
        "and explanations change when CFG edges are removed. The classifier gets "
        f"{pct(clean_accuracy)} accuracy on the {num_test_graphs} held-out CFGs "
        f"({num_benign} benign and {num_malware} malicious). The main result is "
        "that the predictions are fairly stable under moderate changes, while the "
        "gradient-based explanations start changing much earlier."
    ),
    "",
    (
        f"At 20% edge-pair removal, prediction flips occur in "
        f"{pct(explanation_20['prediction_flip_rate'])} of cases, while top-k "
        f"explanation overlap falls to {pct(explanation_20['top_k_overlap'])}, "
        f"rank correlation falls to {pct(explanation_20['spearman_rank_correlation'])}, "
        f"and attribution distance reaches "
        f"{pct(explanation_20['attribution_l1_distance'])}. At 50% removal, "
        f"explanation distance reaches "
        f"{pct(explanation_50['attribution_l1_distance'])}, while prediction "
        f"flips remain at {pct(explanation_50['prediction_flip_rate'])}."
    ),
    "",
    (
        "This is the result I was mainly interested in: explanation instability "
        "seems to show up before prediction instability. At the same time, these "
        "are still random graph-space stress tests, so they should not be treated "
        "as valid, semantics-preserving PE attacks yet."
    ),
    "",
    "## 1. Motivation and research question",
    "",
    (
        "The idea behind this experiment is pretty simple: if a model classifies a "
        "CFG the same way after a small graph change, I would normally expect its "
        "explanation to remain reasonably similar too. If the prediction stays the "
        "same but the important edges change a lot, that could be a useful warning "
        "sign that the model is relying on unstable evidence."
    ),
    "",
    "The study asks:",
    "",
    "1. How sensitive is the frozen classifier to random CFG edge perturbations?",
    "2. How sensitive are its edge-level explanations to the same perturbations?",
    "3. Does explanation instability appear before prediction instability?",
    "",
    "## 2. Relation to prior work",
    "",
    (
        "The explanation part is based on the idea from Li et al. that a graph can be "
        "perturbed while the model still gives the same prediction, even though its "
        "explanation changes substantially. I am testing a similar idea here for "
        "CFG-based malware classification. For now, I used random edge removal as a "
        "first stress test rather than implementing a learned adversarial attack."
    ),
    "",
    (
        "For the malware-security side, I also looked at Zapzalka et al., who used "
        "semantics-preserving node injection against GNN-based attributed CFG malware "
        "classifiers. Their perturbations are much closer to real executable-level "
        "attacks than the random edge deletions used here, so I am using that work "
        "as motivation rather than claiming that my current evasion results are "
        "directly comparable."
    ),
    "",
    "## 3. Experimental setup",
    "",
    "### Dataset and split",
    "",
    (
        "I used CFGs from CIC-DGG-2025. To make sure I was evaluating samples that "
        "were not used for training, I reproduced the original deterministic split "
        "after filtering out the unusable CFG files. The resulting held-out set has "
        f"{num_test_graphs} graphs."
    ),
    "",
    "### Frozen classifier",
    "",
    (
        "I loaded the existing CFG-GCN checkpoint and kept it frozen for all of the "
        "experiments. The model uses hashed opcode counts plus five structural node "
        "features: block size, instruction count, in-degree, out-degree, and whether "
        "the node is a simulated procedure. It has three GCN layers, global mean "
        "pooling, and a two-class classifier."
    ),
    "",
    "### Graph perturbation",
    "",
    (
        "Because the loader stores both directions of each CFG edge, I removed complete "
        "undirected edge pairs instead of removing only one direction. The stronger "
        "stress test covered 1% through 90% edge-pair removal with five random repeats. "
        "These edits were made directly to the graph objects; I did not modify or "
        "regenerate the underlying PE files."
    ),
    "",
    "### Explanation method",
    "",
    (
        "For the first explanation experiment, I used gradient-based edge saliency. I "
        "treated the edge weights as differentiable inputs, calculated the absolute "
        "gradient of the selected target-class logit, and combined the two directions "
        "of each CFG edge. I used the same target class for the clean and perturbed "
        "graphs so that the comparison measured explanation change rather than a "
        "change in the target being explained."
    ),
    "",
    "The explanation comparisons include:",
    "",
    "- Top-k edge overlap",
    "- Spearman rank correlation of edge importance",
    "- Cosine similarity of attribution vectors",
    "- Attribution L1 distance",
    "- Fraction of clean top-k edges removed by perturbation",
    "- Prediction flips for comparison with explanation changes",
    "",
    "## 4. Clean classification results",
    "",
    "| Metric | Result |",
    "|---|---:|",
    f"| Clean accuracy | {pct(clean_accuracy)} |",
    f"| Benign accuracy | {pct(benign_accuracy)} |",
    f"| Malware accuracy | {pct(malware_accuracy)} |",
    f"| Held-out CFGs | {num_test_graphs} |",
    "",
    "The clean model performs better on malicious graphs than benign graphs in this split, "
    "with the largest baseline error contribution coming from benign samples classified "
    "as malicious.",
    "",
    "## 5. Prediction robustness under edge removal",
    "",
    markdown_table(strong_table),
    "",
    (
        f"At the strongest tested perturbation ({strongest_fraction:.0%} edge-pair "
        f"removal), {pct(strongest_row['prediction_flip_rate'])} of evaluations "
        f"flipped. The malware-to-benign rate was only "
        f"{pct(strongest_row['malware_evasion_rate'])}, while the graphs were "
        f"structurally heavily fragmented: the mean number of connected components "
        f"was {decimal(strongest_row['mean_num_components'], 1)} and the largest "
        f"component contained only "
        f"{pct(strongest_row['mean_largest_component_fraction'])} of nodes."
    ),
    "",
    (
        "Thus, the random edge-removal experiment produces increasing classifier "
        "instability only after substantial graph destruction, and the observed flips "
        "are predominantly not malware-evasion flips."
    ),
    "",
    "## 6. Explanation stability results",
    "",
    markdown_table(explanation_table),
    "",
    (
        f"At 10% removal, prediction flips occur in only "
        f"{pct(explanation_10['prediction_flip_rate'])} of evaluations, but top-k "
        f"overlap is already {pct(explanation_10['top_k_overlap'])} and rank "
        f"correlation is {pct(explanation_10['spearman_rank_correlation'])}. At "
        f"20% removal, attribution distance reaches "
        f"{pct(explanation_20['attribution_l1_distance'])}, while prediction flips "
        f"remain at {pct(explanation_20['prediction_flip_rate'])}."
    ),
    "",
    (
        f"At 50% removal, only {pct(explanation_50['top_k_overlap'])} of top-k "
        f"edges overlap and rank correlation falls to "
        f"{pct(explanation_50['spearman_rank_correlation'])}. The attribution "
        f"distance is {pct(explanation_50['attribution_l1_distance'])}, compared "
        f"with a prediction-flip rate of "
        f"{pct(explanation_50['prediction_flip_rate'])}."
    ),
    "",
    (
        "The clearest pattern is that the explanations degrade faster than the "
        "predictions. That makes explanation instability worth investigating as a "
        "possible model-reliability signal, but it does not prove that a sample was "
        "intentionally modified or adversarial."
    ),
    "",
    "## 7. Figures",
    "",
    image_link(
        EXPLANATION_DIR / "plots/01_explanation_similarity.png",
        "Clean versus perturbed explanation similarity",
    ),
    "",
    image_link(
        EXPLANATION_DIR / "plots/02_explanation_disruption.png",
        "Explanation disruption",
    ),
    "",
    image_link(
        EXPLANATION_DIR / "plots/03_prediction_vs_explanation_change.png",
        "Prediction changes versus explanation changes",
    ),
    "",
    image_link(
        STRONG_DIR / "plots/01_prediction_instability.png",
        "Prediction instability under stronger perturbations",
    ),
    "",
    "## 8. Limitations",
    "",
    "- The perturbations are graph-space edits and are not guaranteed to preserve PE validity, program semantics, or executable behavior.",
    "- Edge removal is random rather than optimized against the classifier or explainer.",
    "- The current explanation method is gradient-based edge saliency, not a full GNNExplainer or attack optimized specifically to alter explanations.",
    "- The explanation experiment used one random perturbation per graph and perturbation level, so its reported averages do not estimate repeated-perturbation variance.",
    "- The classifier checkpoint and split are fixed, but the current experiment does not establish performance across multiple seeds or independently trained models.",
    "- Severe perturbation levels, especially 50% and above, create highly fragmented graphs and should be interpreted as stress tests rather than realistic attacks.",
    "",
    "## 9. Future work",
    "",
    "The next step is mainly to get access to the HPC lab so I can run the experiments more comfortably and try more expensive explanation and perturbation methods. I also want to get access to real malicious samples and repeat the experiment with more realistic CFGs instead of relying only on the current dataset and graph-space stress tests.",
    "",
    "## References",
    "",
    "[1] J. Li, M. Pang, Y. Dong, J. Jia, and B. Wang, “Graph Neural Network Explanations are Fragile,” *Proceedings of the 41st International Conference on Machine Learning*, PMLR vol. 235, pp. 28551–28567, 2024. [Publisher page](https://proceedings.mlr.press/v235/li24bd.html).",
    "",
    "[2] D. Zapzalka, S. Salem, and D. Mohaisen, “Semantics-Preserving Node Injection Attacks Against GNN-Based ACFG Malware Classifiers,” *IEEE Transactions on Dependable and Secure Computing*, vol. 22, no. 1, pp. 549–560, 2025. DOI: [10.1109/TDSC.2024.3409410](https://doi.org/10.1109/TDSC.2024.3409410).",
    "",
    "[3] G. Higgins, R. Razavi-Far, H. Shokouhinejad, and A. A. Ghorbani, “Transparent Malware Detection With Granular Assembly Flow Explainability via Graph Neural Networks,” arXiv:2601.14511, 2026. DOI: [10.48550/arXiv.2601.14511](https://doi.org/10.48550/arXiv.2601.14511).",
])


REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
print(f"Saved report to: {REPORT_PATH.resolve()}")
