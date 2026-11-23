from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CHECKPOINT_PATH = Path("cfg_gcn_requested_all_usable_1261_epochs_100_best.pt")
TEST_SPLIT_PATH = Path("cfg_test_split_paths.csv")
OUTPUT_DIR = Path("cfg_explanation_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"

SEED = 42
MAX_NODES = 2_000
OPCODE_FEATURES = 64

# Keep these moderate initially. Explanations are more expensive than ordinary
# predictions because each one requires a backward pass through the model.
PERTURBATION_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
PERTURBATION_REPEATS = 1
MAX_GRAPHS = None  # Set to e.g. 25 for a quick smoke test.
TOP_K = 20


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# CFG loading: this matches the training/perturbation notebooks
# -----------------------------------------------------------------------------

def opcode_index(mnemonic: str) -> int:
    digest = hashlib.sha1(mnemonic.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little") % OPCODE_FEATURES


def node_features(node, graph) -> list[float]:
    features = [0.0] * OPCODE_FEATURES
    block = getattr(node, "block", None)

    if block is not None:
        capstone = getattr(block, "capstone", None)
        instructions = getattr(capstone, "insns", []) or []
        for instruction in instructions:
            features[opcode_index(instruction.mnemonic.lower())] += 1.0

    instruction_addrs = getattr(node, "instruction_addrs", None)
    instruction_count = len(instruction_addrs or [])
    block_size = getattr(node, "size", None) or 0
    in_degree = graph.in_degree(node)
    out_degree = graph.out_degree(node)

    features.extend([
        float(torch.log1p(torch.tensor(block_size))),
        float(torch.log1p(torch.tensor(instruction_count))),
        float(torch.log1p(torch.tensor(in_degree))),
        float(torch.log1p(torch.tensor(out_degree))),
        float(bool(getattr(node, "is_simprocedure", False))),
    ])
    return features


def load_cfg(path: Path, label: int) -> Data | None:
    with path.open("rb") as file:
        cfg = pickle.load(file)

    graph = cfg.graph
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return None

    nodes = nodes[:MAX_NODES]
    node_to_index = {node: index for index, node in enumerate(nodes)}
    features = [node_features(node, graph) for node in nodes]

    edges = []
    for source, target in graph.edges:
        if source in node_to_index and target in node_to_index:
            edges.append([node_to_index[source], node_to_index[target]])

    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        path=str(path),
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

class CFGGCN(nn.Module):
    def __init__(self, input_features: int, hidden_features: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, hidden_features)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_features, num_classes)

    def forward(self, data: Data, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight).relu()
        graph_embedding = global_mean_pool(x, data.batch)
        return self.classifier(graph_embedding)


if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
    )

if not TEST_SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Test split CSV not found: {TEST_SPLIT_PATH.resolve()}"
    )

model = CFGGCN(OPCODE_FEATURES + 5).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

if not isinstance(state_dict, dict):
    raise TypeError("Checkpoint does not contain a PyTorch state_dict.")

state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print(f"Loaded frozen checkpoint: {CHECKPOINT_PATH}")


# -----------------------------------------------------------------------------
# Load only the saved test split
# -----------------------------------------------------------------------------

split_df = pd.read_csv(TEST_SPLIT_PATH)
if "split" in split_df.columns:
    split_df = split_df[split_df["split"].eq("test")].copy()

split_df = split_df.sort_values("test_index").reset_index(drop=True)

if MAX_GRAPHS is not None:
    split_df = split_df.head(MAX_GRAPHS)

test_dataset = []

for row in tqdm(
    split_df.itertuples(index=False),
    total=len(split_df),
    desc="Loading test CFGs",
):
    absolute_path = Path(getattr(row, "absolute_path"))
    relative_path = Path(getattr(row, "path"))
    path = absolute_path if absolute_path.exists() else relative_path

    data = load_cfg(path, int(row.label))
    if data is not None:
        test_dataset.append(data)

if not test_dataset:
    raise RuntimeError("No test CFGs could be loaded.")

print(f"Loaded test CFGs: {len(test_dataset)}")


def single_graph_batch(data: Data) -> Data:
    batch = data.clone()
    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
    return batch.to(device)


@torch.no_grad()
def predict_one(data: Data) -> dict:
    batch = single_graph_batch(data)
    logits = model(batch)
    probabilities = logits.softmax(dim=1)[0]
    return {
        "prediction": int(probabilities.argmax().item()),
        "probability_benign": float(probabilities[0].item()),
        "probability_malicious": float(probabilities[1].item()),
        "confidence": float(probabilities.max().item()),
    }


def unique_undirected_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    pairs = set()
    for source, target in edge_index.t().tolist():
        if source != target:
            pairs.add((min(source, target), max(source, target)))
    return sorted(pairs)


def remove_random_edge_pairs(data: Data, fraction: float, seed: int) -> Data:
    perturbed = data.clone()
    pairs = unique_undirected_edges(perturbed.edge_index)

    if not pairs:
        return perturbed

    remove_count = int(len(pairs) * fraction)
    remove_count = min(remove_count, len(pairs) - 1)

    generator = torch.Generator().manual_seed(seed)
    selected = set(
        torch.randperm(len(pairs), generator=generator)[:remove_count].tolist()
    )
    selected_pairs = {pairs[index] for index in selected}

    keep = []
    for source, target in perturbed.edge_index.t().tolist():
        pair = (min(source, target), max(source, target))
        keep.append(pair not in selected_pairs)

    keep_mask = torch.tensor(
        keep,
        dtype=torch.bool,
        device=perturbed.edge_index.device,
    )
    perturbed.edge_index = perturbed.edge_index[:, keep_mask]
    return perturbed


# -----------------------------------------------------------------------------
# Gradient-based edge explanations
# -----------------------------------------------------------------------------

def edge_explanation(data: Data, target_class: int) -> tuple[dict, float]:
    """
    Compute gradient-based edge saliency for one graph.

    Every directed edge receives a learnable weight initialized to one. The
    absolute gradient of the target logit, multiplied by that edge weight, is
    used as saliency. Reverse directions are then combined into one
    undirected CFG-edge score.
    """
    batch = single_graph_batch(data)
    edge_weight = torch.ones(
        data.edge_index.size(1),
        dtype=data.x.dtype,
        device=device,
        requires_grad=True,
    )

    model.zero_grad(set_to_none=True)
    logits = model(batch, edge_weight=edge_weight)
    target_logit = logits[0, target_class]
    target_logit.backward()

    if edge_weight.grad is None:
        raise RuntimeError("No edge gradients were produced.")

    directed_scores = (
        edge_weight.grad.detach().abs()
        * edge_weight.detach().abs()
    ).cpu().tolist()

    pair_scores = {}
    for index, (source, target) in enumerate(
        data.edge_index.t().detach().cpu().tolist()
    ):
        pair = (min(source, target), max(source, target))
        pair_scores[pair] = pair_scores.get(pair, 0.0) + directed_scores[index]

    total = sum(pair_scores.values())
    if total > 0:
        pair_scores = {
            pair: score / total
            for pair, score in pair_scores.items()
        }

    return pair_scores, float(target_logit.detach().cpu().item())


def top_edges(scores: dict, k: int = TOP_K) -> list[tuple[tuple[int, int], float]]:
    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]


def cosine_similarity(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    if not keys:
        return np.nan

    left_vector = np.array([left.get(key, 0.0) for key in keys])
    right_vector = np.array([right.get(key, 0.0) for key in keys])
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)

    if denominator == 0:
        return np.nan
    return float(np.dot(left_vector, right_vector) / denominator)


def spearman_correlation(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    if len(keys) < 2:
        return np.nan

    left_series = pd.Series({key: left.get(key, 0.0) for key in keys})
    right_series = pd.Series({key: right.get(key, 0.0) for key in keys})
    return float(left_series.corr(right_series, method="spearman"))


def explanation_metrics(clean_scores: dict, perturbed_scores: dict) -> dict:
    clean_top = {
        pair for pair, _ in top_edges(clean_scores)
    }
    perturbed_top = {
        pair for pair, _ in top_edges(perturbed_scores)
    }

    clean_edges = set(clean_scores)
    perturbed_edges = set(perturbed_scores)
    union = clean_edges | perturbed_edges

    clean_vector = np.array([
        clean_scores.get(pair, 0.0)
        for pair in union
    ])
    perturbed_vector = np.array([
        perturbed_scores.get(pair, 0.0)
        for pair in union
    ])

    # Both distributions sum to one, so this is a bounded distance in [0, 1].
    attribution_l1_distance = float(
        0.5 * np.abs(clean_vector - perturbed_vector).sum()
    )

    return {
        "top_k_overlap": len(clean_top & perturbed_top) / max(1, min(len(clean_top), len(perturbed_top))),
        "clean_top_k_removed_fraction": len(clean_top - perturbed_edges) / max(1, len(clean_top)),
        "spearman_rank_correlation": spearman_correlation(clean_scores, perturbed_scores),
        "cosine_similarity": cosine_similarity(clean_scores, perturbed_scores),
        "attribution_l1_distance": attribution_l1_distance,
        "clean_explanation_edge_count": len(clean_scores),
        "perturbed_explanation_edge_count": len(perturbed_scores),
        "clean_top_edges": json.dumps([
            {"source": pair[0], "target": pair[1], "score": score}
            for pair, score in top_edges(clean_scores)
        ]),
        "perturbed_top_edges": json.dumps([
            {"source": pair[0], "target": pair[1], "score": score}
            for pair, score in top_edges(perturbed_scores)
        ]),
    }


# -----------------------------------------------------------------------------
# Run clean/perturbed explanation comparisons
# -----------------------------------------------------------------------------

records = []

for fraction in PERTURBATION_FRACTIONS:
    print(f"Running explanations at {fraction:.0%} edge removal")

    for repeat in range(PERTURBATION_REPEATS):
        for test_index, clean_graph in enumerate(
            tqdm(test_dataset, desc=f"Graphs at {fraction:.0%}, repeat {repeat}", leave=False)
        ):
            clean_prediction = predict_one(clean_graph)
            target_class = clean_prediction["prediction"]
            clean_scores, clean_target_logit = edge_explanation(
                clean_graph,
                target_class,
            )

            perturbed_graph = remove_random_edge_pairs(
                clean_graph,
                fraction=fraction,
                seed=SEED + repeat * 100_000 + test_index,
            )

            perturbed_prediction = predict_one(perturbed_graph)
            perturbed_scores, perturbed_target_logit = edge_explanation(
                perturbed_graph,
                target_class,
            )

            metrics = explanation_metrics(
                clean_scores,
                perturbed_scores,
            )

            records.append({
                "test_index": test_index,
                "path": clean_graph.path,
                "true_label": int(clean_graph.y.item()),
                "fraction": fraction,
                "repeat": repeat,
                "target_class": target_class,
                "clean_prediction": clean_prediction["prediction"],
                "perturbed_prediction": perturbed_prediction["prediction"],
                "prediction_flipped": (
                    clean_prediction["prediction"]
                    != perturbed_prediction["prediction"]
                ),
                "clean_target_logit": clean_target_logit,
                "perturbed_target_logit": perturbed_target_logit,
                "clean_probability_malicious": clean_prediction[
                    "probability_malicious"
                ],
                "perturbed_probability_malicious": perturbed_prediction[
                    "probability_malicious"
                ],
                "clean_confidence": clean_prediction["confidence"],
                "perturbed_confidence": perturbed_prediction["confidence"],
                **metrics,
            })

explanation_df = pd.DataFrame(records)
explanation_df.to_csv(
    OUTPUT_DIR / "clean_perturbed_explanation_results.csv",
    index=False,
)


summary_df = (
    explanation_df.groupby("fraction", as_index=False)
    .agg(
        graphs=("test_index", "nunique"),
        evaluations=("test_index", "size"),
        prediction_flip_rate=("prediction_flipped", "mean"),
        top_k_overlap=("top_k_overlap", "mean"),
        top_k_overlap_std=("top_k_overlap", "std"),
        clean_top_k_removed_fraction=("clean_top_k_removed_fraction", "mean"),
        spearman_rank_correlation=("spearman_rank_correlation", "mean"),
        cosine_similarity=("cosine_similarity", "mean"),
        attribution_l1_distance=("attribution_l1_distance", "mean"),
        clean_target_logit=("clean_target_logit", "mean"),
        perturbed_target_logit=("perturbed_target_logit", "mean"),
    )
)

summary_df.to_csv(
    OUTPUT_DIR / "explanation_comparison_summary.csv",
    index=False,
)

print("\nExplanation summary:")
print(summary_df.to_string(index=False))


# -----------------------------------------------------------------------------
# Save plots
# -----------------------------------------------------------------------------

def save_plot(fig, filename: str):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {path}")
    plt.show()


x = summary_df["fraction"] * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, summary_df["top_k_overlap"] * 100, marker="o", label="Top-k overlap")
ax.plot(x, summary_df["spearman_rank_correlation"] * 100, marker="o", label="Rank correlation")
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Similarity (%)")
ax.set_title("Clean versus perturbed explanation similarity")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_plot(fig, "01_explanation_similarity.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, summary_df["clean_top_k_removed_fraction"] * 100, marker="o", label="Clean top-k edges removed")
ax.plot(x, summary_df["attribution_l1_distance"] * 100, marker="o", label="Attribution distance")
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Percentage")
ax.set_title("Explanation disruption")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_plot(fig, "02_explanation_disruption.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, summary_df["prediction_flip_rate"] * 100, marker="o", label="Prediction flips")
ax.plot(x, summary_df["attribution_l1_distance"] * 100, marker="o", label="Explanation distance")
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Percentage")
ax.set_title("Prediction changes versus explanation changes")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_plot(fig, "03_prediction_vs_explanation_change.png")

print(f"\nSaved results under: {OUTPUT_DIR.resolve()}")
