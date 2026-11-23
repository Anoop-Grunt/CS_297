
from __future__ import annotations

import hashlib
import math
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm


# Paths and experiment settings
CHECKPOINT_PATH = Path("cfg_gcn_requested_all_usable_1261_epochs_100_best.pt")
TEST_SPLIT_PATH = Path("cfg_test_split_paths.csv")
OUTPUT_DIR = Path("cfg_strong_perturbation_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"

MAX_NODES = 2_000
OPCODE_FEATURES = 64
SEED = 42

# Higher fractions are intentional here: this is a stress test designed to
# expose a transition from stable to structurally and predictively unstable.
PERTURBATION_FRACTIONS = [
    0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90,
]
PERTURBATION_REPEATS = 5


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


class CFGGCN(nn.Module):
    def __init__(self, input_features: int, hidden_features: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, hidden_features)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_features, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
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


# Load and freeze the original model.
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


# Read only the previously saved test paths.
split_df = pd.read_csv(TEST_SPLIT_PATH)

if "split" in split_df.columns:
    split_df = split_df[split_df["split"].eq("test")].copy()

split_df = split_df.sort_values("test_index").reset_index(drop=True)

test_dataset = []
skipped = []

for row in tqdm(split_df.itertuples(index=False), total=len(split_df), desc="Loading test CFGs"):
    path = Path(getattr(row, "absolute_path", getattr(row, "path")))
    label = int(row.label)

    try:
        data = load_cfg(path, label)
        if data is None:
            skipped.append((str(path), "load_cfg returned None"))
        else:
            test_dataset.append(data)
    except Exception as exc:
        skipped.append((str(path), f"{type(exc).__name__}: {exc}"))

if skipped:
    print(f"Warning: skipped {len(skipped)} test CFGs while loading.")

if not test_dataset:
    raise RuntimeError("No test CFGs could be loaded.")

print(f"Loaded test CFGs: {len(test_dataset)}")


def single_graph_batch(data: Data, target_device=device) -> Data:
    batch = data.clone()
    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
    return batch.to(target_device)


@torch.no_grad()
def predict_one(data: Data) -> dict:
    logits = model(single_graph_batch(data))
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


def structural_stats(data: Data) -> dict:
    """Compute connected-component statistics without requiring NetworkX."""
    num_nodes = data.num_nodes
    parent = list(range(num_nodes))
    size = [1] * num_nodes

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left = find(left)
        right = find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    pairs = unique_undirected_edges(data.edge_index)
    for source, target in pairs:
        union(source, target)

    component_sizes = {}
    for node in range(num_nodes):
        root = find(node)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    largest_component = max(component_sizes.values()) if component_sizes else 0
    return {
        "num_nodes": num_nodes,
        "num_edge_pairs": len(pairs),
        "num_components": len(component_sizes),
        "largest_component_fraction": largest_component / max(1, num_nodes),
    }


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


clean_records = []
for test_index, data in enumerate(test_dataset):
    clean_prediction = predict_one(data)
    clean_structure = structural_stats(data)
    clean_records.append({
        "test_index": test_index,
        "path": data.path,
        "true_label": int(data.y.item()),
        **clean_prediction,
        "clean_num_edge_pairs": clean_structure["num_edge_pairs"],
        "clean_num_components": clean_structure["num_components"],
    })

clean_df = pd.DataFrame(clean_records)
clean_df.to_csv(OUTPUT_DIR / "clean_test_predictions.csv", index=False)

print("Clean test accuracy:", (clean_df.true_label == clean_df.prediction).mean())


records = []
for fraction in PERTURBATION_FRACTIONS:
    print(f"Running perturbation fraction: {fraction:.0%}")
    for repeat in range(PERTURBATION_REPEATS):
        for test_index, clean_graph in enumerate(test_dataset):
            clean = predict_one(clean_graph)
            clean_structure = structural_stats(clean_graph)
            perturbed_graph = remove_random_edge_pairs(
                clean_graph,
                fraction=fraction,
                seed=SEED + repeat * 100_000 + test_index,
            )
            perturbed = predict_one(perturbed_graph)
            perturbed_structure = structural_stats(perturbed_graph)

            records.append({
                "test_index": test_index,
                "path": clean_graph.path,
                "true_label": int(clean_graph.y.item()),
                "fraction": fraction,
                "repeat": repeat,
                "clean_prediction": clean["prediction"],
                "perturbed_prediction": perturbed["prediction"],
                "clean_probability_malicious": clean["probability_malicious"],
                "perturbed_probability_malicious": perturbed["probability_malicious"],
                "clean_confidence": clean["confidence"],
                "perturbed_confidence": perturbed["confidence"],
                "clean_num_edge_pairs": clean_structure["num_edge_pairs"],
                "perturbed_num_edge_pairs": perturbed_structure["num_edge_pairs"],
                "perturbed_num_components": perturbed_structure["num_components"],
                "perturbed_largest_component_fraction": perturbed_structure[
                    "largest_component_fraction"
                ],
            })

perturbation_df = pd.DataFrame(records)
perturbation_df["edge_retention_fraction"] = (
    perturbation_df["perturbed_num_edge_pairs"]
    / perturbation_df["clean_num_edge_pairs"].clip(lower=1)
)
perturbation_df["prediction_flipped"] = (
    perturbation_df["clean_prediction"]
    != perturbation_df["perturbed_prediction"]
)
perturbation_df["malware_to_benign"] = (
    (perturbation_df["true_label"] == 1)
    & (perturbation_df["clean_prediction"] == 1)
    & (perturbation_df["perturbed_prediction"] == 0)
)

perturbation_df.to_csv(OUTPUT_DIR / "strong_perturbation_results.csv", index=False)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction, group in results.groupby("fraction"):
        clean_correct = group.clean_prediction.eq(group.true_label)
        malware_clean = group.true_label.eq(1) & group.clean_prediction.eq(1)

        per_graph_flip_rate = (
            group.groupby("test_index")["prediction_flipped"].mean()
        )

        # Entropy is high when repeated perturbations make the classifier
        # disagree about a graph, and low when it predicts consistently.
        prediction_probability = (
            group.groupby("test_index")["perturbed_prediction"].mean()
        ).clip(1e-9, 1 - 1e-9)
        prediction_entropy = -(
            prediction_probability * prediction_probability.apply(math.log)
            + (1 - prediction_probability)
            * (1 - prediction_probability).apply(math.log)
        ).mean()

        rows.append({
            "fraction": fraction,
            "prediction_flip_rate": group.prediction_flipped.mean(),
            "clean_correct_to_wrong_rate": (
                clean_correct & group.perturbed_prediction.ne(group.true_label)
            ).mean(),
            "malware_evasion_rate": (
                group.malware_to_benign.sum() / max(1, malware_clean.sum())
            ),
            "mean_malicious_probability_change": (
                group.perturbed_probability_malicious
                - group.clean_probability_malicious
            ).mean(),
            "mean_confidence_change": (
                group.perturbed_confidence - group.clean_confidence
            ).mean(),
            "mean_edge_retention_fraction": group.edge_retention_fraction.mean(),
            "mean_num_components": group.perturbed_num_components.mean(),
            "mean_largest_component_fraction": (
                group.perturbed_largest_component_fraction.mean()
            ),
            "mean_per_graph_flip_rate": per_graph_flip_rate.mean(),
            "prediction_entropy_across_repeats": prediction_entropy,
        })

    return pd.DataFrame(rows)


summary_df = summarize(perturbation_df)
summary_df.to_csv(OUTPUT_DIR / "strong_perturbation_summary.csv", index=False)
display(summary_df)


def save_figure(fig, filename: str):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {path}")
    return path


# Plot 1: prediction instability.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(summary_df.fraction * 100, summary_df.prediction_flip_rate * 100, marker="o", label="Flip rate")
ax.plot(summary_df.fraction * 100, summary_df.clean_correct_to_wrong_rate * 100, marker="o", label="Correct → wrong")
ax.plot(summary_df.fraction * 100, summary_df.malware_evasion_rate * 100, marker="o", label="Malware → benign")
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Rate (%)")
ax.set_title("Prediction instability under stronger perturbations")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_figure(fig, "01_prediction_instability.png")
plt.show()


# Plot 2: score changes.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(summary_df.fraction * 100, summary_df.mean_malicious_probability_change, marker="o", label="Malicious probability change")
ax.plot(summary_df.fraction * 100, summary_df.mean_confidence_change, marker="o", label="Confidence change")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Perturbed − clean")
ax.set_title("Confidence and malicious-score changes")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_figure(fig, "02_score_changes.png")
plt.show()


# Plot 3: structural instability.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(summary_df.fraction * 100, summary_df.mean_num_components, marker="o")
axes[0].set_xlabel("Removed edge pairs (%)")
axes[0].set_ylabel("Mean connected components")
axes[0].set_title("Graph fragmentation")
axes[0].grid(alpha=0.3)

axes[1].plot(summary_df.fraction * 100, summary_df.mean_largest_component_fraction * 100, marker="o", color="darkorange")
axes[1].set_xlabel("Removed edge pairs (%)")
axes[1].set_ylabel("Largest component (%)")
axes[1].set_title("Largest connected component")
axes[1].grid(alpha=0.3)

plt.tight_layout()
save_figure(fig, "03_structural_instability.png")
plt.show()


# Plot 4: disagreement across repeated perturbations.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(summary_df.fraction * 100, summary_df.mean_per_graph_flip_rate * 100, marker="o", label="Mean per-graph flip rate")
ax.plot(summary_df.fraction * 100, summary_df.prediction_entropy_across_repeats, marker="o", label="Prediction entropy")
ax.set_xlabel("Removed edge pairs (%)")
ax.set_ylabel("Instability")
ax.set_title("Disagreement across random perturbation repeats")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
save_figure(fig, "04_repeat_disagreement.png")
plt.show()
