
from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm


# Change this to the location of your already-trained checkpoint.
CHECKPOINT_PATH = Path("cfg_gcn_requested_all_usable_1261_epochs_100_best.pt")

DATA_ROOT = Path("data/cic_dgg_2025/cfgs")
SEED = 42
SAMPLES_PER_CLASS = None  # Set to an integer for a smaller smoke test.
MAX_NODES = 2_000
OPCODE_FEATURES = 64
PERTURBATION_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
PERTURBATION_REPEATS = 3


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


def collect_paths() -> list[tuple[Path, int]]:
    benign = sorted((DATA_ROOT / "DikeDataset").glob("*.pkl"))
    malicious = sorted((DATA_ROOT / "pe-machine-learning-dataset").glob("*.pkl"))

    rng = random.Random(SEED)
    rng.shuffle(benign)
    rng.shuffle(malicious)

    if SAMPLES_PER_CLASS is None:
        selected = [(path, 0) for path in benign]
        selected.extend((path, 1) for path in malicious)
    else:
        selected = [(path, 0) for path in benign[:SAMPLES_PER_CLASS]]
        selected.extend((path, 1) for path in malicious[:SAMPLES_PER_CLASS])

    rng.shuffle(selected)
    return selected


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
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}\n"
        "Set CHECKPOINT_PATH to the saved checkpoint used by the original model."
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


paths = collect_paths()
print(f"Attempting to load {len(paths)} CFGs...")

dataset = []
for path, label in tqdm(paths, desc="Loading CFGs"):
    try:
        data = load_cfg(path, label)
        if data is not None:
            dataset.append(data)
    except Exception as exc:
        print(f"Skipping {path}: {type(exc).__name__}: {exc}")

if len(dataset) < 4:
    raise RuntimeError("Too few usable graphs loaded. Check DATA_ROOT.")

labels = [int(data.y.item()) for data in dataset]
train_indices, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

train_dataset = [dataset[index] for index in train_indices]
test_dataset = [dataset[index] for index in test_indices]
train_paths = {data.path for data in train_dataset}
test_paths = {data.path for data in test_dataset}
assert train_paths.isdisjoint(test_paths)

print("Loaded labels:", {0: labels.count(0), 1: labels.count(1)})
print("Training graphs:", len(train_dataset))
print("Testing graphs:", len(test_dataset))
print("Path overlap between train/test:", len(train_paths & test_paths))


def single_graph_batch(data: Data, target_device=device) -> Data:
    batch = data.clone()
    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
    return batch.to(target_device)


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


clean_records = []
for index, data in enumerate(tqdm(test_dataset, desc="Evaluating clean test graphs")):
    result = predict_one(data)
    clean_records.append({
        "test_index": index,
        "path": data.path,
        "true_label": int(data.y.item()),
        **result,
    })

clean_df = pd.DataFrame(clean_records)
print(clean_df[["true_label", "prediction", "confidence"]].head())
print("Clean test accuracy:", (clean_df.true_label == clean_df.prediction).mean())


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

    remove_count = max(1, int(len(pairs) * fraction))
    remove_count = min(remove_count, len(pairs) - 1) if len(pairs) > 1 else 0
    generator = torch.Generator().manual_seed(seed)
    selected = set(torch.randperm(len(pairs), generator=generator)[:remove_count].tolist())
    selected_pairs = {pairs[index] for index in selected}

    keep = []
    for source, target in perturbed.edge_index.t().tolist():
        pair = (min(source, target), max(source, target))
        keep.append(pair not in selected_pairs)

    keep_mask = torch.tensor(keep, dtype=torch.bool, device=perturbed.edge_index.device)
    perturbed.edge_index = perturbed.edge_index[:, keep_mask]
    return perturbed


def evaluate_perturbations(test_graphs, fractions, repeats):
    records = []
    for fraction in fractions:
        for repeat in range(repeats):
            for test_index, clean_graph in enumerate(test_graphs):
                clean = predict_one(clean_graph)
                perturbed_graph = remove_random_edge_pairs(
                    clean_graph,
                    fraction=fraction,
                    seed=SEED + repeat * 100_000 + test_index,
                )
                perturbed = predict_one(perturbed_graph)
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
                })
    return pd.DataFrame(records)


perturbation_df = evaluate_perturbations(
    test_dataset,
    PERTURBATION_FRACTIONS,
    PERTURBATION_REPEATS,
)

perturbation_df.to_csv("cfg_edge_perturbation_results.csv", index=False)
print(perturbation_df.head())
print("Saved cfg_edge_perturbation_results.csv")


def summarize_perturbations(results: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for fraction, group in results.groupby("fraction"):
        originally_correct = group.clean_prediction.eq(group.true_label)
        malicious_clean = group.true_label.eq(1) & group.clean_prediction.eq(1)
        grouped.append({
            "fraction": fraction,
            "prediction_flip_rate": group.clean_prediction.ne(group.perturbed_prediction).mean(),
            "clean_correct_to_wrong_rate": (
                originally_correct & group.perturbed_prediction.ne(group.true_label)
            ).mean(),
            "malware_evasion_rate": (
                malicious_clean & group.perturbed_prediction.eq(0)
            ).sum() / max(1, malicious_clean.sum()),
            "mean_malicious_probability_change": (
                group.perturbed_probability_malicious
                - group.clean_probability_malicious
            ).mean(),
            "mean_confidence_change": (
                group.perturbed_confidence - group.clean_confidence
            ).mean(),
        })
    return pd.DataFrame(grouped)


summary_df = summarize_perturbations(perturbation_df)
display(summary_df)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(
    summary_df["fraction"] * 100,
    summary_df["prediction_flip_rate"] * 100,
    marker="o",
)
axes[0].set_title("Prediction flips")
axes[0].set_xlabel("Removed edge pairs (%)")
axes[0].set_ylabel("Graphs flipped (%)")

axes[1].plot(
    summary_df["fraction"] * 100,
    summary_df["malware_evasion_rate"] * 100,
    marker="o",
    color="darkred",
)
axes[1].set_title("Malware-to-benign flips")
axes[1].set_xlabel("Removed edge pairs (%)")
axes[1].set_ylabel("Evasion rate (%)")

axes[2].plot(
    summary_df["fraction"] * 100,
    summary_df["mean_malicious_probability_change"],
    marker="o",
    label="P(malware)",
)
axes[2].plot(
    summary_df["fraction"] * 100,
    summary_df["mean_confidence_change"],
    marker="o",
    label="Confidence",
)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title("Mean score changes")
axes[2].set_xlabel("Removed edge pairs (%)")
axes[2].set_ylabel("Perturbed − clean")
axes[2].legend()

plt.tight_layout()
plt.show()

