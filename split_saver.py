
from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# These settings must exactly match the original training notebook.
DATA_ROOT = Path("data/cic_dgg_2025/cfgs")
SEED = 42
TEST_SIZE = 0.30
SAMPLES_PER_CLASS = None
MAX_NODES = 2_000
OPCODE_FEATURES = 64
OUTPUT_PATH = Path("cfg_test_split_paths.csv")


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


def is_usable_cfg(path: Path) -> bool:
    """Match the original notebook's CFG filtering behavior."""
    with path.open("rb") as file:
        cfg = pickle.load(file)

    graph = cfg.graph
    nodes = list(graph.nodes)

    if len(nodes) < 2:
        return False

    nodes = nodes[:MAX_NODES]
    node_to_index = {node: index for index, node in enumerate(nodes)}

    # Run the same feature extraction used during training so feature-related
    # loading errors are handled the same way as in the original notebook.
    _ = [node_features(node, graph) for node in nodes]

    edges = [
        (source, target)
        for source, target in graph.edges
        if source in node_to_index and target in node_to_index
    ]

    return bool(edges)


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


candidate_paths = collect_paths()
usable_paths = []
skipped_paths = []

print(f"Validating {len(candidate_paths)} candidate CFGs...")

for path, label in tqdm(candidate_paths, desc="Validating CFGs"):
    try:
        if is_usable_cfg(path):
            usable_paths.append((path, label))
        else:
            skipped_paths.append((path, label, "too few nodes or no usable edges"))
    except Exception as exc:
        skipped_paths.append((path, label, f"{type(exc).__name__}: {exc}"))

if len(usable_paths) < 4:
    raise RuntimeError("Too few usable CFGs. Check DATA_ROOT and the dataset.")

labels = [label for _, label in usable_paths]
all_indices = list(range(len(usable_paths)))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=labels,
)

test_indices = set(test_indices)
test_ordered_indices = [index for index in all_indices if index in test_indices]

records = []
for test_index, dataset_index in enumerate(test_ordered_indices):
    path, label = usable_paths[dataset_index]
    records.append({
        "path": str(path),
        "absolute_path": str(path.resolve()),
        "label": label,
        "split": "test",
        "dataset_index": dataset_index,
        "test_index": test_index,
    })

test_df = pd.DataFrame(records)
test_df.to_csv(OUTPUT_PATH, index=False)

print(f"Candidate CFG paths: {len(candidate_paths)}")
print(f"Usable CFG paths: {len(usable_paths)}")
print(f"Skipped CFG paths: {len(skipped_paths)}")
print(f"Training paths: {len(usable_paths) - len(test_df)}")
print(f"Testing paths: {len(test_df)}")
print("Test labels:")
print(test_df["label"].value_counts().sort_index())
print(f"Saved exact test split to: {OUTPUT_PATH.resolve()}")

