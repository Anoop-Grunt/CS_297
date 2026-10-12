from __future__ import annotations

import hashlib
import pickle
import random
from collections import Counter
from pathlib import Path

import angr
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm


DATA_ROOT = Path("data/cic_dgg_2025/cfgs")

SEED = 42
SAMPLES_PER_CLASS = None
MAX_NODES = 2_000
EPOCHS = 100
BATCH_SIZE = 2
OPCODE_FEATURES = 64

random.seed(SEED)
torch.manual_seed(SEED)


def opcode_index(mnemonic: str) -> int:
    digest = hashlib.sha1(
        mnemonic.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
    ) % OPCODE_FEATURES


def node_features(node, graph) -> list[float]:
    features = [0.0] * OPCODE_FEATURES

    block = getattr(node, "block", None)

    if block is not None:
        capstone = getattr(block, "capstone", None)
        instructions = getattr(
            capstone,
            "insns",
            [],
        ) or []

        for instruction in instructions:
            mnemonic = instruction.mnemonic.lower()
            index = opcode_index(mnemonic)
            features[index] += 1.0

    instruction_addrs = getattr(
        node,
        "instruction_addrs",
        None,
    )

    instruction_count = len(
        instruction_addrs or []
    )

    block_size = getattr(node, "size", None) or 0
    in_degree = graph.in_degree(node)
    out_degree = graph.out_degree(node)

    features.extend(
        [
            float(torch.log1p(torch.tensor(block_size))),
            float(
                torch.log1p(
                    torch.tensor(instruction_count)
                )
            ),
            float(
                torch.log1p(
                    torch.tensor(in_degree)
                )
            ),
            float(
                torch.log1p(
                    torch.tensor(out_degree)
                )
            ),
            float(
                bool(
                    getattr(
                        node,
                        "is_simprocedure",
                        False,
                    )
                )
            ),
        ]
    )

    return features


def load_cfg(path: Path, label: int) -> Data | None:
    with path.open("rb") as file:
        cfg = pickle.load(file)

    graph = cfg.graph
    nodes = list(graph.nodes)

    if len(nodes) < 2:
        return None

    if len(nodes) > MAX_NODES:
        nodes = nodes[:MAX_NODES]

    node_to_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    features = [
        node_features(node, graph)
        for node in nodes
    ]

    edges: list[list[int]] = []

    for source, target in graph.edges:
        if (
            source in node_to_index
            and target in node_to_index
        ):
            edges.append(
                [
                    node_to_index[source],
                    node_to_index[target],
                ]
            )

    if not edges:
        return None

    x = torch.tensor(
        features,
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()

    reverse_edges = edge_index[[1, 0]]

    edge_index = torch.cat(
        [
            edge_index,
            reverse_edges,
        ],
        dim=1,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(
            [label],
            dtype=torch.long,
        ),
        path=str(path),
    )


class CFGGCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features,
        )

        self.conv2 = GCNConv(
            hidden_features,
            hidden_features,
        )

        self.conv3 = GCNConv(
            hidden_features,
            hidden_features,
        )

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(
            hidden_features,
            num_classes,
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(
            x,
            edge_index,
        ).relu()

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        ).relu()

        x = self.dropout(x)

        x = self.conv3(
            x,
            edge_index,
        ).relu()

        graph_embedding = global_mean_pool(
            x,
            data.batch,
        )

        return self.classifier(graph_embedding)


def collect_paths() -> list[tuple[Path, int]]:
    benign = sorted(
        (
            DATA_ROOT / "DikeDataset"
        ).glob("*.pkl")
    )

    malicious = sorted(
        (
            DATA_ROOT / "pe-machine-learning-dataset"
        ).glob("*.pkl")
    )

    random.shuffle(benign)
    random.shuffle(malicious)

    selected: list[tuple[Path, int]] = []

    selected.extend(
        (path, 0)
        for path in benign[:SAMPLES_PER_CLASS]
    )

    selected.extend(
        (path, 1)
        for path in malicious[:SAMPLES_PER_CLASS]
    )

    random.shuffle(selected)

    return selected


def main() -> None:
    paths = collect_paths()

    print(
        f"Attempting to load {len(paths)} CFGs..."
    )

    dataset: list[Data] = []

    for path, label in tqdm(
        paths,
        desc="Loading CFGs",
    ):
        try:
            data = load_cfg(path, label)

            if data is not None:
                dataset.append(data)

        except Exception as exc:
            print(f"\nSkipping {path}")
            print(
                f"{type(exc).__name__}: {exc}"
            )

    if len(dataset) < 4:
        raise RuntimeError(
            "Too few usable CFGs loaded."
        )

    labels = [
        int(data.y.item())
        for data in dataset
    ]

    print(
        "\nLoaded label distribution:",
        Counter(labels),
    )

    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = [
        dataset[index]
        for index in train_indices
    ]

    test_dataset = [
        dataset[index]
        for index in test_indices
    ]

    print(
        "Training labels:",
        Counter(
            int(data.y.item())
            for data in train_dataset
        ),
    )

    print(
        "Testing labels:",
        Counter(
            int(data.y.item())
            for data in test_dataset
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsable graphs: {len(dataset)}"
    )

    print(
        f"Training graphs: {len(train_dataset)}"
    )

    print(
        f"Testing graphs: {len(test_dataset)}"
    )

    print(f"Device: {device}")

    model = CFGGCN(
        input_features=OPCODE_FEATURES + 5,
        hidden_features=64,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.CrossEntropyLoss()

    if SAMPLES_PER_CLASS is None:
        requested_name = "all"
    else:
        requested_name = str(2 * SAMPLES_PER_CLASS)
    
    checkpoint_path = Path(
        f"cfg_gcn_requested_{requested_name}_"
        f"usable_{len(dataset)}_"
        f"epochs_{EPOCHS}_best.pt"
    )

    best_test_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(
                logits,
                batch.y,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            correct += int(
                (
                    predictions == batch.y
                ).sum().item()
            )

            total += batch.y.numel()

        train_accuracy = correct / total

        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                logits = model(batch)

                predictions = logits.argmax(
                    dim=1
                )

                test_correct += int(
                    (
                        predictions == batch.y
                    ).sum().item()
                )

                test_total += batch.y.numel()

        test_accuracy = test_correct / test_total

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Train acc: {train_accuracy:.3f} | "
            f"Test acc: {test_accuracy:.3f}"
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "test_accuracy": (
                        test_accuracy
                    ),
                    "train_accuracy": (
                        train_accuracy
                    ),
                    "input_features": (
                        OPCODE_FEATURES + 5
                    ),
                    "hidden_features": 64,
                    "num_classes": 2,
                    "max_nodes": MAX_NODES,
                    "samples_per_class_requested": (
                        SAMPLES_PER_CLASS
                    ),
                    "usable_graphs": len(dataset),
                    "train_graphs": (
                        len(train_dataset)
                    ),
                    "test_graphs": (
                        len(test_dataset)
                    ),
                    "seed": SEED,
                },
                checkpoint_path,
            )

            print(
                f"  Saved checkpoint: "
                f"{checkpoint_path}"
            )

    print("\nTraining complete.")
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best test accuracy: "
        f"{best_test_accuracy:.3f}"
    )
    print(
        f"Checkpoint: {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
