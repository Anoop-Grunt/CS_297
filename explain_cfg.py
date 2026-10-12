from __future__ import annotations

import csv
import hashlib
import math
import pickle
import random
from pathlib import Path

import angr
import torch
from captum.attr import IntegratedGradients
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


DATA_ROOT = Path("data/cic_dgg_2025/cfgs")

CHECKPOINT = Path(
    "cfg_gcn_requested_all_usable_1261_epochs_100_best.pt"
)

OUTPUT_DIR = Path("cfg_explanations")

SAMPLES_PER_CLASS = 2
MAX_NODES = 2_000
OPCODE_FEATURES = 64
HIDDEN_FEATURES = 64
TOP_NODES = 20
IG_STEPS = 32
SEED = 123


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
            index = opcode_index(
                instruction.mnemonic.lower()
            )
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
            math.log1p(block_size),
            math.log1p(instruction_count),
            math.log1p(in_degree),
            math.log1p(out_degree),
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


def build_graph(
    path: Path,
    label: int,
) -> tuple[Data, list[object]] | None:
    with path.open("rb") as file:
        cfg = pickle.load(file)

    graph = cfg.graph
    nodes = list(graph.nodes)

    if len(nodes) < 2:
        return None

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

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(
            [label],
            dtype=torch.long,
        ),
    )

    return data, nodes


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


class ExplanationModel(nn.Module):
    """
    Wrapper required by Captum.

    Captum varies x while edge_index remains fixed.
    """

    def __init__(
        self,
        model: CFGGCN,
        edge_index: torch.Tensor,
    ) -> None:
        super().__init__()

        self.model = model
        self.edge_index = edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )

        data = Data(
            x=x,
            edge_index=self.edge_index,
            batch=batch,
        )

        return self.model(data)


def instruction_text(node: object) -> str:
    block = getattr(node, "block", None)

    if block is None:
        return ""

    capstone = getattr(block, "capstone", None)
    instructions = getattr(
        capstone,
        "insns",
        [],
    ) or []

    text: list[str] = []

    for instruction in instructions:
        mnemonic = getattr(
            instruction,
            "mnemonic",
            "",
        )

        op_str = getattr(
            instruction,
            "op_str",
            "",
        )

        text.append(
            f"{mnemonic} {op_str}".strip()
        )

    return " ; ".join(text)


def choose_examples() -> list[tuple[Path, int]]:
    benign = list(
        (DATA_ROOT / "DikeDataset").glob("*.pkl")
    )

    malicious = list(
        (
            DATA_ROOT / "pe-machine-learning-dataset"
        ).glob("*.pkl")
    )

    random.shuffle(benign)
    random.shuffle(malicious)

    examples = [
        (path, 0)
        for path in benign[:SAMPLES_PER_CLASS]
    ]

    examples.extend(
        (path, 1)
        for path in malicious[:SAMPLES_PER_CLASS]
    )

    return examples


def explain_file(
    model: CFGGCN,
    path: Path,
    true_label: int,
    device: torch.device,
) -> None:
    result = build_graph(path, true_label)

    if result is None:
        print(f"Skipping unusable CFG: {path}")
        return

    data, nodes = result

    data = data.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(
            Data(
                x=data.x,
                edge_index=data.edge_index,
                batch=torch.zeros(
                    data.num_nodes,
                    dtype=torch.long,
                    device=device,
                ),
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted_label = int(
            logits.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[predicted_label].item()
        )

    explanation_model = ExplanationModel(
        model,
        data.edge_index,
    ).to(device)

    integrated_gradients = IntegratedGradients(
        explanation_model
    )

    baseline = torch.zeros_like(data.x)

    attributions = integrated_gradients.attribute(
        data.x,
        baselines=baseline,
        target=predicted_label,
        n_steps=IG_STEPS,
    )

    node_scores = attributions.abs().sum(
        dim=1
    )

    ranked_nodes = torch.argsort(
        node_scores,
        descending=True,
    )

    safe_name = path.stem[:32]

    output_path = (
        OUTPUT_DIR / f"{safe_name}_explanation.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "node_index",
                "address",
                "score",
                "instructions",
            ]
        )

        rank = 0

        for node_index in ranked_nodes[
            :TOP_NODES
        ].tolist():
            node = nodes[node_index]

            writer.writerow(
                [
                    rank + 1,
                    node_index,
                    hex(
                        getattr(
                            node,
                            "addr",
                            0,
                        )
                    ),
                    float(
                        node_scores[node_index].item()
                    ),
                    instruction_text(node),
                ]
            )

            rank += 1

    print("\n" + "=" * 80)
    print(f"File: {path}")
    print(
        f"True label:      "
        f"{true_label} "
        f"({'benign' if true_label == 0 else 'malicious'})"
    )
    print(
        f"Predicted label: "
        f"{predicted_label} "
        f"({'benign' if predicted_label == 0 else 'malicious'})"
    )
    print(f"Confidence:      {confidence:.4f}")
    print(f"Nodes:           {data.num_nodes}")
    print(f"Edges:           {data.num_edges}")
    print(f"Saved:           {output_path}")

    print("\nMost important CFG nodes:")

    for rank, node_index in enumerate(
        ranked_nodes[:10].tolist(),
        start=1,
    ):
        node = nodes[node_index]

        print(
            f"{rank:02d}. "
            f"addr={hex(getattr(node, 'addr', 0))} "
            f"score={node_scores[node_index].item():.6f}"
        )

        text = instruction_text(node)

        if text:
            print(f"    {text}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = CFGGCN(
        input_features=checkpoint[
            "input_features"
        ],
        hidden_features=checkpoint[
            "hidden_features"
        ],
        num_classes=checkpoint[
            "num_classes"
        ],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Checkpoint test accuracy: "
        f"{checkpoint['test_accuracy']:.3f}"
    )
    print(f"Device: {device}")

    examples = choose_examples()

    for path, label in examples:
        try:
            explain_file(
                model=model,
                path=path,
                true_label=label,
                device=device,
            )

        except Exception as exc:
            print(f"\nFailed to explain {path}")
            print(
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
