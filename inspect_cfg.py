from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import angr


DATA_ROOT = Path("data/cic_dgg_2025/cfgs")


def public_attributes(obj: Any) -> list[str]:
    """Return accessible non-private attributes without using vars()."""
    return [
        name
        for name in dir(obj)
        if not name.startswith("_")
    ]


def safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception as exc:
        return f"<error reading attribute: {exc}>"


def inspect_cfg(path: Path) -> None:
    print("=" * 80)
    print(f"File: {path}")
    print(f"Dataset: {path.parent.name}")

    with path.open("rb") as file:
        cfg = pickle.load(file)

    print(f"CFG type: {type(cfg)}")
    print(f"CFG module: {type(cfg).__module__}")

    print("\nCFG public attributes:")
    print(public_attributes(cfg))

    model = getattr(cfg, "_model", None)

    if model is None:
        print("\nNo cfg._model found.")
        return

    print("\nModel type:")
    print(type(model))
    print(f"Model module: {type(model).__module__}")

    print("\nModel public attributes:")
    print(public_attributes(model))

    graph = safe_getattr(model, "graph")

    if isinstance(graph, str) or graph is None:
        print("\nCould not access model.graph:")
        print(graph)
        return

    print("\nGraph type:")
    print(type(graph))

    try:
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
    except Exception as exc:
        print(f"Could not count graph contents: {exc}")
        return

    try:
        nodes = list(graph.nodes(data=True))
    except Exception as exc:
        print(f"Could not read graph nodes: {exc}")
        return

    if not nodes:
        print("\nThe graph contains no nodes.")
        return

    node, node_data = nodes[0]

    print("\nFirst graph node:")
    print(f"Node type: {type(node)}")
    print(f"Node module: {type(node).__module__}")
    print(f"Node representation: {node!r}")

    print("\nFirst node public attributes:")
    print(public_attributes(node))

    print("\nFirst node data:")
    print(node_data)

    print("\nCommon node attributes:")

    for attribute_name in [
        "addr",
        "size",
        "block_id",
        "instruction_addrs",
        "name",
        "simprocedure_name",
    ]:
        if hasattr(node, attribute_name):
            value = safe_getattr(node, attribute_name)
            print(f"{attribute_name}: {value!r}")

    block = None

    for attribute_name in ["block", "vex", "irsb"]:
        if hasattr(node, attribute_name):
            candidate = safe_getattr(node, attribute_name)

            if not isinstance(candidate, str) and candidate is not None:
                block = candidate
                print(f"\nFound node.{attribute_name}:")
                print(f"Block type: {type(block)}")
                print(f"Block representation: {block!r}")
                break

    if block is not None:
        print("\nBlock public attributes:")
        print(public_attributes(block))

        for attribute_name in [
            "addr",
            "size",
            "bytes",
            "capstone",
            "vex",
            "statements",
            "instruction_addrs",
        ]:
            if hasattr(block, attribute_name):
                value = safe_getattr(block, attribute_name)

                if attribute_name == "bytes" and isinstance(value, bytes):
                    value = f"<{len(value)} bytes>"

                print(f"{attribute_name}: {value!r}")

        capstone = safe_getattr(block, "capstone")

        if not isinstance(capstone, str) and capstone is not None:
            print("\nCapstone object type:")
            print(type(capstone))

            instructions = safe_getattr(capstone, "insns")

            if not isinstance(instructions, str) and instructions is not None:
                print(f"Number of instructions: {len(instructions)}")

                for index, instruction in enumerate(instructions[:5]):
                    print(f"Instruction {index}: {instruction!r}")

    print("\nFirst five graph nodes:")
    for index, (graph_node, graph_data) in enumerate(nodes[:5]):
        print(f"\nNode {index}:")
        print(f"  type: {type(graph_node)}")
        print(f"  repr: {graph_node!r}")
        print(f"  data: {graph_data}")

    try:
        edges = list(graph.edges(data=True))
        print("\nFirst five graph edges:")

        for index, edge in enumerate(edges[:5]):
            print(f"  Edge {index}: {edge}")

    except Exception as exc:
        print(f"\nCould not read graph edges: {exc}")


def find_example(dataset_name: str) -> Path:
    dataset_dir = DATA_ROOT / dataset_name
    files = sorted(dataset_dir.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(
            f"No pickle files found in {dataset_dir}"
        )

    return files[0]


if __name__ == "__main__":
    benign_path = find_example("DikeDataset")
    malicious_path = find_example("pe-machine-learning-dataset")

    inspect_cfg(benign_path)
    inspect_cfg(malicious_path)
