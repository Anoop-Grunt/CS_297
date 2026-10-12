import argparse
import logging
import pickle
from pathlib import Path

# Import angr before unpickling so its serialized classes are registered.
import angr  # noqa: F401


logging.getLogger("angr").setLevel(logging.ERROR)


def inspect_object(obj):
    print("Object type:", type(obj))

    if isinstance(obj, dict):
        print("Dictionary keys:", list(obj.keys()))

    if hasattr(obj, "nodes") and hasattr(obj, "edges"):
        print("Graph nodes:", obj.number_of_nodes())
        print("Graph edges:", obj.number_of_edges())

        try:
            node_id, attrs = next(iter(obj.nodes(data=True)))
            print("Example node:", node_id)
            print("Example node attributes:", list(attrs.keys()))
        except StopIteration:
            print("Graph is empty.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_root", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.cfg_root.exists():
        raise FileNotFoundError(args.cfg_root)

    pickle_files = sorted(args.cfg_root.rglob("*.pkl"))

    if not pickle_files:
        raise RuntimeError(f"No pickle files found under {args.cfg_root}")

    files_to_test = pickle_files if args.all else pickle_files[:args.limit]

    print("CFG root:", args.cfg_root.resolve())
    print("Total pickle files:", len(pickle_files))
    print("Testing:", len(files_to_test))

    successful = 0
    failed = []

    for index, path in enumerate(files_to_test, start=1):
        print(f"\n[{index}/{len(files_to_test)}] {path.name}")

        try:
            with path.open("rb") as f:
                obj = pickle.load(f)

            successful += 1
            inspect_object(obj)

        except Exception as exc:
            failed.append((path, repr(exc)))
            print("FAILED:", repr(exc))

    print("\nSummary")
    print("Successfully loaded:", successful)
    print("Failed:", len(failed))

    if failed:
        print("\nFailed files:")
        for path, error in failed[:20]:
            print(f"- {path}: {error}")

        raise RuntimeError("Some CFG pickles could not be loaded.")


if __name__ == "__main__":
    main()
