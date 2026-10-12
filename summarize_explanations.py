from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_DIR = Path("cfg_explanations")
OUTPUT_DIR = Path("explanation_report")

TOP_N = 10


CATEGORIES = {
    "Control flow": {
        "call",
        "ret",
        "retn",
        "jmp",
        "je",
        "jne",
        "jz",
        "jnz",
        "ja",
        "jb",
        "jg",
        "jl",
        "loop",
    },
    "Memory/stack": {
        "mov",
        "movabs",
        "lea",
        "push",
        "pop",
        "xchg",
        "cmpxchg",
    },
    "Arithmetic/logic": {
        "add",
        "sub",
        "imul",
        "idiv",
        "inc",
        "dec",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "sar",
    },
    "Comparison/testing": {
        "cmp",
        "test",
        "bt",
        "bts",
        "btr",
    },
    "System/special": {
        "syscall",
        "int",
        "cpuid",
        "rdtsc",
        "ud2",
    },
}


def extract_mnemonics(instruction_text: str) -> list[str]:
    mnemonics = []

    for instruction in instruction_text.split(" ; "):
        match = re.match(
            r"\s*([a-zA-Z0-9]+)",
            instruction,
        )

        if match:
            mnemonics.append(
                match.group(1).lower()
            )

    return mnemonics


def categorize_instructions(
    rows: list[dict[str, str]],
) -> dict[str, float]:
    category_scores = defaultdict(float)

    for row in rows:
        score = float(row["score"])
        instructions = row["instructions"]

        mnemonics = extract_mnemonics(
            instructions
        )

        matched_categories = set()

        for mnemonic in mnemonics:
            for category, category_mnemonics in CATEGORIES.items():
                if mnemonic in category_mnemonics:
                    matched_categories.add(category)

        for category in matched_categories:
            category_scores[category] += score

    return dict(category_scores)


def heuristic_interpretation(
    category_scores: dict[str, float],
) -> str:
    if not category_scores:
        return (
            "No recognized instruction categories "
            "were found in the attributed nodes."
        )

    ranked = sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    strongest_category, strongest_score = ranked[0]

    interpretations = {
        "Control flow": (
            "The model relied heavily on control-flow "
            "operations such as calls, returns, or branches."
        ),
        "Memory/stack": (
            "The model relied heavily on memory-access "
            "and stack-manipulation instructions."
        ),
        "Arithmetic/logic": (
            "The model relied heavily on arithmetic and "
            "bitwise operations."
        ),
        "Comparison/testing": (
            "The model relied heavily on comparisons "
            "and condition-testing instructions."
        ),
        "System/special": (
            "The model relied heavily on system-level or "
            "special-purpose instructions."
        ),
    }

    interpretation = interpretations[
        strongest_category
    ]

    if len(ranked) > 1:
        secondary = ", ".join(
            category
            for category, _ in ranked[1:3]
        )

        interpretation += (
            f" Secondary attributed categories were: "
            f"{secondary}."
        )

    interpretation += (
        " This describes which instruction regions "
        "influenced the model prediction; it does not "
        "prove that those instructions are malicious."
    )

    return interpretation


def save_node_plot(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
) -> None:
    rows = rows[:TOP_N]
    rows = list(reversed(rows))

    labels = [
        f"{row['rank']}: {row['address']}"
        for row in rows
    ]

    scores = [
        float(row["score"])
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        labels,
        scores,
        color="firebrick",
    )

    axis.set_title(title)
    axis.set_xlabel(
        "Absolute Integrated Gradients attribution"
    )
    axis.set_ylabel("CFG node")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_category_plot(
    category_scores: dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    ranked = sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return

    labels = [
        category
        for category, _ in ranked
    ]

    scores = [
        score
        for _, score in ranked
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        labels,
        scores,
        color="steelblue",
    )

    axis.set_title(title)
    axis.set_ylabel(
        "Total attribution score"
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing explanation directory: "
            f"{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = sorted(
        INPUT_DIR.glob("*_explanation.csv")
    )

    if not csv_files:
        raise RuntimeError(
            f"No explanation CSV files found in "
            f"{INPUT_DIR}"
        )

    report_path = (
        OUTPUT_DIR / "explanation_report.md"
    )

    with report_path.open("w") as report:
        report.write(
            "# Integrated Gradients Explanation Report\n\n"
        )

        report.write(
            "The scores below are absolute Integrated "
            "Gradients attributions aggregated over the "
            "features of each CFG node. Higher values "
            "indicate greater influence on the model's "
            "selected prediction.\n\n"
        )

        report.write(
            "These explanations describe model reliance, "
            "not causal evidence that an instruction is "
            "malicious.\n\n"
        )

        for csv_path in csv_files:
            with csv_path.open() as file:
                rows = list(
                    csv.DictReader(file)
                )

            if not rows:
                continue

            sample_name = csv_path.stem.replace(
                "_explanation",
                "",
            )

            category_scores = (
                categorize_instructions(rows)
            )

            interpretation = (
                heuristic_interpretation(
                    category_scores
                )
            )

            node_plot_path = (
                OUTPUT_DIR
                / f"{sample_name}_nodes.png"
            )

            category_plot_path = (
                OUTPUT_DIR
                / f"{sample_name}_categories.png"
            )

            save_node_plot(
                rows,
                node_plot_path,
                f"Important CFG Nodes: {sample_name}",
            )

            save_category_plot(
                category_scores,
                category_plot_path,
                f"Instruction Categories: {sample_name}",
            )

            report.write(
                f"## Sample `{sample_name}`\n\n"
            )

            report.write(
                f"![Important CFG nodes]"
                f"({node_plot_path.name})\n\n"
            )

            report.write(
                f"![Instruction categories]"
                f"({category_plot_path.name})\n\n"
            )

            report.write(
                f"**Interpretation:** {interpretation}\n\n"
            )

            report.write(
                "### Top attributed nodes\n\n"
            )

            report.write(
                "| Rank | Address | Score | Instructions |\n"
            )
            report.write(
                "|---:|---|---:|---|\n"
            )

            for row in rows[:TOP_N]:
                instructions = (
                    row["instructions"]
                    .replace("|", "\\|")
                    .replace("\n", " ")
                )

                report.write(
                    f"| {row['rank']} "
                    f"| `{row['address']}` "
                    f"| {float(row['score']):.6f} "
                    f"| `{instructions}` |\n"
                )

            report.write("\n")

    print(
        f"Processed {len(csv_files)} explanation files."
    )
    print(f"Report saved to: {report_path}")
    print(f"Figures saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
