from pathlib import Path
import re

import matplotlib.pyplot as plt


LOG_FILE = Path("full_training.log")
OUTPUT_FILE = Path("training_curves.png")


pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+"
    r"Loss:\s+([0-9.]+)\s+\|\s+"
    r"Train acc:\s+([0-9.]+)\s+\|\s+"
    r"Test acc:\s+([0-9.]+)"
)

epochs = []
losses = []
train_accuracy = []
test_accuracy = []

with LOG_FILE.open() as file:
    for line in file:
        match = pattern.search(line)

        if match is None:
            continue

        epochs.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        train_accuracy.append(float(match.group(3)))
        test_accuracy.append(float(match.group(4)))


if not epochs:
    raise RuntimeError(
        "No epoch records found in full_training.log."
    )


figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

axes[0].plot(
    epochs,
    losses,
    marker="o",
    markersize=3,
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(True)

axes[1].plot(
    epochs,
    train_accuracy,
    marker="o",
    markersize=3,
    label="Train accuracy",
)

axes[1].plot(
    epochs,
    test_accuracy,
    marker="o",
    markersize=3,
    label="Test accuracy",
)

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.0, 1.0)
axes[1].legend()
axes[1].grid(True)

figure.tight_layout()
figure.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)

print(f"Saved plot to {OUTPUT_FILE}")
