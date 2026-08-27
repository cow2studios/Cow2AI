import os
import json
import matplotlib.pyplot as plt


def plot_training_loss(json_path, output_image_path):
    # Load training history from trainer_state.json
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract step counts and corresponding loss values
    log_history = data.get("log_history", [])
    steps = [entry["step"] for entry in log_history if "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]

    # Configure plot dimensions and aesthetics
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, color="#2b5c8f", linewidth=2, label="Pre-training Loss")

    # Label axes and title
    plt.title(
        "750M Base Model Pre-training Loss (Steps 50 - 8500)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    # Annotate significant loss milestones
    if steps and losses:
        plt.annotate(
            f"Start: {losses[0]:.2f}",
            (steps[0], losses[0]),
            textcoords="offset points",
            xytext=(10, -10),
            ha="left",
        )
        plt.annotate(
            f"Step 10000: {losses[-1]:.2f}",
            (steps[-1], losses[-1]),
            textcoords="offset points",
            xytext=(-15, 10),
            ha="right",
        )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Graph saved successfully as: {output_image_path}")


if __name__ == "__main__":
    checkpoint_json = os.path.abspath(
        "models/base_750m/checkpoint-10000/trainer_state.json"
    )
    output_png = os.path.abspath("models/base_750m/loss_graph.png")

    plot_training_loss(checkpoint_json, output_png)
