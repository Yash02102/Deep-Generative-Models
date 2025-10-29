import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path):
    with open(path) as f:
        return json.load(f)


def plot_history(history, title, out_path):
    epochs = range(1, len(history["g_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["g_loss"], label="Generator loss")
    plt.plot(epochs, history["d_loss"], label="Critic loss")
    plt.plot(epochs, history["wasserstein"], label="Wasserstein estimate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--title", default="WGAN-GP Training Curves")
    parser.add_argument("--output", type=Path, default=Path("reports/wgan_gp_history.png"))
    args = parser.parse_args()

    history = load_history(args.history)
    plot_history(history, args.title, args.output)
