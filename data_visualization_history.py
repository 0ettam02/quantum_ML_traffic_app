import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        description="Parse the script arguments."
    )


    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input (aka results) directory."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory."
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parsing degli argomenti
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    history_df = pd.read_csv(f"/training_history.csv")

    accuracy = history_df["accuracy"].tolist()
    val_accuracy = history_df["val_accuracy"].tolist()
    loss = history_df["loss"].tolist()
    val_loss = history_df["val_loss"].tolist()

    plt.figure(figsize=(8, 5))
    plt.plot([a*100 for a in accuracy], label="Training Accuracy")
    plt.plot([a*100 for a in val_accuracy], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("%")
    plt.title("Accuracy across epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_history.png", dpi=600)
    plt.savefig(f"{output_dir}/accuracy_history.pdf", dpi=600)

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("")
    plt.title("Loss across epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_history.png", dpi=600)
    plt.savefig(f"{output_dir}/loss_history.pdf", dpi=600)