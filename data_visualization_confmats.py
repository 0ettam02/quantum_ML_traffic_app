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

    # Lettura dei file
    pred_filepath = f"{input_dir}/predictions.dat"
    labels_map_filepath = f"{input_dir}/labels_map.json"
    with open(labels_map_filepath, 'r') as f:
        labels_map = json.load(f)
    labels_map = {int(k): v for k, v in labels_map.items()}

    df_pred = pd.read_csv(pred_filepath, sep="\t")
    df_pred["Actual"] = df_pred["Actual"].map(labels_map)
    df_pred["Predicted"] = df_pred["Predicted"].map(labels_map)

    y_test = df_pred["Actual"].tolist()
    y_pred = df_pred["Predicted"].tolist()

    # Plotting della matrice di confusione
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    fontsize = 18
    classes = list(labels_map.values())
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.matshow(cm, cmap='Blues')

    plt.colorbar(im)
    ax.set_xticks(np.arange(len(classes)))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)


    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm.png", dpi=600)
    plt.savefig(f"{output_dir}/cm.pdf", dpi=600)