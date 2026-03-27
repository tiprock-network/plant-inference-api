import matplotlib.pyplot as plt
import math
import pandas as pd
from utils.stats_module import rices_rule
import numpy as np

def plot_histograms(features: list, df: pd.DataFrame):
    n_features = len(features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=False
    )

    # Flatten axes array for easy indexing
    axs = axs.flatten()

    for i, feature in enumerate(features):
        observations = np.array(df[feature].dropna())
        n_observations = len(observations)
        N_BINS = rices_rule(n_observations)

        axs[i].hist(observations, bins=N_BINS, edgecolor="black")
        axs[i].set_title(f"Histogram of {feature}")
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel("Frequency")

    # Turn off unused subplots
    for j in range(n_features, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()