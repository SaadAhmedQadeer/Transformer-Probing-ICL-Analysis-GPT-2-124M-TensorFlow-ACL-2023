# probing_classifier.py
# Layer-wise linear probing classifier
# Reproduces Wang et al. ACL 2023 Figure 3

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class LayerwiseProber:
    """
    Trains one linear probe per transformer layer on frozen hidden states.
    Probing accuracy across layers reveals WHERE label information
    is encoded — the core diagnostic of Wang et al. ACL 2023.
    """

    def __init__(self, n_layers=13):
        self.n_layers   = n_layers
        self.probes     = {}
        self.accuracies = []

    def fit_and_evaluate(self, hidden_states_by_layer, labels):
        """
        Parameters
        ----------
        hidden_states_by_layer : list of n_layers arrays,
                                 each shape [n_samples, d_model]
        labels                 : 1-D int array, shape [n_samples]

        Returns
        -------
        accuracies : list of float (mean 5-fold CV accuracy per layer)
        """
        self.accuracies = []

        for layer_idx in range(self.n_layers):
            X = hidden_states_by_layer[layer_idx]   # [n_samples, d_model]

            probe = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                multi_class='multinomial',
                random_state=42
            )

            # 5-fold cross-validation
            scores = cross_val_score(probe, X, labels, cv=5,
                                     scoring='accuracy')
            mean_acc = float(scores.mean())
            std_acc  = float(scores.std())

            self.accuracies.append(mean_acc)
            self.probes[layer_idx] = probe.fit(X, labels)  # fit on full set

            print(f"  Layer {layer_idx:2d}: "
                  f"accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

        return self.accuracies

    def plot_layer_curve(self,
                         save_path='figures/probe_accuracy_corrected.png'):
        """
        Plots and saves the layer-wise probing accuracy curve.
        Automatically creates the figures/ directory if absent.
        """
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(range(self.n_layers), self.accuracies,
                marker='o', linewidth=2, markersize=6,
                color='steelblue', label='Probe Accuracy')

        # Mark the peak layer
        peak_layer = int(np.argmax(self.accuracies))
        peak_acc   = self.accuracies[peak_layer]
        ax.scatter([peak_layer], [peak_acc],
                   s=120, color='orange', zorder=5,
                   label=f'Peak: layer {peak_layer} ({peak_acc:.3f})')

        ax.axvline(x=6, color='red', linestyle='--',
                   alpha=0.7, label='Shallow/Deep boundary (k=6)')

        # Chance level
        n_classes = len(set(range(2)))  # binary SST-2
        ax.axhline(y=0.5, color='grey', linestyle=':',
                   alpha=0.6, label='Chance (0.50)')

        ax.set_xlabel('Layer Index', fontsize=13)
        ax.set_ylabel('Probing Accuracy (5-fold CV)', fontsize=13)
        ax.set_title(
            'Label Information Localization Across Layers\n'
            '(Reproducing Wang et al. ACL 2023, Figure 3)',
            fontsize=13
        )
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.show()
        print(f"  Figure saved → {save_path}")
