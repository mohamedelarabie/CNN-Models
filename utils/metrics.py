"""
metrics.py

A reusable classification evaluation logger using TorchMetrics.

Features:
    - Calculate classification metrics using TorchMetrics
    - Save metrics after every epoch into CSV
    - Generate training performance curves
    - Support multiple models by model_name

Supported Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - AUROC
"""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC
)



class ClassificationMetricsLogger:
    """
    Classification metrics logger for deep learning experiments.

    This class calculates classification metrics using TorchMetrics,
    stores results for every epoch, and generates final performance
    visualization curves after training.

    Parameters
    ----------
    model_name : str
        Name of the evaluated model.

    num_classes : int
        Number of classes in the classification task.

    device : str or torch.device, optional
        Device used for metric calculation.
        Default: "cuda"

    save_dir : str, optional
        Root directory where experiment results are saved.
        Default: "results"


    Example
    -------
    >>> logger = ClassificationMetricsLogger(
    ...     model_name="LeNet",
    ...     num_classes=10
    ... )

    """



    def __init__(self, model_name, num_classes, device="cuda", save_dir="results"):

        self.model_name = model_name
        self.device = device
        self.output_dir = os.path.join(save_dir, model_name )
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")

        # Initialize TorchMetrics
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        self.auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)



    def calculate_metrics(self, out, target, probabilities):
        """
        Calculate classification metrics.

        Parameters
        ----------
        out : torch.Tensor
            Predicted class indices.
            Shape:
                (number_of_samples,)

        target : torch.Tensor
            Ground truth class labels.
            Shape:
                (number_of_samples,)

        probabilities : torch.Tensor
            Predicted class probabilities after softmax.
            Shape:
                (number_of_samples, num_classes)


        Returns
        -------
        dict
            Dictionary containing calculated metrics.


        Example output
        --------------
        {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.94,
            "f1_score": 0.94,
            "auroc": 0.98
        }

        """


        out = torch.as_tensor(out, dtype=torch.long)
        target = torch.as_tensor(target, dtype=torch.long)
        probabilities = torch.as_tensor(probabilities, dtype=torch.float)


        results = {"accuracy":self.accuracy(out, target).item(),
                   "precision":self.precision(out, target).item(),
                    "recall": self.recall(out, target).item(),
                    "f1_score": self.f1(out, target).item(),
                    "auroc": self.auroc(probabilities, target).item()}
        return results



    def log_epoch_metrics( self, epoch, out, target, probabilities):
       
        """
        Calculate and save metrics for one epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        out : torch.Tensor
            Model predicted labels.

        target : torch.Tensor
            Ground truth labels.

        probabilities : torch.Tensor
            Softmax probabilities.


        Returns
        -------
        dict
            Epoch metrics.


        """

        metrics = self.calculate_metrics( out, target, probabilities)
        metrics["model"] = self.model_name
        metrics["epoch"] = epoch
        self.save_csv(metrics)
        return metrics



    def save_csv( self, metrics):
        """
        Append epoch metrics into CSV file.

        Parameters
        ----------
        metrics : dict
            Dictionary containing model performance values.

        """

        file_exists = os.path.exists(self.csv_path)

        with open( self.csv_path, "a", newline="") as file:
            writer = csv.DictWriter(file,fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)



    def generate_training_curves(self):
        """
        Generate final metric curves after training.

        Generated files:
            - accuracy_curve.png
            - precision_curve.png
            - recall_curve.png
            - f1_score_curve.png
            - auroc_curve.png
            - all_metrics_overview.png

        """

        history = pd.read_csv(self.csv_path)
        metrics = ["accuracy", "precision", "recall", "f1_score", "auroc"]
        for metric in metrics:
            self.plot_metric( history, metric)
        self.plot_all_metrics( history)



    def plot_metric(self, history, metric):
        """
        Plot one metric over training epochs.

        Parameters
        ----------
        history : pandas.DataFrame
            Training history loaded from CSV.

        metric : str
            Metric name to plot.

        """
        epochs = history["epoch"]
        values = history[metric]
        best_epoch = epochs[values.idxmax()]
        best_value = values.max()

        plt.figure(figsize=(10,6),dpi=300)

        plt.plot(epochs, values, marker="o", linewidth=3, label=metric.upper())
        plt.scatter(best_epoch, best_value, color="red", label=f"Best {metric.upper()}", zorder=5)
        plt.annotate(f"Best {metric.upper()}: {best_value:.4f}\nEpoch: {best_epoch}",
                    xy=(best_epoch, best_value),xytext=(best_epoch, best_value),textcoords="offset points",
                    ha="right", va="bottom")
        plt.title(f"{self.model_name} - {metric.upper()} Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid( alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig( os.path.join(self.output_dir, f"{metric}_curve.png"), bbox_inches="tight")
        plt.close()



    def plot_all_metrics(self, history):
        """
        Plot all metrics in one figure.

        Parameters
        ----------
        history : pandas.DataFrame
            Training history.

        """

        metrics = ["accuracy", "precision", "recall", "f1_score", "auroc"]
        plt.figure(figsize=(12,7), dpi=300)
        for metric in metrics:
            plt.plot( history["epoch"], history[metric], marker="o", linewidth=3, label=metric.upper())


        plt.title(f"{self.model_name} Performance Overview")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig( os.path.join(self.output_dir, "all_metrics_overview.png"), bbox_inches="tight")
        plt.close()