from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

experiment_id = 1
project_name = "MLOps-test"


def read_csv(data_dir, filename):
    return pd.read_csv(f"{data_dir}/{filename}")


def log_metrics(wandbRun, y_true, y_pred, dryRun=True):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    if not dryRun:
        print("Logging metrics to wandb...")
        wandbRun.summary["accuracy"] = accuracy
        wandbRun.summary["precision"] = precision
        wandbRun.summary["recall"] = recall
        wandbRun.summary["f1"] = f1
