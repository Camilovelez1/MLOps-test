import os
import argparse
import wandb
from constants import project_name, experiment_id, log_metrics, read_csv
from dotenv import load_dotenv
import pickle

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--IdExecution", type=str, help="ID of the execution", required=True
)
parser.add_argument(
    "--dryRun",
    action=argparse.BooleanOptionalAction,
    help="Dry Run.",
    required=False,
)
args = parser.parse_args()

executionId = args.IdExecution
dryRun = args.dryRun

def predict(trained_model, test_dataset):
    X_train = test_dataset.drop("label", axis=1)
    y_train = test_dataset["label"]
    y_pred = trained_model.predict(X_train)
    return y_train, y_pred


def test_and_log(experiment_id=1):
    with wandb.init(
        project=project_name,
        name=f"Evaluate Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="eval-model",
    ) as run:
        data = run.use_artifact("test_data:latest")
        data_dir = data.download()

        test_dataset = read_csv(data_dir, "test_data.csv")

        model_artifact = run.use_artifact("trained_model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pkl")

        with open(model_path, "rb") as f:
            trained_model = pickle.load(f)

        y_train, y_pred = predict(trained_model, test_dataset)
        log_metrics(run, y_train, y_pred, dryRun)


test_and_log(experiment_id)
