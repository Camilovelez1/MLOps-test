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
    help="Dry Run",
    required=False,
)
args = parser.parse_args()

executionId = args.IdExecution
dryRun = args.dryRun

# Check if the directory "./model" exists
if not os.path.exists("./models_data"):
    # If it doesn't exist, create it
    os.makedirs("./models_data")


def train(model, trainining_dataset):
    X_train = trainining_dataset.drop("label", axis=1)
    y_train = trainining_dataset["label"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    return y_train, y_pred


def train_and_log(experiment_id=1):
    with wandb.init(
        project=project_name,
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="train-model",
    ) as run:
        data = run.use_artifact("train_data:latest")
        data_dir = data.download()

        training_dataset = read_csv(data_dir, "train_data.csv")

        model_artifact = run.use_artifact("KNN:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "KNN.pkl")
        model_config = model_artifact.metadata

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_train, y_pred = train(model, training_dataset)
        log_metrics(run, y_train, y_pred, dryRun)

        if not dryRun:
            with open("models_data/trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            model_artifact = wandb.Artifact(
                "trained_model",
                type="model",
                description="Trained KNN model.",
                metadata=dict(model_config),
            )
            model_artifact.add_file("models_data/trained_model.pkl")
            wandb.save("trained_model.pkl")
            run.log_artifact(model_artifact)


train_and_log(experiment_id)
