import os
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from constants import project_name
import argparse
import pandas as pd
from dotenv import load_dotenv
import wandb


load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="ID of the execution")
args = parser.parse_args()

# Check if the directory "./model" exists
if not os.path.exists("data/input_model"):
    # If it doesn't exist, create it
    os.makedirs("data/input_model")

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")


def load_csv_from_folder(folder):
    df = pd.read_csv('card_transdata.csv')
    return df

def create_csv_and_labels(path_folder):
    df = load_csv_from_folder(path_folder)
    return df

def variable_selection(df):
    # Selecciona las primeras tres columnas que deseas estandarizar
    columnas_a_estandarizar = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

    # Crea el objeto StandardScaler
    scaler = StandardScaler()

    # Aplica el escalado a las primeras tres columnas y almacena los resultados en un nuevo DataFrame
    df1 = pd.DataFrame(scaler.fit_transform(df[columnas_a_estandarizar]), columns=columnas_a_estandarizar)

    # Selecciona las columnas restantes del DataFrame original
    columnas_restantes =  ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']
    df2 = df[columnas_restantes]

    # Asegúrate de que los índices estén alineados
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Combina los dos DataFrames
    df_estandarizado = pd.concat([df1, df2], axis=1)

    return df_estandarizado

def create_train_test_data(df_estandarizado):
    X = df_estandarizado[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
    X = sm.add_constant(X)
    y = df_estandarizado['fraud']

    # Divide el conjunto de datos estandarizado en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project=project_name,
        name=f"Load Preproccesed Data ExecId-{args.IdExecution}",
        job_type="load-data",
    ) as run:

        # Load datasets
        X_train, y_train, X_test, y_test = create_train_test_data(
            "data/raw/card_transdata.csv/"
        )

        # Convert to pandas DataFrames for easier CSV handling
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test)
        test_df["label"] = y_test

        # Save datasets to CSV
        train_df.to_csv("data/input_model/train_data.csv", index=False)
        test_df.to_csv("data/input_model/test_data.csv", index=False)

        # 🏺 Create our Artifacts for W&B
        train_data_artifact = wandb.Artifact(
            "train_data",
            type="dataset",
            description="Training data for fraud classification",
            metadata={"source": "custom dataset", "num_samples": len(train_df)},
        )

        test_data_artifact = wandb.Artifact(
            "test_data",
            type="dataset",
            description="Test data for fraud classification.",
            metadata={"source": "custom dataset", "num_samples": len(test_df)},
        )

        # Add CSV files to the artifacts
        train_data_artifact.add_file("data/input_model/train_data.csv")
        test_data_artifact.add_file("data/input_model/test_data.csv")

        # ✍️ Log the artifacts to W&B
        run.log_artifact(train_data_artifact)
        run.log_artifact(test_data_artifact)


load_and_log()