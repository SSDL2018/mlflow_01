# register.py
import mlflow
from mlflow.tracking import MlflowClient

# 🔑 SAME tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "diabetes-baseline"
REGISTERED_MODEL_NAME = "diabetes_regressor"


def main():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experiment not found")

    # Get best run by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No runs found")

    best_run = runs[0]
    run_id = best_run.info.run_id
    rmse = best_run.data.metrics["rmse"]

    print(f"Best run: {run_id} (RMSE={rmse:.4f})")

    model_uri = f"runs:/{run_id}/model"

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    version = result.version
    print(f"Registered model version {version}")

    # Promote to Production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    print("Model promoted to Production 🚀")


if __name__ == "__main__":
    main()
