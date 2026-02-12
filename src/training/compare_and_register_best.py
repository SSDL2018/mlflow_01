import os
import mlflow
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# 🔒 Use environment variable for tracking URI
# --------------------------------------------------

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "diabetes-baseline"
REGISTERED_MODEL_NAME = "diabetes_best_model"

mlflow.set_tracking_uri(TRACKING_URI)
artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "file:./mlruns")


def main():
    print(f"📍 Using MLflow tracking URI: {TRACKING_URI}")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"❌ Experiment '{EXPERIMENT_NAME}' not found")

    # 1️⃣ Fetch best baseline run (lowest RMSE)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'baseline'",
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError("❌ No baseline runs found")

    best_run = runs[0]
    run_id = best_run.info.run_id
    rmse = best_run.data.metrics.get("rmse")
    model_type = best_run.data.tags.get("model_type", "unknown")

    print("\n🏆 Best run selected")
    print(f"Run ID     : {run_id}")
    print(f"Model type : {model_type}")
    print(f"RMSE       : {rmse:.4f}")

    # 2️⃣ Register model
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    model_version = result.version

    print(f"\n📦 Registered model version: {model_version}")

    # 3️⃣ Add descriptions
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            "Diabetes regression model trained on sklearn diabetes dataset. "
            "Auto-selected based on lowest RMSE."
        )
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        description=(
            f"Selected for production.\n"
            f"Model type: {model_type}\n"
            f"RMSE: {rmse:.4f}\n"
            f"Selection criteria: lowest validation RMSE."
        )
    )

    # 4️⃣ Promote to Production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"\n🚀 Model promoted to Production: {REGISTERED_MODEL_NAME} v{model_version}")

    # 5️⃣ Tag run
    client.set_tag(run_id, "selected_for_production", "true")
    client.set_tag(run_id, "selection_reason", "lowest_rmse")


if __name__ == "__main__":
    main()
