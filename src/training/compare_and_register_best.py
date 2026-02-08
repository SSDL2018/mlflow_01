import mlflow
from mlflow.tracking import MlflowClient

# 🔑 MUST match training + UI
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetes-baseline"
REGISTERED_MODEL_NAME = "diabetes_best_model"

mlflow.set_tracking_uri(TRACKING_URI)

def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

    # 1️⃣ Fetch all baseline runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'baseline'",
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No baseline runs found")

    best_run = runs[0]
    run_id = best_run.info.run_id
    rmse = best_run.data.metrics["rmse"]
    model_type = best_run.data.tags.get("model_type", "unknown")

    print("🏆 Best run found")
    print(f"Run ID     : {run_id}")
    print(f"Model type: {model_type}")
    print(f"RMSE      : {rmse:.4f}")

    model_uri = f"runs:/{run_id}/model"

    # 2️⃣ Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    model_version = result.version
    print(f"📦 Registered model version: {model_version}")

    client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description="""Diabetes regression model trained on the sklearn diabetes dataset.
                This model is automatically selected based on lowest RMSE across
                multiple baseline algorithms (Random Forest, Ridge, SVR).
                """)

    client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=model_version,
    description=f"""Selected for production.
        Model type: {model_type}
        RMSE: {rmse:.4f}
        Selection criteria: lowest validation RMSE among baseline models.
        """)

    # 3️⃣ Promote to Production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model promoted to Production: {REGISTERED_MODEL_NAME} v{model_version}")

    # 4️⃣ Tag the run for traceability
    client.set_tag(run_id, "selected_for_production", "true")
    client.set_tag(run_id, "selection_reason", "lowest_rmse")


if __name__ == "__main__":
    main()
