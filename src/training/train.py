import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():
    # 1. Set / create experiment (MUST be before start_run)
    mlflow.set_experiment("diabetes-baseline")

    # 2. Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Parameter sweep
    param_options = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 7},
    ]

    for params in param_options:
        with mlflow.start_run():
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = mean_squared_error(y_test, preds)**0.5

            # 4. Log everything
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            mlflow.set_tag("model_type", "random_forest")
            mlflow.set_tag("stage", "baseline")

            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )




    # 5. Find best model and register it
    from mlflow.tracking import MlflowClient
    
    best_run = None
    lowest_rmse = float("inf")

    # Loop over runs to find the best one
    client = MlflowClient()
    experiment = client.get_experiment_by_name("diabetes-baseline")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    if runs:
        best_run = runs[0]
        lowest_rmse = best_run.data.metrics["rmse"]
        print(f"Best run_id: {best_run.info.run_id}, RMSE: {lowest_rmse}")

        # Register model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        registered_model_name = "diabetes_regressor"
        mlflow.register_model(model_uri, registered_model_name)

        # Promote to Production stage
        client.transition_model_version_stage(
            name=registered_model_name,
            version=1,  # first version
            stage="Production"
        )
        print(f"Model {registered_model_name} promoted to Production")


if __name__ == "__main__":
    main()