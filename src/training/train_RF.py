import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 🔑 CRITICAL: same tracking URI as UI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "file:./mlruns")

# Create experiment with relative artifact location if it doesn't exist
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiment = client.get_experiment_by_name("diabetes-baseline")
if experiment is None:
    mlflow.create_experiment("diabetes-baseline", artifact_location=artifact_root)

mlflow.set_experiment("diabetes-baseline")



def main():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_options = [
        {"n_estimators": 55, "max_depth": 3},
        {"n_estimators": 2000, "max_depth": 5},
        {"n_estimators": 20, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 20},
    ]

    for params in param_options:
        with mlflow.start_run():
            model = RandomForestRegressor(
                random_state=42,
                **params
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5

            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            # 🔖 Common tags
            mlflow.set_tag("dataset", "sklearn_diabetes")
            mlflow.set_tag("run_type", "baseline")

            # 🔖 Model-specific
            mlflow.set_tag("model_type", "random_forest")

            from mlflow.models.signature import infer_signature

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_train[:5]
            )


            print(f"Logged run with RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
