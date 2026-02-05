import time
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 🔑 CRITICAL: same tracking URI as UI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-baseline")


def main():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_options = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 7},
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
            rmse = mean_squared_error(y_test, preds, squared=False)

            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            mlflow.set_tag("model_type", "random_forest")

            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            print(f"Logged run with RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
