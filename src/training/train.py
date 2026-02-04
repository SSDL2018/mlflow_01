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


if __name__ == "__main__":
    main()