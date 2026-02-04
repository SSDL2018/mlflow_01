import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #  This sets or creates the experiment
    mlflow.set_experiment("diabetes-baseline")


    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

if __name__ == "__main__":
    main()