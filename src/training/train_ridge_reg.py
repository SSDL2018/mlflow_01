import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-baseline")


def main():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alphas = [0.01, 0.1, 1.0, 10.0]

    for alpha in alphas:
        with mlflow.start_run():
            model = Ridge(alpha=alpha)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5

            mlflow.log_param("alpha", alpha)
            mlflow.log_metric("rmse", rmse)


            # 🔖 Common tags 
            mlflow.set_tag("dataset", "sklearn_diabetes")
            mlflow.set_tag("run_type", "baseline")

            # 🔖 Model-specific
            mlflow.set_tag("model_type", "ridge_regression")
            mlflow.set_tag("model_family", "linear")

            from mlflow.models.signature import infer_signature

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5]
            )

            print(f"[Ridge] alpha={alpha}, RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
