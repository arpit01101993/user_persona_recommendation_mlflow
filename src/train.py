import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("UserPersonaRecommendations")

with mlflow.start_run():
    df = pd.read_parquet("data/processed/features.parquet") #replace this with real time reader
    X = df.drop(["user_id", "product_id", "label"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")
