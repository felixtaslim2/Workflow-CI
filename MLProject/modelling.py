import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

print("Running modelling.py...")

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

df = pd.read_csv("../MLProject/Concrete Compressive Strength_preprocessing.csv")

X = df.drop('concrete_compressive_strength', axis=1)
y = df['concrete_compressive_strength']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42
)

model = RandomForestRegressor(random_state=42)

with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print("Model finished.")
    print("R2 Score:", score)

    mlflow.log_param("test_size", args.test_size)
    mlflow.log_metric("r2_score", score)

    mlflow.sklearn.log_model(model, "model")
