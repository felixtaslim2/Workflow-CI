import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
path = 'Membangun Model\Concrete Compressive Strength_preprocessing.csv'
df = pd.read_csv(path)

X = df.drop('concrete_compressive_strength', axis=1)
y = df['concrete_compressive_strength']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Autologging
mlflow.sklearn.autolog()

# MLFlow run
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("concrete-compressive-strength_randomforest")

with mlflow.start_run(run_name="experiment_concrete_compressive_strength"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

print("Model training complete and logged to MLflow.")