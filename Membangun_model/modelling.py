import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

def main():
    # 1. Load Data
    # Pake try-except biar aman path-nya
    try:
        df = pd.read_csv('../preprocessing/telco_churn_preprocessing/train_processed.csv')
    except FileNotFoundError:
        df = pd.read_csv('telco_churn_preprocessing/train_processed.csv')

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup Eksperimen
    mlflow.set_experiment("Eksperimen_Local_Basic")
    mlflow.sklearn.autolog()  # <--- INI WAJIB BIAR OTOMATIS

    # 3. Training
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {acc}")
        
       
        mlflow.sklearn.log_model(model, "model", registered_model_name="model_random_forest")

if __name__ == "__main__":
    main()