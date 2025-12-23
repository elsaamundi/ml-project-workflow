import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # Load Data (Path ini udah bener buat Docker)
    data_path = 'telco_churn_preprocessing/train_processed.csv'
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {data_path}")
        return

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BAGIAN INI DIHAPUS BIAR GAK CRASH DI GITHUB ---
    # mlflow.set_experiment("Eksperimen_CI_Docker") 
    # ---------------------------------------------------

    mlflow.sklearn.autolog()

    # start_run() akan otomatis pakai eksperimen yang disiapin GitHub Action
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {acc}")
        
        mlflow.sklearn.log_model(model, "model", registered_model_name="model_random_forest")

if __name__ == "__main__":
    main()