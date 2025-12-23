import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- BAGIAN TOKEN (Sama seperti sebelumnya) ---
try:
    # Masukkan token DagsHub jika running lokal
    dagshub.auth.add_app_token("TOKEN_DAGSHUB_KAMU")
except:
    print("Token error atau sudah ter-set env, lanjut.")

dagshub.init(repo_owner='elsaamundi', repo_name='submission-sml-elsa', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/elsaamundi/submission-sml-elsa.mlflow")
mlflow.set_experiment("Eksperimen Churn Prediction Tuning")

def main():
    data_path = 'telco_churn_preprocessing/train_processed.csv'
    if not os.path.exists(data_path):
        print(f"File dataset gak ketemu di: {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("Mulai Hyperparameter Tuning...")

        # 1. Definisi Model
        rf = RandomForestClassifier(random_state=42)

        # 2. Definisi Parameter Grid (INI YANG DIMINTA REVIEWER)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        # 3. Jalankan Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)
        grid_search.fit(X_train, y_train)

        # Ambil parameter terbaik
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print(f"Parameter Terbaik: {best_params}")

        # 4. Log Parameter Terbaik ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "Random Forest (Tuned)")

        # 5. Evaluasi Model Terbaik
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Akurasi Model Terbaik: {accuracy:.4f}")

        # 6. Simpan Model Terbaik
        mlflow.sklearn.log_model(best_model, "model_best_random_forest")

        # --- ARTEFAK (Confusion Matrix dll tetap sama) ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Bersih-bersih file
        if os.path.exists("confusion_matrix.png"):
            os.remove("confusion_matrix.png")

if __name__ == "__main__":
    main()
