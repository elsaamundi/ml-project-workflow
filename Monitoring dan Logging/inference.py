import requests
import json

# URL Model Serving (Pastikan serving 'mlflow models serve' jalan di terminal lain)
# Default port MLflow serve biasanya 5000 atau 5002, sesuaikan dengan terminalmu
url = "http://127.0.0.1:5002/invocations"

# Data dummy (contoh) untuk tes prediksi
# Pastikan format kolom ini sesuai dengan format saat training model kamu
data = {
    "dataframe_split": {
        "columns": [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ],
        "data": [
            # Contoh data pelanggan (sesuaikan tipe datanya dengan modelmu)
            [0, 0, 1, 0, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 50.5, 600.0]
        ]
    }
}

headers = {'Content-Type': 'application/json'}

try:
    print(f"Mengirim request ke {url}...")
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    print("\nStatus Code:", response.status_code)
    if response.status_code == 200:
        print("✅ Prediksi Berhasil!")
        print("Hasil:", response.json())
    else:
        print("❌ Gagal!")
        print("Response:", response.text)
        
except Exception as e:
    print(f"\nError: {e}")
    print("Pastikan terminal 'mlflow models serve' sudah jalan!")