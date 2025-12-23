import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data():
    # 1. Definisikan Path (Lokasi File)
    # Sesuaikan dengan struktur folder yang sudah kita buat
    raw_data_path ="dataset_raw/telco_churn_raw.csv"
    output_path = 'telco_churn_preprocessing/train_processed.csv'

    # Cek apakah file dataset ada
    if not os.path.exists(raw_data_path):
        print(f"ERROR: File dataset tidak ditemukan di {raw_data_path}")
        return

    print("--- Memulai Proses Otomatisasi ---")
    
    # 2. Load Dataset
    df = pd.read_csv(raw_data_path)
    
    # --- SAMPLING (PENTING) ---
    # Wajib sama dengan notebook agar konsisten dan ringan
    df = df.sample(n=1000, random_state=42)
    print(f"Data dimuat & dipotong menjadi {df.shape[0]} baris.")

    # 3. Cleaning (TotalCharges)
    # Mengubah string ke angka, handling error, dan isi NaN dengan 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 4. Dropping (Hapus customerID)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # 5. Encoding (Label Encoder untuk Kategori)
    le = LabelEncoder()
    # Loop otomatis untuk semua kolom bertipe object (teks)
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # 6. Scaling (Standarisasi Angka)
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 7. Simpan Hasil (Saving)
    # Pastikan folder output tersedia
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"SUKSES! Data bersih tersimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data()