# File: prometheus_exporter.py (PERBAIKAN)
import time
import psutil
import random
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# --- METRIK 1: CPU Usage (Gauge) ---
cpu_usage = Gauge('system_cpu_usage', 'Penggunaan CPU dalam persen')

# --- METRIK 2: RAM Usage (Gauge) ---
ram_usage = Gauge('system_ram_usage', 'Penggunaan RAM dalam persen')

# --- METRIK 3: Disk Usage (Gauge) - TAMBAHAN ---
disk_usage = Gauge('system_disk_usage', 'Penggunaan Disk dalam persen')

# --- METRIK 4: Request Count (Counter) ---
request_count = Counter('model_request_count', 'Total Request yang masuk ke model')

# --- METRIK 5: Latency (Histogram/Gauge) - TAMBAHAN ---
# Simulasi seberapa lama model memproses data (ms)
latency = Gauge('model_processing_latency', 'Waktu proses model dalam milidetik')

def collect_metrics():
    # 1. CPU
    cpu = psutil.cpu_percent(interval=1)
    cpu_usage.set(cpu)
    
    # 2. RAM
    ram = psutil.virtual_memory().percent
    ram_usage.set(ram)

    # 3. Disk (TAMBAHAN)
    disk = psutil.disk_usage('/').percent
    disk_usage.set(disk)
    
    # 4 & 5. Simulasi Request & Latency
    if random.random() > 0.3: 
        request_count.inc()
        # Simulasi latency antara 100ms - 500ms
        lat_val = random.uniform(100, 500)
        latency.set(lat_val)
        
    print(f"Metrics Updated | CPU: {cpu}% | RAM: {ram}% | Disk: {disk}%")

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus Exporter (5 Metrik) berjalan di port 8000...")
    while True:
        collect_metrics()
        time.sleep(1)