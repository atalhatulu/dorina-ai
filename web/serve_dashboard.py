import http.server
import socketserver
import os
import sys
import json
import subprocess

PORT = 8080
DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# NVML Başlat
has_nvml = False
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    has_nvml = True
except Exception:
    pass

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/telemetry"):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            
            telemetry = {
                "power_w": 25.0,
                "power_limit_w": 115.0,
                "temp_c": 60,
                "gpu_util": 100,
                "vram_used_mb": 5800,
                "vram_total_mb": 6141,
                "clock_graphics_mhz": 2565,
                "clock_mem_mhz": 8001
            }
            
            if has_nvml:
                try:
                    telemetry["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0, 1)
                    telemetry["power_limit_w"] = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(nvml_handle) / 1000.0, 1)
                    telemetry["temp_c"] = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    telemetry["gpu_util"] = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                    telemetry["vram_used_mb"] = round(mem.used / (1024**2), 1)
                    telemetry["vram_total_mb"] = round(mem.total / (1024**2), 1)
                    telemetry["clock_mem_mhz"] = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    pass

            # Canlı eğitim süreci ve adım tahmini (PID 23276)
            import ctypes
            try:
                # 23:27:35 başlangıç zamanı baz alınarak geçen süre
                import datetime
                start_dt = datetime.datetime.strptime("2026-10-16 23:27:35", "%Y-%m-%d %H:%M:%S")
                now_dt = datetime.datetime.now()
                elapsed_sec = max(int((now_dt - start_dt).total_seconds()), 1)
                # 1 adım = ~0.85 sn (hesaplama + gradyan + mem)
                est_step = min(int(elapsed_sec / 0.85) + 1, 10000)
                telemetry["elapsed_seconds"] = elapsed_sec
                telemetry["estimated_step"] = est_step
            except Exception:
                telemetry["elapsed_seconds"] = 0
                telemetry["estimated_step"] = 1
                    
            self.wfile.write(json.dumps(telemetry).encode('utf-8'))
            return

        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"[OK] Dorina Neural Dashboard Server active at: http://localhost:{PORT}/web/dashboard.html", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
