import pyautogui
import time
import sys

print("=" * 60)
print("Dorina-Alchemist: 4-Sekme Koordinat Kalibratörü")
print("=" * 60)
print("Talha, 5 saniye sonra fare imlecinin durduğu yerin X, Y koordinatını alacağız.")
print("Lütfen fareni ChatGPT'nin METİN YAZMA KUTUSUNUN üzerine getir ve bekle...")

for i in range(5, 0, -1):
    print(f"{i} saniye...")
    time.sleep(1)

pos = pyautogui.position()
print("\n" + "*" * 60)
print(f"YAKALANAN KOORDİNAT: X = {pos.x}, Y = {pos.y}")
print("*" * 60)
