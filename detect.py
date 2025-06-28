
from ultralytics import YOLO
import os
import csv

model = YOLO('yolov8n.pt')

input_folder = "uydu_gorselleri"
output_folder = "sonuc"
os.makedirs(output_folder, exist_ok=True)

csv_file = os.path.join(output_folder, "rapor.csv")
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Gorsel", "Hedef", "Koordinatlar", "Guven"])

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            results = model(img_path)[0]
            boxes = results.boxes
            for box in boxes:
                cls = results.names[int(box.cls[0])]
                conf = round(float(box.conf[0]), 2)
                coords = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                writer.writerow([filename, cls, coords, conf])
            results.save(filename=os.path.join(output_folder, f"tespit_{filename}"))
print("Tespit ve raporlama tamamlandı.")
