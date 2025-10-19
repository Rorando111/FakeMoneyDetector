from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/weights/best.pt")

# Run inference quietly
results = model.predict(
    source="dataset/valid/images",
    show=False,   # ❌ disable window display
    save=False,   # ❌ don’t save output images
    conf=0.5,     # confidence threshold
    verbose=False # ✅ suppress extra logs
)

# Print class predictions and confidence
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls]} ({conf:.2f})")
