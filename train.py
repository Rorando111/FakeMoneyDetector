from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='dataset/data.yaml',     # Path to your dataset config
    epochs=50,                    # Total epochs
    imgsz=640,                    # Image size
    batch=8,                      # Adjust for your CPU/GPU
    name='fake_money_detector_v2',
    optimizer='SGD',              # Or 'Adam' if you prefer
    lr0=0.01,                     # Initial learning rate
    lrf=0.01,                     # Final learning rate factor
    momentum=0.937,               # Momentum term
    weight_decay=0.0005,          # Regularization
    warmup_epochs=3.0,            # Warm-up
    patience=5,                   # Early stopping
    degrees=5,                    # Random rotation
    translate=0.1,                # Random translation
    scale=0.5,                    # Random scaling
    shear=2.0,                    # Random shear
    fliplr=0.5,                   # Left-right flip
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentations
    augment=True,                 # Enable augmentation
)
