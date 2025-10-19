# Import necessary Django and Python modules
from django.shortcuts import render        # Used to render HTML templates
from ultralytics import YOLO               # YOLO library for object detection (used here to detect fake features)
import os                                  # Used to handle file paths
from django.conf import settings           # Gives access to Django project settings like BASE_DIR, MEDIA_ROOT


# ✅ Load your trained YOLO model once when the server starts
# This prevents the model from being reloaded on every request, improving speed
MODEL_PATH = os.path.join(settings.BASE_DIR, "runs/weights/best.pt")

# Create a YOLO model instance using your trained weights
model = YOLO(MODEL_PATH)


def index(request):
    """
    🏠 The index view displays the upload form.
    - It responds to GET requests (when user first visits the site)
    - It simply renders the 'index.html' template
    """
    return render(request, 'index.html')


def predict_image(request):
    """
    🔍 The prediction view handles:
        1. Receiving the uploaded image
        2. Saving it temporarily
        3. Running the YOLO model on it
        4. Displaying the result (Real or Fake)
    """

    # Check if this request came from the upload form and has an image file
    if request.method == 'POST' and request.FILES.get('image'):

        # Get the uploaded image file object from the request
        image = request.FILES['image']

        # Build a full path for saving the uploaded image temporarily
        # MEDIA_ROOT is typically something like: BASE_DIR / 'media'
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)

        # Save the uploaded image to disk in chunks (in case it's a large file)
        with open(image_path, 'wb+') as f:
            for chunk in image.chunks():  # Django uploads files in "chunks" to save memory
                f.write(chunk)

        # 🧠 Run YOLO detection on the uploaded image
        results = model(image_path)

        # Extract detection boxes (if any objects were found)
        detections = results[0].boxes

        # If detections exist, map each detected class index to its label name (model.names)
        # Otherwise, create an empty list
        labels = [model.names[int(cls)] for cls in detections.cls] if detections else []

        # 🧩 FIX: Keep labels as a LIST (not a string)
        # This allows Django to loop over them properly
        label_list = [label for label in labels]

        # Determine if the money is real or fake
        # If *any* detected label contains the word "false", we classify it as fake
        prediction = "Fake Money ❌" if any("false" in label for label in labels) else "Real Money ✅"

        # 🖼️ Render result page with both prediction and label list
        return render(request, 'result.html', {
            'prediction': prediction,
            'labels': label_list
        })

    # If user visits this URL without uploading anything, just show the upload page
    return render(request, 'index.html')
