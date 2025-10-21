# Import necessary Django and Python modules
from django.shortcuts import render        # Used to render HTML templates
from ultralytics import YOLO               # YOLO library for object detection (used here to detect fake features)
import os                                  # Used to handle file paths
from django.conf import settings           # Gives access to Django project settings like BASE_DIR, MEDIA_ROOT
from .models import DetectorConfig         # Import configuration constants
import logging                             # For error logging
from io import BytesIO                     # For in-memory image processing

# Configure logging
logger = logging.getLogger(__name__)

# ✅ Load your trained YOLO model once when the server starts
# This prevents the model from being reloaded on every request, improving speed
MODEL_PATH = os.path.join(settings.BASE_DIR, DetectorConfig.MODEL_PATH)

# Create a YOLO model instance using your trained weights
model = YOLO(MODEL_PATH)


def validate_image_file(image_file):
    """
    Validate uploaded image file for size and format

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    # Check file size
    if image_file.size > DetectorConfig.MAX_FILE_SIZE:
        return False, DetectorConfig.MESSAGES['file_too_large']

    # Check file extension
    file_extension = os.path.splitext(image_file.name)[1].lower()
    if file_extension not in DetectorConfig.ALLOWED_EXTENSIONS:
        return False, DetectorConfig.MESSAGES['invalid_format']

    return True, None


def get_highest_confidence_detection(results):
    """
    Extract the detection with highest confidence from YOLO results

    Returns:
        tuple: (best_class: str, best_confidence: float) or (None, 0.0) if no detections
    """
    try:
        detections = results[0].boxes
        if not detections:
            return None, 0.0

        # Find detection with highest confidence
        confidences = detections.conf
        if len(confidences) == 0:
            return None, 0.0

        best_idx = confidences.argmax()
        best_confidence = float(confidences[best_idx])
        best_class_idx = int(detections.cls[best_idx])
        best_class = model.names[best_class_idx]

        return best_class, best_confidence

    except Exception as e:
        logger.error(f"Error extracting detection results: {e}")
        return None, 0.0


def index(request):
    """
    🏠 The index view displays the upload form.
    - It responds to GET requests (when user first visits the site)
    - It simply renders the 'index.html' template
    """
    return render(request, 'index.html')


def predict_image(request):
    """
    🔍 Enhanced prediction view with better error handling and confidence scores:
        1. Validates uploaded image (format, size)
        2. Saves it temporarily with error handling
        3. Runs the YOLO model with error handling
        4. Extracts confidence scores and best detection
        5. Displays result with confidence information
    """

    # Check if this request came from the upload form and has an image file
    if request.method == 'POST' and request.FILES.get('image'):

        image = request.FILES['image']

        # Validate the uploaded image
        is_valid, error_message = validate_image_file(image)
        if not is_valid:
            return render(request, 'result.html', {
                'prediction': f'Error: {error_message}',
                'error': True,
                'confidence': 0.0
            })

        try:
            # 🖼️ Process image in-memory using BytesIO (no disk saving)
            image_buffer = BytesIO()
            for chunk in image.chunks():
                image_buffer.write(chunk)
            image_buffer.seek(0)  # Reset buffer position to beginning

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return render(request, 'result.html', {
                'prediction': DetectorConfig.MESSAGES['processing_error'],
                'error': True,
                'confidence': 0.0
            })

        try:
            # 🧠 Run YOLO detection on the in-memory image buffer
            results = model(image_buffer)

            # Get the best detection with confidence score
            best_class, confidence = get_highest_confidence_detection(results)

            if best_class is None:
                # No detections found
                return render(request, 'result.html', {
                    'prediction': DetectorConfig.MESSAGES['no_detection'],
                    'error': True,
                    'confidence': 0.0
                })

            # Determine if the money is real or fake based on the best detection
            # If the detected class contains "false", classify as fake
            is_fake = "false" in best_class.lower()
            prediction = DetectorConfig.MESSAGES['fake_money'] if is_fake else DetectorConfig.MESSAGES['real_money']

            # 🖼️ Render result page with prediction, confidence, and detection details
            return render(request, 'result.html', {
                'prediction': prediction,
                'confidence': confidence,
                'detected_class': best_class,
                'error': False
            })

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render(request, 'result.html', {
                'prediction': DetectorConfig.MESSAGES['processing_error'],
                'error': True,
                'confidence': 0.0
            })

    # If user visits this URL without uploading anything, just show the upload page
    return render(request, 'index.html')
