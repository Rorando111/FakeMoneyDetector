from django.db import models

# Configuration constants for the fake money detector
class DetectorConfig:
    """Configuration constants for the fake money detection system"""

    # File upload settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    MIN_DETECTION_CONFIDENCE = 0.1

    # Messages
    MESSAGES = {
        'invalid_format': 'Please upload a valid image file (JPG, PNG, BMP, WebP)',
        'file_too_large': 'File size must be less than 10MB',
        'no_detection': 'Could not detect money in the image. Please try a clearer image.',
        'processing_error': 'Error processing image. Please try again.',
        'real_money': 'Real Money ✅',
        'fake_money': 'Fake Money ❌'
    }

    # Model settings
    MODEL_PATH = "runs/weights/best.pt"
    INPUT_IMAGE_SIZE = 640
