from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import DetectorConfig
from .views import validate_image_file, get_highest_confidence_detection
import os
import tempfile

class DetectorTests(TestCase):
    """Test cases for the fake money detector functionality"""

    def test_validate_image_file_valid_jpg(self):
        """Test validation of valid JPG file"""
        # Create a dummy JPG file
        file_content = b'fake image content'
        uploaded_file = SimpleUploadedFile(
            "test.jpg",
            file_content,
            content_type="image/jpeg"
        )

        is_valid, error_message = validate_image_file(uploaded_file)
        self.assertTrue(is_valid)
        self.assertIsNone(error_message)

    def test_validate_image_file_valid_png(self):
        """Test validation of valid PNG file"""
        file_content = b'fake png content'
        uploaded_file = SimpleUploadedFile(
            "test.png",
            file_content,
            content_type="image/png"
        )

        is_valid, error_message = validate_image_file(uploaded_file)
        self.assertTrue(is_valid)
        self.assertIsNone(error_message)

    def test_validate_image_file_invalid_extension(self):
        """Test validation of invalid file extension"""
        file_content = b'fake content'
        uploaded_file = SimpleUploadedFile(
            "test.txt",
            file_content,
            content_type="text/plain"
        )

        is_valid, error_message = validate_image_file(uploaded_file)
        self.assertFalse(is_valid)
        self.assertIn("valid image file", error_message)

    def test_validate_image_file_too_large(self):
        """Test validation of file that's too large"""
        # Create a file larger than MAX_FILE_SIZE (10MB)
        large_content = b'x' * (11 * 1024 * 1024)  # 11MB
        uploaded_file = SimpleUploadedFile(
            "large.jpg",
            large_content,
            content_type="image/jpeg"
        )

        is_valid, error_message = validate_image_file(uploaded_file)
        self.assertFalse(is_valid)
        self.assertIn("10MB", error_message)

    def test_detector_config_constants(self):
        """Test that configuration constants are properly defined"""
        self.assertEqual(DetectorConfig.MAX_FILE_SIZE, 10 * 1024 * 1024)
        self.assertIn('.jpg', DetectorConfig.ALLOWED_EXTENSIONS)
        self.assertIn('.png', DetectorConfig.ALLOWED_EXTENSIONS)
        self.assertEqual(DetectorConfig.CONFIDENCE_THRESHOLD, 0.5)
        self.assertIn('invalid_format', DetectorConfig.MESSAGES)
        self.assertIn('file_too_large', DetectorConfig.MESSAGES)

    def test_get_highest_confidence_detection_no_detections(self):
        """Test handling of results with no detections"""
        # Mock results object with no boxes
        mock_results = type('MockResults', (), {'boxes': None})()

        best_class, confidence = get_highest_confidence_detection([mock_results])
        self.assertIsNone(best_class)
        self.assertEqual(confidence, 0.0)

    def test_get_highest_confidence_detection_empty_boxes(self):
        """Test handling of results with empty boxes"""
        # Mock results object with empty boxes
        mock_boxes = type('MockBoxes', (), {
            'conf': [],
            'cls': []
        })()
        mock_results = type('MockResults', (), {'boxes': mock_boxes})()

        best_class, confidence = get_highest_confidence_detection([mock_results])
        self.assertIsNone(best_class)
        self.assertEqual(confidence, 0.0)
