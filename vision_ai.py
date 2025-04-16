import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class VisionAI:
    def __init__(self, text_ai=None, camera_index=0):
        """
        Initialize VisionAI with option to connect to existing TextGenerationAI

        Args:
            text_ai: Optional TextGenerationAI instance
            camera_index: Camera device index (default: 0)
        """
        self.text_ai = text_ai
        self.camera_index = camera_index
        self.camera = None
        self.processing_mode = "normal"

    def start_camera(self):
        """Initialize and start the camera"""
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return False
        return True

    def stop_camera(self):
        """Release the camera resources"""
        if self.camera is not None:
            self.camera.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Process a frame based on current mode"""
        if self.processing_mode == "normal":
            return frame
        elif self.processing_mode == "grayscale":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.processing_mode == "edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return edges
        elif self.processing_mode == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.processing_mode == "face_detection":
            return self.detect_faces(frame)
        elif self.processing_mode == "text_detection":
            return self.detect_text(frame)
        return frame

    def detect_faces(self, frame):
        """Detect faces in the frame"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame

    def detect_text(self, frame):
        """Simple text detection (requires text to be highlighted)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])

        # Create mask and apply it
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around potential text areas
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def describe_scene(self, frame):
        """Generate text description about what the AI sees"""
        # Use connected text_ai to generate description if available

        if self.processing_mode == "face_detection":
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                seed_text = "I don't see any faces in this image."
            else:
                seed_text = f"I see {len(faces)} faces in this image. They appear to be"
        else:
            # Default seed text based on the current mode
            seed_text = f"The image I'm seeing is processed with {self.processing_mode} mode. I can see"

        # Generate text description using the text generation model if available
        if self.text_ai and self.text_ai.model is not None:
            description = self.text_ai.generate_text(seed_text=seed_text, length=100, temperature=0.7)
            return description
        else:
            return seed_text + " (text generation model not available)"

    def capture_and_save(self, filename=None):
        """Capture a single frame and save it"""
        if not self.start_camera():
            return None

        ret, frame = self.camera.read()
        self.stop_camera()

        if not ret:
            print("Failed to capture image")
            return None

        processed = self.process_frame(frame)

        if filename:
            if len(processed.shape) == 2:
                processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(filename, processed_color)
            else:
                cv2.imwrite(filename, processed)
            print(f"Saved image to {filename}")

        return processed

    def visualize_frame(self, frame):
        """Display a frame using matplotlib"""
        if frame is None:
            return

        plt.figure(figsize=(10, 6))
        if len(frame.shape) == 3:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(frame, cmap='gray')
        plt.title(f"Processing Mode: {self.processing_mode}")
        plt.axis('off')
        plt.show()

    def run_vision_loop(self, show_window=True, enable_text_generation=False):
        """Run the main vision loop"""
        if not self.start_camera():
            return

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                processed_frame = self.process_frame(frame)

                display_frame = processed_frame.copy()
                if len(display_frame.shape) == 2:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                cv2.putText(
                    display_frame,
                    f"Mode: {self.processing_mode}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if show_window:
                    cv2.imshow('AI Vision', display_frame)

                if enable_text_generation and time.time() % 5 < 0.1:
                    description = self.describe_scene(frame)
                    print("\nAI Description:")
                    print(description)
                    print("\nPress 'q' to quit, 'm' to change mode, 's' to save frame, 'd' to describe")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    modes = ["normal", "grayscale", "edges", "blur", "face_detection", "text_detection"]
                    current_index = modes.index(self.processing_mode)
                    self.processing_mode = modes[(current_index + 1) % len(modes)]
                    print(f"Switched to mode: {self.processing_mode}")
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"ai_vision_{timestamp}.jpg"
                    if len(processed_frame.shape) == 2:  # If grayscale
                        save_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                    else:
                        save_frame = processed_frame
                    cv2.imwrite(filename, save_frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('d'):
                    # Generate a description on demand
                    description = self.describe_scene(frame)
                    print("\nAI Description:")
                    print(description)

        finally:
            self.stop_camera()