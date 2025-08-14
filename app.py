from flask import Flask, render_template, send_file, request, jsonify
from flask_scss import Scss
import io
import os
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import base64
from collections import Counter

# Create an instance from the object flask
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define class detection
class Detection():
    def __init__(self):
        model_path = 'best.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")
        print(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully")
        
    def predict(self, imgs, classes=[], conf=0.5):
        try:
            if classes:
                results = self.model.predict(imgs, classes=classes, conf=conf)
            else:
                results = self.model.predict(imgs, conf=conf)
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
            
    def predict_and_detect(self, img, classes=[], conf=0.3, rectangle_thickness=2, text_thickness=2):
        try:
            results = self.predict(img, classes, conf=conf)
            
            # Make a copy to avoid modifying original
            img_copy = img.copy()
            detected_classes = []
            
            for result in results:
                if result.boxes is not None:  # Check if boxes exist
                    for box in result.boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = result.names[class_id]
                        detected_classes.append(class_name)
                        
                        # Draw rectangle
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), rectangle_thickness)
                        
                        # Draw label with background for better visibility
                        label = f"{class_name}: {confidence:.2f}"
                        
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_thickness)
                        
                        # Draw text background rectangle
                        cv2.rectangle(img_copy, 
                                    (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), 
                                    (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(img_copy, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), text_thickness)
                        
            return img_copy, results, detected_classes
        except Exception as e:
            print(f"Detection error: {e}")
            raise

    def detect_from_image(self, image):
        try:
            # Convert RGB to BGR for OpenCV operations
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
                
            result_img, results, detected_classes = self.predict_and_detect(image_bgr, classes=[], conf=0.3)
            
            # Convert back to RGB for PIL
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            return result_rgb, detected_classes
        except Exception as e:
            print(f"detect_from_image error: {e}")
            raise

# Initialize detection instance
try:
    detection = Detection()
except Exception as e:
    print(f"Failed to initialize detection: {e}")
    detection = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    try:
        print("Starting detection process...")
        
        if detection is None:
            return "Detection model not initialized", 500
            
        if 'image' not in request.files:
            print("No file part in request")
            return 'No file part', 400
        
        file = request.files['image']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("No file selected")
            return 'No selected file', 400
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return f'Invalid file type. Allowed: {allowed_extensions}', 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {file_path}")
        file.save(file_path)
        
        print("Loading and processing image...")
        # Load image
        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        
        print(f"Original image shape: {img.shape}")
        
        # Resize image
        img = cv2.resize(img, (640, 640))  # YOLO typically works better with 640x640
        print(f"Resized image shape: {img.shape}")
        
        print("Running detection...")
        result_img, detected_classes = detection.detect_from_image(img)
        print("Detection completed successfully")
        
        # Count class distribution
        class_distribution = dict(Counter(detected_classes))
        print(f"Class distribution: {class_distribution}")
        
        print("Converting to output format...")
        # Ensure proper data type
        if result_img.dtype != np.uint8:
            result_img = result_img.astype(np.uint8)
        
        # Convert to PIL Image
        output = Image.fromarray(result_img)
        
        # Convert to base64 for JSON response
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Clean up
        os.remove(file_path)
        print("Process completed successfully")
        
        # Return JSON response with image and class distribution
        return jsonify({
            'image': img_base64,
            'class_distribution': class_distribution
        })
        
    except Exception as e:
        print(f"Error in apply_detection: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up file if it exists
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
            
        return f"Detection failed: {str(e)}", 500

if __name__ == "__main__":

      app.run(host='0.0.0.0', port=5000, debug=False)
