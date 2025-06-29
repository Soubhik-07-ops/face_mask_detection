import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os # Import os for path checks

# Define the path to your trained model relative to where this script will run
MODEL_PATH = 'saved_models/mask_detection_model.h5'

# Load the model globally when this script is imported
try:
    # It's good practice to ensure the model architecture matches too if not using custom objects
    # For a simple Sequential model, load_model is usually sufficient.
    FACES_MODEL = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    FACES_MODEL = None # Set to None to handle cases where model loading fails

# Define the class names your model predicts, matching the order from flow_from_directory
# Based on alphabetical sorting of your folder names in 'train', 'test', 'val'
# You can verify this order from your `test_generator.class_indices` output in the notebook.
# Common order is: "mask_weared_incorrect", "with_mask", "without_mask"
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# --- Face Detection setup (using OpenCV Haar Cascade) ---
# You MUST download 'haarcascade_frontalface_default.xml' and place it
# in the same directory as this 'model_inference.py' file.
# The path uses cv2.data.haarcascades, which is useful for default OpenCV installations.
# If you place it in your project root, you might need: FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    # This path is generally reliable if OpenCV is installed correctly and has its data files
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if FACE_CASCADE.empty():
        # Fallback if the above path doesn't work (e.g., custom environment)
        # Try loading from the current directory if the default data path fails
        local_cascade_path = 'haarcascade_frontalface_default.xml'
        if os.path.exists(local_cascade_path):
            FACE_CASCADE = cv2.CascadeClassifier(local_cascade_path)
            if FACE_CASCADE.empty():
                raise IOError(f"haarcascade_frontalface_default.xml not loaded from default or local path: {local_cascade_path}")
        else:
            raise IOError("haarcascade_frontalface_default.xml not found. Please ensure it's in your project root.")
    print("Haar Cascade classifier loaded successfully.")
except Exception as e:
    print(f"Error loading Haar Cascade classifier: {e}")
    FACE_CASCADE = None

# --- NMS Helper Function ---
def non_max_suppression_fast(boxes, overlapThresh):
    """
    Applies Non-Maximum Suppression to a list of bounding boxes.
    Parameters:
        boxes (np.array): A NumPy array of bounding boxes in (x1, y1, x2, y2) format.
        overlapThresh (float): The IoU (Intersection Over Union) threshold.
                                Boxes with IoU greater than this threshold will be suppressed.
    Returns:
        np.array: The filtered list of bounding boxes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2) # Sorting by y2 is a common heuristic

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates for the end of
        # the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap (IoU)
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        # an overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked, converted back to integer type
    return boxes[pick].astype("int")


# Function to preprocess a single image for prediction
def preprocess_image(pil_img_object):
    """
    Preprocesses a PIL Image object for the mask detection model.
    Matches the target_size and rescaling used in your ImageDataGenerator
    (35x35 pixels, RGB, normalized to [0,1]).
    """
    # Ensure image is RGB (convert from potentially RGBA or grayscale)
    # This is important as your model expects 3 channels.
    img_rgb = pil_img_object.convert("RGB")
    
    # Resize the image to match model input shape (35, 35)
    # Your model summary shows input_shape=(35,35,3), so target_size (35,35) is correct.
    resized_img = img_rgb.resize((35, 35))

    # Convert to numpy array and normalize to [0,1]
    # Your ImageDataGenerator uses rescale=1.0/255, so this matches.
    img_array = np.array(resized_img).astype("float32") / 255.0
    
    # Expand dimensions to add batch size (1, 35, 35, 3)
    # Keras models expect a batch dimension, even for a single image prediction.
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# Function to detect faces and predict mask status
def detect_and_predict_faces(image_path):
    """
    Detects faces in an image, applies Non-Maximum Suppression,
    and predicts mask status for each *distinct* detected face.
    
    Parameters:
        image_path (str): The file path to the input image.
        
    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary contains
                    'box' (list of [x, y, w, h] for the detected face),
                    'prediction' (str label), and 'confidence' (str percentage).
            - str or None: An error message string if an error occurs, otherwise None.
    """
    if FACES_MODEL is None:
        return [], "Error: Model not loaded. Please check model path and file."
    if FACE_CASCADE is None:
        return [], "Error: Face detector not loaded. Please ensure 'haarcascade_frontalface_default.xml' is present and accessible."

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return [], "Error: Could not read image from path. Check file integrity or permissions."
    
    # Convert image to grayscale for face detection (Haar cascades work on grayscale)
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image using Haar Cascade Classifier
    # Returns a list of (x, y, w, h) bounding boxes.
    # minSize=(60, 60) added to reduce very small, likely false detections.
    raw_faces = FACE_CASCADE.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    # If no faces are detected by the initial cascade, return early
    if len(raw_faces) == 0:
        return [], "No faces detected in the image."

    # Convert raw_faces (x, y, w, h) into (x1, y1, x2, y2) format for NMS
    # Also ensure they are standard Python integers as numpy.int32 can cause JSON serialization issues.
    boxes_for_nms = []
    for (x, y, w, h) in raw_faces:
        boxes_for_nms.append([int(x), int(y), int(x + w), int(y + h)])
    
    # Apply Non-Maximum Suppression to filter out overlapping bounding boxes
    # overlapThresh (IoU threshold):
    # - A higher threshold (e.g., 0.5) allows more overlap, keeping more boxes.
    # - A lower threshold (e.g., 0.2) allows less overlap, suppressing more boxes.
    # Adjust this value (0.3 is a good starting point) based on your test image results.
    picked_boxes = non_max_suppression_fast(np.array(boxes_for_nms), overlapThresh=0.3)
    
    results = []
    if len(picked_boxes) == 0:
        return [], "No distinct faces detected in the image after filtering. Try a different image or adjust NMS threshold."

    for (x1, y1, x2, y2) in picked_boxes:
        # Convert the (x1, y1, x2, y2) format back to (x, y, w, h) for output consistency
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        # Replicate your notebook's crop_img logic for shifting the bounding box
        # This expands the detected face region slightly before cropping.
        x_shift = (w) * 0.1
        y_shift = (h) * 0.1

        # Calculate new crop coordinates, ensuring they stay within the original image bounds
        crop_x_min = int(max(0, x - x_shift))
        crop_y_min = int(max(0, y - y_shift))
        crop_x_max = int(min(img_cv.shape[1], x + w + x_shift))
        crop_y_max = int(min(img_cv.shape[0], y + h + y_shift))

        # Crop the face region from the original OpenCV image array
        cropped_face_cv = img_cv[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # Check if the cropped region is valid (not empty)
        if cropped_face_cv.size == 0 or cropped_face_cv.shape[0] == 0 or cropped_face_cv.shape[1] == 0:
            print(f"Warning: Empty or invalid crop for face at original box ({x},{y},{w},{h}). Skipping.")
            continue

        # Convert BGR (OpenCV's default color order) to RGB (PIL and Keras default)
        cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_cv, cv2.COLOR_BGR2RGB))
        
        # Preprocess the cropped face image for input to the Keras model
        processed_face_array = preprocess_image(cropped_face_pil)
        
        # Make prediction using the loaded Keras model
        predictions_raw = FACES_MODEL.predict(processed_face_array)
        
        # Get the predicted class index (highest probability) and its confidence
        predicted_class_index = np.argmax(predictions_raw[0])
        confidence = predictions_raw[0][predicted_class_index] * 100
        
        # Get the human-readable label for the predicted class
        predicted_label = CLASS_NAMES[predicted_class_index]
        
        # Append the results to the list
        results.append({
            # Convert NumPy integers to standard Python integers for JSON serialization
            'box': [int(x), int(y), int(w), int(h)], 
            'prediction': predicted_label,
            'confidence': f"{confidence:.2f}%" # Format confidence as a string
        })
    
    return results, None # Return list of results and no error message if successful