from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from model_inference import detect_and_predict_faces # Import your inference logic
import uuid # For unique filenames

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads' # Store uploads in static/uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size 16MB

# Ensure upload folder exists. This is primarily for local development environments
# where the directory might not exist. On Render, the build process should handle this
# if 'static/uploads' is committed, or you can rely on the OS creation if not.
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the uploaded file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Handles image uploads, performs face mask detection, and returns JSON results.
    """
    # Check if a file was actually sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if a file was selected by the user
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Process the file if it exists and has an allowed extension
    if file and allowed_file(file.filename):
        # Generate a unique filename to prevent potential naming conflicts
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + os.path.splitext(original_filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # Save the uploaded file temporarily
            file.save(filepath)
            
            # Call the face detection and mask prediction function from model_inference.py
            predictions, error_message = detect_and_predict_faces(filepath)

            # Clean up the uploaded file immediately after processing
            # This is crucial for web servers to manage disk space.
            if os.path.exists(filepath):
                os.remove(filepath)

            # Handle errors returned by the detection function
            if error_message:
                return jsonify({'error': error_message}), 400
            
            # Handle case where no faces were detected in the image
            if not predictions:
                return jsonify({'message': 'No faces detected in the image.'}), 200

            # Return the prediction results as JSON to the frontend
            return jsonify({'predictions': predictions}), 200

        except Exception as e:
            # Catch any other unexpected errors during file saving or prediction
            print(f"An unexpected error occurred: {e}") # Log the error for debugging
            if os.path.exists(filepath):
                os.remove(filepath) # Attempt to clean up temporary file even on error
            return jsonify({'error': f'Server error during prediction: {str(e)}'}), 500
    else:
        # Return error if the file type is not allowed
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg.'}), 400

# Removed:
# if __name__ == '__main__':
#     if not os.path.exists(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])):
#         os.makedirs(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER']))
#     app.run(debug=True)