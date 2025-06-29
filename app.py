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

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to prevent overwrites
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + os.path.splitext(original_filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            
            # Perform face detection and mask prediction
            predictions, error_message = detect_and_predict_faces(filepath)

            # Clean up the uploaded file after processing to save disk space
            if os.path.exists(filepath):
                os.remove(filepath)

            if error_message:
                return jsonify({'error': error_message}), 400
            
            if not predictions:
                # This case is handled by error_message, but good to have a fallback
                return jsonify({'message': 'No faces detected in the image.'}), 200

            # Return results to the frontend
            return jsonify({'predictions': predictions}), 200

        except Exception as e:
            # Catch any other unexpected errors during file saving or prediction
            if os.path.exists(filepath):
                os.remove(filepath) # Try to clean up even on error
            return jsonify({'error': f'Server error during prediction: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg.'}), 400

if __name__ == '__main__':
    # Make sure the 'static/uploads' directory exists on startup
    if not os.path.exists(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])):
        os.makedirs(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER']))
    
    app.run(debug=True) # Set debug=False for production