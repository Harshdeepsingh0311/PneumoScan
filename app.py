import os
# Minimize TensorFlow memory overhead on free tier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template

# Restrict TensorFlow to a single thread to prevent CPU spike & OOM on Render
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration for allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

_model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pneumonia_classifier.h5')

def _fix_model_config(obj):
    """
    Recursively patch model config dict to fix cross-version Keras issues.
    """
    import ast
    if isinstance(obj, dict):
        new_obj = dict(obj)

        # --- Fix 1: DTypePolicy → plain dtype string ---
        if 'dtype' in new_obj and isinstance(new_obj['dtype'], dict):
            dtype_info = new_obj['dtype']
            if dtype_info.get('class_name') == 'DTypePolicy':
                dtype_name = dtype_info.get('config', {}).get('name', 'float32')
                new_obj['dtype'] = dtype_name

        # --- Fix 2: batch_shape → batch_input_shape in InputLayer ---
        if new_obj.get('class_name') == 'InputLayer' and isinstance(new_obj.get('config'), dict):
            cfg = dict(new_obj['config'])
            if 'batch_shape' in cfg:
                cfg['batch_input_shape'] = cfg.pop('batch_shape')
                new_obj['config'] = cfg

        # --- Fix 3: Aggressive Shape String to List Conversion ---
        for k, v in new_obj.items():
            if isinstance(v, str):
                # Try to parse any string that looks like a shape (contains commas and (None or numbers))
                if ',' in v and ('None' in v or any(c.isdigit() for c in v)):
                    try:
                        # Handle both bracketed and non-bracketed strings
                        s = v.strip()
                        if not (s.startswith('(') or s.startswith('[')):
                            s = f"({s})"
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (tuple, list)):
                            new_obj[k] = list(parsed)
                    except Exception:
                        pass

        # Recursively apply to all values
        return {k: _fix_model_config(v) for k, v in new_obj.items()}

    elif isinstance(obj, list):
        return [_fix_model_config(item) for item in obj]

    return obj


def _load_patched_model(model_path):
    """
    Patch the H5 model's config JSON in-memory before handing it to Keras.
    This avoids deep registry errors that custom_objects cannot intercept.
    """
    import h5py, json, shutil, tempfile

    # Work on a temp copy so we never corrupt the original
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h5')
    os.close(tmp_fd)
    shutil.copy2(model_path, tmp_path)

    try:
        with h5py.File(tmp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                raw = f.attrs['model_config']
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                config = json.loads(raw)
                patched = _fix_model_config(config)
                f.attrs['model_config'] = json.dumps(patched)
                print("✅ Model config patched for Keras compatibility.")

        model = load_model(tmp_path, compile=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return model


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please ensure pneumonia_classifier.h5 is committed to the repository."
            )
        print(f"⏳ Loading model from {MODEL_PATH}...")
        _model = _load_patched_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    return _model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    """
    Preprocessing pipeline matched to training architecture:
    1. Load image and resize to 224x224
    2. Convert to array
    3. Rescale pixel values (1./255)
    4. Expand dimension to create batch size of 1
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescaling
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 224, 224, 3)
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Render the main frontend page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render uptime monitoring."""
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive image, run inference, and return JSON."""
    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Model failed to load: {str(e)}'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 1. Preprocess the image
            prepared_img = prepare_image(filepath)

            # 2. Run inference
            prediction = model.predict(prepared_img)
            probability = float(prediction[0][0])

            # 3. Format output based on architecture thresholds
            # 0.0 to 0.499: Normal | 0.5 to 1.0: Pneumonia
            if probability >= 0.5:
                diagnosis = "Pneumonia Detected 🦠"
                color = "danger"
            else:
                diagnosis = "Normal Lungs 🫁"
                color = "success"

            response = {
                'diagnosis': diagnosis,
                'probability': probability,
                'confidence': f"{(probability if probability >= 0.5 else 1 - probability) * 100:.2f}%",
                'color': color,
                'message': 'Prediction successful'
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file to save disk space
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'Invalid file format. Please upload a PNG, JPG, or JPEG file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
