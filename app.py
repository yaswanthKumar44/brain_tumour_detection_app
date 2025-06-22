from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.utils import get_custom_objects
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout
import numpy as np
import os
import cv2
import tensorflow as tf

# Register custom objects for deserialization
get_custom_objects().update({
    'swish': swish,
    'FixedDropout': Dropout
})

# Initialize Flask app
app = Flask(__name__)

# Load trained EfficientNet model
model = load_model("brain_tumour_best_model.h5")

# Class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def generate_gradcam(model, img_array, layer_name, class_idx):
    """
    Generate Grad-CAM heatmap for the input image using the specified model and layer.
    """
    # Create a model that outputs the activations of the specified layer and the final output
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Get the layer activations and predictions
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Compute gradients of the class output with respect to the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)[0]

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)  # Normalize to [0, 1]

    return heatmap.numpy()

def superimpose_heatmap(heatmap, img_path, alpha=0.4):
    """
    Superimpose the heatmap on the original image using OpenCV.
    """
    # Load original image with OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input

    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply JET colormap

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0.0)

    # Save the heatmap image
    heatmap_path = os.path.join('static/uploads', f"heatmap_{os.path.basename(img_path)}")
    cv2.imwrite(heatmap_path, superimposed_img)

    return heatmap_path

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']

    if file.filename == '':
        return "No selected file."

    if file:
        # Save uploaded image
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        # Predict class
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_idx]
        confidence = np.max(prediction) * 100

        # Generate Grad-CAM heatmap
        # Specify the last convolutional layer (update this based on your model's architecture)
        last_conv_layer = 'top_conv'  # Replace with the actual last conv layer name
        try:
            heatmap = generate_gradcam(model, img_array, last_conv_layer, predicted_class_idx)
            heatmap_path = superimpose_heatmap(heatmap, filepath)
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            heatmap_path = None

        result = f"Predicted: {predicted_class.upper()} ({confidence:.2f}%)"
        return render_template('index.html', prediction=result, img_path=filepath, heatmap_path=heatmap_path)

# Run app
if __name__ == '__main__':
    app.run(debug=True)