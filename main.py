from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

app = FastAPI()

# Enable CORS (for frontend interaction)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("models\multi_task_chest_xray_model.h5")

# Class names and severity levels
class_names = ["COVID-19","Pneumonia","Normal"]
severity_levels = ["Mild", "Moderate", "Severe"]

# Grad-CAM utility functions
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_encode_gradcam(heatmap, original_image):
    img = Image.open(io.BytesIO(original_image)).convert("RGB")
    img = img.resize((224, 224))
    heatmap = np.uint8(255 * heatmap)

    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((img.width, img.height))
    superimposed_img = Image.blend(img, jet_heatmap, alpha=0.4)

    buf = io.BytesIO()
    superimposed_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Preprocess image
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    # Predict
    disease_pred, severity_pred = model.predict(img_array)
    disease_idx = np.argmax(disease_pred[0])
    disease = class_names[disease_idx]

    if disease != "Normal":
        severity_idx = np.argmax(severity_pred[0])
        severity = severity_levels[severity_idx]
    else:
        severity = "Normal"

    # Grad-CAM
    last_conv_layer = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=disease_idx)
    gradcam_encoded = save_and_encode_gradcam(heatmap, contents)

    return {
        "disease": disease,
        "severity": severity,
        "gradcam": gradcam_encoded
    }
