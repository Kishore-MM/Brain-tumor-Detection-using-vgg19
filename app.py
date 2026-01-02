import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Class names for the 4 classes used during training
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "Healthy Brain", "Pituitary Tumor"]

# Load the trained model
model_path = "brain_tumor_vgg19_classifier.h5"
model = load_model(model_path)

app = Flask(__name__)
print("✅ Model loaded. Visit http://127.0.0.1:5000/ to test.")

def get_result(img_path):
    """
    Process the image and predict the tumor type.
    Returns:
      prediction: A string with the predicted class or an uncertain message.
      conf: Confidence percentage.
      announcement: A custom message to be announced.
    """
    # 1) Read the image using OpenCV
    bgr_image = cv2.imread(img_path)
    if bgr_image is None:
        return "Error: Could not read image", 0.0, "There was an error reading the image."
    
    # 2) Convert BGR -> RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # 3) Resize to 224x224 for VGG19 input
    pil_img = Image.fromarray(rgb_image)
    pil_img = pil_img.resize((224, 224))
    
    # 4) Scale pixel values to [0, 1]
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    
    # 5) Predict with the model
    preds = model.predict(arr)[0]
    top_idx = np.argmax(preds)
    top_conf = preds[top_idx] * 100.0
    
    threshold = 65.0  # Confidence threshold
    
    # 6) Formulate the result based on the threshold
    if top_conf < threshold:
        sorted_indices = np.argsort(preds)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        prediction = (f"Uncertain. Possibly {CLASS_NAMES[top1_idx]} "
                      f"({preds[top1_idx]*100:.2f}%) or {CLASS_NAMES[top2_idx]} "
                      f"({preds[top2_idx]*100:.2f}%).")
        announcement = ("The results are uncertain. Please consult with your doctor for "
                        "further evaluation.")
    else:
        prediction = CLASS_NAMES[top_idx]
        # Customize the announcement text based on the prediction.
        if prediction == "Healthy Brain":
            announcement = "Congratulations, no tumor detected."
        else:
            announcement = (f"A {prediction} has been detected. Please contact the hospital immediately "
                            "for further consultation.")
    
    return prediction, top_conf, announcement

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detector")
def detector():
    return render_template("detector.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/hospital")
def hospital():
    return render_template("hospital.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is part of the request
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file to a temporary uploads folder
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, secure_filename(file.filename))
    file.save(file_path)

    # Get prediction results
    prediction, conf, announcement = get_result(file_path)
    return jsonify({
        "prediction": prediction,
        "confidence": f"{conf:.2f}%",
        "announcement": announcement
    }), 200



if __name__ == "__main__":
    app.run(debug=True)
