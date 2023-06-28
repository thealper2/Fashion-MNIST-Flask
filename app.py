from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import load_model
import os
import numpy as np

app = Flask(__name__)

model = load_model("models/clothing.h5")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ["jpeg", "png", "jpg"]

def read_image(filename):
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape((-1, 28, 28, 1))
    img = img / 255.0
    return img

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["img"]
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = file.filename
            file_path = os.path.join("static/images", filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            print(class_prediction)
            classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            predicted_class_index = np.argmax(class_prediction)
            predicted_class = classes[predicted_class_index]
    
    return render_template("index.html", prediction=predicted_class)

if __name__ == "__main__":
    app.run(port=5000, debug=True)