from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
new_model = load_model("model_weights.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Load the image from the POST request
    file = request.files['image']
    img = load_img(file, target_size=(200, 200), grayscale=True)

    # Convert the image to a numpy array
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Normalize the pixel values
    x = x / 255.0

    # Make the prediction
    prediction = new_model.predict(x)

    # Get the class label with the highest probability
    predicted_class = np.argmax(prediction)

    # Return the predicted tumor type as a JSON response
    if predicted_class == 0:
        tumor_type = "Glioma"
    elif predicted_class == 1:
        tumor_type = "Meningioma"
    elif predicted_class == 2:
        tumor_type = "No tumor"
    elif predicted_class == 3:
        tumor_type = "Pituitary"
    else:
        tumor_type = "Unknown class label"
    
    response = {'tumor_type': tumor_type}
    return jsonify(response)

if __name__ == '__main__':
    app.run()