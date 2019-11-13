from flask import Flask, escape, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import re, time, base64
# Importing the Keras libraries and packages
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from prediction import get_prediction

# model = load_model('./static/models/convomodel.h5')
model = load_model('./static/models/myResult.model')
model.summary()

app = Flask(__name__)

# Add a route for the web page.
@app.route('/')
def home():
    return app.send_static_file('index.html')

# adapted from https://github.com/python-pillow/Pillow/issues/3400#issuecomment-428104239
@app.route('/guess', methods=['GET', 'POST'])
def get_image():
    imgString = request.values.get("imgURL")  # form -post   args -get
    base64_data = re.sub('^data:image/.+;base64,', '', imgString)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    img.save("imagefile.png")
    img = Image.open('imagefile.png').convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    im2arr = tf.cast(im2arr, tf.float64)
    # Predicting the Test set results
    predict = model.predict(im2arr)
    #predict = get_prediction(im2arr, model)
    print(predict)
    correct_indices = np.nonzero(predict)
    print(correct_indices)
    print("The program predicts image number to be:", correct_indices[-1])

    return ''

# testing versions
@app.route('/version')
def version():
    print('keras version' + keras.__version__)
    print('tensorflow version' + tf.__version__)
    return ''


if __name__ == '__main__':
    app.run(host='127.0.0.0', port=5000, threaded=False)
