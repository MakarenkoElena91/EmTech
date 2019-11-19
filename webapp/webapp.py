from flask import Flask, escape, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import re, time, base64
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

model = load_model('./static/models/model1.h5')
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
    print(image_data.getvalue())
    img = Image.open(image_data)
    img.save("imagefile.png")
    img = Image.open('imagefile.png').convert("L")
    print(img)
    img = img.resize((28, 28))
    im2arr = ~np.array(img)/255.0
    im2arr = keras.utils.normalize(im2arr, axis=1)

    for r in im2arr:
        for c in r:
            print(round(c, 1), end=" ")
        print()

    im2arr = im2arr.reshape(1, 28, 28, 1)
    # im2arr = im2arr.astype('float32')
    im2arr = tf.cast(im2arr, tf.float32)
    # im2arr = np_utils.to_categorical(im2arr)

    # Predicting the Test set results
    predict = model.predict(im2arr)
    args = int(np.argmax(predict))
    args = json.dumps(args)
    print("args", np.argmax(predict))
    print("Number is ", predict)

    return {'message': args}

# testing versions
@app.route('/version')
def version():
    print('keras version' + keras.__version__)
    print('tensorflow version' + tf.__version__)
    return ''


if __name__ == '__main__':
    app.run(host='127.0.0.0', port=5000, threaded=False)
