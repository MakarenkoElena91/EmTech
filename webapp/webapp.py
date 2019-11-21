import PIL
from flask import Flask, request
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import re, time, base64
import keras
import tensorflow as tf
import json
from tensorflow.keras.models import load_model

model = load_model('./static/models/mnistModel.h5')
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
    image_data = BytesIO(byte_data) # print(image_data.getvalue())
    img = Image.open(image_data).convert("L")
    # img.save("original.png")
    # ----------------------------------------------------------------------------------------------------
    # crop the image
    rows, cols = np.nonzero(img)
    left = cols.min()
    up = rows.min()
    right = (199-cols.max())
    bottom = 199 - rows.max()
    border = (left, up, right, bottom)
    img = ImageOps.crop(img, border)
    img.save("cropped.png")
    print("cropped", img.size)
    # ----------------------------------------------------------------------------------------------------
    # resize the image 20x20
    size = 20
    width, height = img.size[:2]
    # ? x 20
    if height > width:
        baseheight = 20
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), Image.ANTIALIAS)
        background = Image.new("RGB", (20, 20))
        offset = ((size-img.size[0])//2, 0)
        print("offset", (size-img.size[0])//2)
        background.paste(img, offset)
        background.save('20x20.png')
        print("size1", background.size)
    #  20 x  ?
    else:
        basewidth = 20
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        background = Image.new("RGB", (20, 20))
        offset = (0, (size - img.size[1]) // 2)
        print("offset", offset[1])
        background.paste(img, offset)
        background.save('20x20.png')
        print("size2", background.size)
    # -----------------------------------------------------------------------------------------------------
    # find center_of_mass img.measurements.center_of_mass(n)
    img = Image.open("20x20.png").convert("L")
    n = np.array(img)
    print('n.size', n.size)
    x = 0
    y = 0
    numberOfBlackPixels = 0
    rows = n.shape[0]
    cols = n.shape[1]
    print("n.shape", n.shape[0], n.shape[1])
    for i in range(0, rows):
        for j in range(0, cols):
            if n[i][j] == 255:
                numberOfBlackPixels = numberOfBlackPixels + 1
                x += i
    x = x//numberOfBlackPixels
    print("x", x)

    for i in range(0, rows):
        for j in range(0, cols):
            if n[i][j] == 255:
                y += j
    y = y//numberOfBlackPixels
    print("y", y)
    # -----------------------------------------------------------------------------------------------------
    # recenter and resize image to 28x28  ref:https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python
    # img = Image.open("20x20.png").
    up = 14 - y
    left = 14 - x
    background = Image.new("RGB", (28, 28))
    offset = (left, up)
    print("offset", offset[0], offset[1])
    background.paste(img, offset)
    background.save("28x28.png")
    print("28x28", img)
    img = Image.open('28x28.png').convert("L")
    print("img.size", img.size)
    im2arr = ~np.array(img)/255.0
    # print("im2arr.size", im2arr.size)
    # im2arr = keras.utils.normalize(im2arr, axis=1)

    # for r in im2arr:
    #     for c in r:
    #         print(round(c, 1), end=" ")
    #     print()

    im2arr = im2arr.reshape(1, 28, 28, 1) # im2arr = im2arr.astype('float32')
    im2arr = tf.cast(im2arr, tf.float32)
    # -----------------------------------------------------------------------------------------------------
    # Predicting the Test set results
    predict = model.predict(im2arr)
    args = int(np.argmax(predict))
    args = json.dumps(args)
    print("args", np.argmax(predict))
    print("Number is ", predict)

    return {'message': args}
    # -----------------------------------------------------------------------------------------------------
    # testing versions
@app.route('/version')
def version():
    print('keras version' + keras.__version__)
    print('tensorflow version' + tf.__version__)
    return ''

if __name__ == '__main__':
    app.run(host='127.0.0.0', port=5000, threaded=False)
