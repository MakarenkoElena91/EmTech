from flask import Flask, escape, request
import numpy as np
from PIL import Image
import re
import io
from io import BytesIO
import re, time, base64
# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('static/myResult.h5')
model.summary()

app = Flask(__name__)

# Add a route for the web page.
@app.route('/')
def home():
  return app.send_static_file('index.html')

@app.route('/guess', methods=['GET','POST'])
def get_image(): 
  imgString = request.values.get("imgURL")# form -post   args -get
  print("image is : ", imgString)
  
# https://github.com/python-pillow/Pillow/issues/3400#issuecomment-428104239
  base64_data = re.sub('^data:image/.+;base64,', '', imgString)
  byte_data = base64.b64decode(base64_data)
  image_data = BytesIO(byte_data)
  img = Image.open(image_data)
 
  t = time.time()
  img.save("imagefile.png")
  img = Image.open('imagefile.png').convert("L")
  print("img is : ", img)
  img = img.resize((28,28))
  im2arr = np.array(img)
  im2arr = (im2arr.reshape(1,28,28,1))/255
  print("im2arr is : ", im2arr)
  # Predicting the Test set results
  pred = model.predict(im2arr)
  print(pred)
  # correct_indices = np.nonzero(pred)
  # print(correct_indices)
  # print("The program predicts image number to be:", correct_indices[-1])

  return ''

if __name__ == '__main__':
    app.run(debug = True)