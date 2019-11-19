import numpy as np

# Predicting the Test set results
def get_prediction(im2arr, model):
    predict = model.predict(im2arr)
    prediction = np.array(predict[0])
    return np.argmax(prediction)
    return predict
