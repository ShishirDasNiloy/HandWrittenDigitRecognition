import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_train.training import data


""" 
The MNIST database (Modified National Institute of 
Standards and Technology database) is a large database 
of handwritten digits that is commonly used for 
training various image processing systems.
"""
mnist = tf.keras.datasets.mnist


def test_model(model):
    num = 1
    while os.path.isfile(f"../images/img{num}.png"):
        try:
            image = cv2.imread(f"../images/img{num}.png")[:, :, 0]
            image = np.invert(np.array([image]))
            prediction = model.predict(image)
            print(f"The Digit Probably: {np.argmax(prediction)}")
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")
        finally:
            num += 1


if __name__ == "__main__":
    if os.path.isdir("hw.model"):
        model = tf.keras.models.load_model('hwd.model')
        test_model(model=model)
    else:
        data.model_create(mnist=mnist)
        model = tf.keras.models.load_model('hwd.model')
        test_model(model=model)
