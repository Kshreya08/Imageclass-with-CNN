import tensorflow as tf
import tensorflow.keras as keras
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

IMAGE_SIZE = 256
BATCH_SIZE = 64
CHANNELS = 3 # RGB
EPOCHS = 10

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:5000]
training_labels = training_labels[:5000]
testing_images = testing_images[:1000]
testing_labels = testing_labels[:1000]

reconstructed_model = keras.models.load_model('image_classifier.keras')

img = cv.imread('deer.jpg (2).jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)
plt.show()

prediction = reconstructed_model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
print("Thank you")
