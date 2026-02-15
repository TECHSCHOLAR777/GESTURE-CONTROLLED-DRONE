import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = r"C:\Users\HP\OneDrive\RISHI GARG LAB\RISHI_GARG_SKILL_STACK\PYTHON WORKSPACE\PYTHON HEAVY PROJECTS\CNN TRAINED GESTURE CONTROLLED DRONE PROJECT\dataset"


genimagedata = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)


train_data = genimagedata.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


val_data = genimagedata.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


gesture_model = Sequential()

gesture_model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

gesture_model.add(Conv2D(32, (3,3), activation="relu"))
gesture_model.add(MaxPooling2D(2,2))

gesture_model.add(Conv2D(64, (3,3), activation="relu"))
gesture_model.add(MaxPooling2D(2,2))

gesture_model.add(Conv2D(128, (3,3), activation="relu"))
gesture_model.add(MaxPooling2D(2,2))

gesture_model.add(Flatten())

gesture_model.add(Dense(128, activation="relu"))
gesture_model.add(Dropout(0.35))

gesture_model.add(Dense(7, activation="softmax"))


gesture_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

gesture_model.summary()


history = gesture_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


gesture_model.save("gesture_cnn_model_version2.h5")


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])

plt.show()


print("Model trained and saved as gesture_cnn_model.h5")
