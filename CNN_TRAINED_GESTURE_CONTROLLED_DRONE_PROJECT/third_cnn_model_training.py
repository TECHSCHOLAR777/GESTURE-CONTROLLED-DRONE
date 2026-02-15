import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import AdamW
from PIL import ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG = 128
BATCH = 32
EPOCHS = 25

DATASET_PATH = r"C:\Users\HP\OneDrive\RISHI GARG LAB\RISHI_GARG_SKILL_STACK\PYTHON WORKSPACE\PYTHON HEAVY PROJECTS\CNN TRAINED GESTURE CONTROLLED DRONE PROJECT\dataset"


gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    brightness_range=[0.85,1.15],
    horizontal_flip=True,
    validation_split=0.2
)

train = gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG,IMG),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

val = gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG,IMG),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)


init = HeNormal()

model = Sequential([
    Input(shape=(IMG,IMG,3)),

    Conv2D(32,(3,3),activation="relu",kernel_initializer=init),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu",kernel_initializer=init),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu",kernel_initializer=init),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation="relu",kernel_initializer=init),
    Dropout(0.4),

    Dense(7,activation="softmax")
])


optimizer = AdamW(
    learning_rate=0.001,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

early = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=[early]
)

model.save("gesture_cnn_model_version2.h5")


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Train")
plt.plot(history.history["val_accuracy"],label="Val")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"],label="Val")
plt.legend()
plt.title("Loss")

plt.show()

print("Model trained and saved")
