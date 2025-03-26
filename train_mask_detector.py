# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\aditya\project\Face-Mask-Detection-master"
CATEGORIES = ["with_mask", "without_mask"]

# Define image generator with preprocessing and splitting
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,  # Split data into training and validation (20% validation)
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Generate training data
train_generator = datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='training'  # Specify training subset
)

# Generate validation data
validation_generator = datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='validation'  # Specify validation subset
)

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")

# Get true labels and predicted probabilities for classification report
y_true = validation_generator.classes
num_validation_samples = validation_generator.samples
y_pred_probabilities = model.predict(validation_generator, steps=(num_validation_samples + BS - 1) // BS)  # Ensure complete coverage
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Get the class labels in the correct order
labels = list(validation_generator.class_indices.keys())

# Check if the number of unique predicted labels matches the number of target names
if len(np.unique(y_pred)) != len(labels):
    print("[WARNING] Number of unique predicted labels does not match number of target names.")
    print("[WARNING] This might lead to issues with classification report.")

# show a nicely formatted classification report
print(classification_report(y_true, y_pred,
    target_names=labels, zero_division=0))  # Use labels here and handle zero division

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"],
    label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")