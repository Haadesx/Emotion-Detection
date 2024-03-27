# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:27:59 2024

@author: conta
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
picture_size = 48
folder_path = r"D:\SEM 6\mini project\Datasets\archive"
train_path=r"D:\SEM 6\mini project\Datasets\archive\test"
test_path=r"D:\SEM 6\mini project\Datasets\archive\train"
batch_size = 32
no_of_classes = 7
input_shape = (picture_size, picture_size, 1)

# Data generators
datagen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)

train_set = datagen_train.flow_from_directory(
    test_path,
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
)

val_set = datagen_train.flow_from_directory(
    test_path,
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
)

# Model definition
model = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),

    Dense(512),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(no_of_classes, activation='softmax')
])

# Optimizer
opt = Adam(learning_rate=0.0001)

# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "potentially_best_model.keras",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Training
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=50,
    callbacks=[checkpoint]  # Pass the callback here
)

# Save model
model.save('D:\SEM 6\mini project\Models\gpu_trained_model_v4.keras')

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
