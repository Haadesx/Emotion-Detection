from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.backend import train # For explicit GPU training
import tensorflow as tf

# Check GPU availability (optional)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

picture_size = 48
folder_path = r"D:\SEM 6\mini project\Datasets\archive"
picture_size = 48
input_shape = (picture_size, picture_size, 1)

# Load sample image (optional)
# ... (code to load sample image, similar to previous example)

batch_size = 32  # Adjust based on GPU memory (might need to be lower)

datagen_train = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, featurewise_center=True)
datagen_val = ImageDataGenerator()  # No augmentation for validation data

train_set = datagen_train.flow_from_directory(
    folder_path + "/train",  # Point to the parent directory containing train subfolders
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    # Include subdirectories (assuming expressions are in subdirectories)
    classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
)

test_set = datagen_val.flow_from_directory(
    folder_path + "/test",  # Point to the parent directory containing test subfolders
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    # Include subdirectories (assuming expressions are in subdirectories)
    classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
)


no_of_classes = 7

model = Sequential()
  # Define input shape implicitly with the first layer
Conv2D(64, (3, 3), padding='same'),
BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

  # 2nd CNN layer
Conv2D(128, (5, 5), padding='same'),
BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

# 3rd CNN layer (optional, add more as needed)
Conv2D(256, (3, 3), padding='same'),
BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

# Flatten the feature maps
Flatten(),

# Fully connected layer 1
Dense(256),
BatchNormalization(),
Activation('relu'),
Dropout(0.25),

  # Fully connected layer 2
Dense(512),
BatchNormalization(),
Activation('relu'),
Dropout(0.25),

# Output layer with softmax activation for probabilities of each class
Dense(no_of_classes, activation='softmax')



opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train the model on GPU (replace with actual GPU ID if you have multiple)
with tf.device('/GPU:0'):
  model.fit(x=train_set, validation_data=test_set, epochs=48)

# ... (Optional: evaluate model, save model, etc.)
model.save('D:\SEM 6\mini project\Models\gpu_trained_model.keras')
#model.save_weights('path/to/folder/my_model_weights.h5')

