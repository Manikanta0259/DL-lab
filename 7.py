# Step 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Step 2: Load Pre-trained VGG16 model (without top layers)
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
(include_top=False removes VGG16’s original classifier.
# Step 3: Freeze convolution layers
for layer in vgg16.layers:
    layer.trainable = False
(Prevents changing pre-trained weights.
# Step 4: Build new model
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
(Custom classifier added for binary classification.
# Step 5: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Step 6: Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
# Step 7: Train the model
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)
# Step 8: Model summary
model.summary()