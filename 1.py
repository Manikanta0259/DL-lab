# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Step 2: Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0
# Step 3: Build the Multi-Layer Perceptron model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))   # Input Layer
model.add(Dense(128, activation='relu'))   # Hidden Layer 1
model.add(Dense(64, activation='relu'))    # Hidden Layer 2
model.add(Dense(10, activation='softmax')) # Output Layer
# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Step 5: Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
# Step 6: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
# Step 7: Display the accuracy
print("Test Accuracy:", accuracy)