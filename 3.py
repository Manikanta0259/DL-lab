# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
# Step 1: Load Reuters dataset
num_words = 10000
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
# Step 2: Vectorize the data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results
x_train = vectorize_sequences(x_train, num_words)
x_test = vectorize_sequences(x_test, num_words)
# Step 3: One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Step 4: Build the Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(num_words,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Step 6: Train the model
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2)
# Step 7: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
# Step 8: Display the result
print("Test Accuracy:", accuracy)