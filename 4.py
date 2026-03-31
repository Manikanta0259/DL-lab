# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import backend as K
# Step 1: Load Boston Housing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# Step 2: Normalize the data
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
# Step 3: Build the Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer (Linear activation)
# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
# Step 5: Train the model
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
# Step 6: Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
# Step 7: Display the result
print("Mean Absolute Error:", mae)