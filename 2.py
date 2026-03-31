# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Step 1: Load IMDB dataset
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
# Step 2: Pad sequences to same length
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
# Step 3: Build the Neural Network model
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Step 5: Train the model
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2)
# Step 6: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
# Step 7: Display the result
print("Test Accuracy:", accuracy)