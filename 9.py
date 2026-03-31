# Step 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Step 2: Load IMDB dataset
vocab_size = 10000  # consider top 10,000 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
# Step 3: Pad sequences
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
# Step 4: Build the model
model = Sequential()
# Embedding Layer
model.add(Embedding(input_dim=vocab_size,
                    output_dim=128,
                    input_length=max_length))
# Flatten layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))
# Step 5: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Step 6: Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
# Step 7: Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)