# Step 1: Input sentence
sentence = "I love AI"

# Step 2: Split sentence into words
words = sentence.split()

# Step 3: Get unique words
unique_words = list(set(words))

# Step 4: Create one-hot encoding
one_hot = {}

for word in unique_words:
    vector = []
    for w in unique_words:
        if word == w:
            vector.append(1)
        else:
            vector.append(0)
    one_hot[word] = vector

# Step 5: Display output
for word, vector in one_hot.items():
    print(word, ":", vector)