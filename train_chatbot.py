import nltk
import numpy as np
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize Stemmer
stemmer = LancasterStemmer()

# Load the intents file
try:
    with open('intents.json', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: intents.json not found.")
    exit()

words = []
classes = []
documents = []
ignore_words = ['?', '!', ',', '.']

# Loop through each sentence in intents
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to documents list
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

# Create training data
training = []
output_empty = [0] * len(classes)

# training set, bag of words for each pattern
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current intent's tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the data
np.random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
# Use SGD optimizer with Nesterov accelerated gradient
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model and data structures
model.save('chatbot_model.keras')
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data.pkl', 'wb'))

print("\nModel training complete and saved.")