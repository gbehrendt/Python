# ME:4150 Artificial Intelligence in Engineering
# Project 4: ANN - movie review classification

import keras

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

# here is how to decode one review back to English words
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# one-hot-encoding

import numpy as np

def vectorize_sequences(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# vectorizing the dataset
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = train_labels
y_test = test_labels


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Optimizer (is this in the right plce?)
#optimizer = keras.optimizers.Adam(lr=0.001) #don't need bc the optimizer is already in the next line of code

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# training
history = model.fit(x_train, y_train, epochs=20, batch_size=512)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# best model
best_score = 0
best_epoch = 1
for c in range (1, len(acc)+1, 1):
    if val_acc[c-1] > best_score:
        best_epoch = c
        best_score = val_acc[c-1]

print("The best model is obtained right after {} epochs".format(best_epoch))

#reinitiate model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#train the model
model.fit(x_train, y_train, epochs=20, batch_size=512)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('The test score is {:.2f}%'.format(test_acc*100))

