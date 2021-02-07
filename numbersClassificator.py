import tensorflow as tf
import numpy as np


(train_photos, train_answers), (test_photos, test_answers) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

train_photos = train_photos.reshape(60000, 784)

test_photos = test_photos.reshape(10000, 784)

train_answers = tf.keras.utils.to_categorical(train_answers, 10)

test_answers = tf.keras.utils.to_categorical(test_answers, 10)

answers = ["One", "Two", "Three", "Four", "Five", "Six", "Seven"
           ,"Eight", "Nine", "Ten"]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(250, activation=("sigmoid")))

model.add(tf.keras.layers.Dense(100, activation=("sigmoid")))

model.add(tf.keras.layers.Dense(500, activation=("sigmoid")))

model.add(tf.keras.layers.Dense(10, activation=("softmax")))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

model.fit(train_photos, train_answers, batch_size=200, epochs=50, validation_split=0.2)

accuracy = model.evaluate(test_photos, test_answers)

print(round(accuracy[1] * 100, 2), "%")

predictions = model.predict(test_photos)

i = 0
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
countCorrect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

while i < 10000:
    j = np.argmax(test_answers[i])
    if np.argmax(predictions[i]) == j:
        count[j] += 1
        countCorrect[j] += 1
    else: countCorrect[j] += 1    
    i += 1

k = 0

while k < 10:
    print("Percentage of correct answers", k, count[k] / countCorrect[k] * 100)
    k += 1
      

    
