import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

main_path = './samo_dobre_mace/'

from keras.utils import image_dataset_from_directory


import os

folders = os.listdir(main_path)
folder_counts = {}
for folder in folders:
    folder_path = main_path + folder
    folder_counts[folder] = len(os.listdir(folder_path))
folder_counts_df = pd.DataFrame.from_dict(folder_counts, orient='index', columns=['counts'])
folder_counts_df.plot(kind='bar')
plt.show()


Xtrain = image_dataset_from_directory(main_path)
classes = Xtrain.class_names

import os
from PIL import Image

mindim = float('inf')
for i in classes:
    folder_path = main_path + i
    min_dimension = float('inf')
    for file in os.listdir(folder_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(folder_path, file)
            with Image.open(file_path) as img:
                width, height = img.size
                min_dimension = min(min_dimension, width, height)
    if (mindim > min_dimension):
        mindim = min_dimension

print("Minimalna dimenzija ucitanih slika je: ", mindim,
      "\n\n sada cemo ucitati ponovo slike ali ce ovaj put biti resized na velicinu minimalne slike\n")
if mindim > 64:
    imagesize = (64, 64)
else:
    imagesize = (32, 32)

# ponovno ucitavanje
# alternativno da se proba i sa 64x64 ako odje bude lose jelte

Xtrain = image_dataset_from_directory(main_path,
                                      batch_size=128,
                                      image_size=imagesize,
                                      validation_split=0.3,
                                      subset="training",
                                      seed=123)
Xval = image_dataset_from_directory(main_path,
                                    batch_size=128,
                                    image_size=imagesize,
                                    validation_split=0.3,
                                    subset="validation",
                                    seed=123)
Xval, test_data = Xval.take(int(len(Xval) * 0.5)), Xval.skip(int(len(Xval) * 0.5))
print(Xval)

classes = Xtrain.class_names
print(classes)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N / 2), i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
plt.show()

from keras import layers, Model
from keras import Sequential

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(imagesize[0], imagesize[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.5)

    ]
)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N / 2), i + 1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()

from keras import Sequential
from keras import layers
from keras.optimizers import Adam

num_classes = len(classes)
print(num_classes)


def make_model():
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(imagesize[0], imagesize[1], 3)),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),

        # Additional convolutional layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Additional convolutional layer
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Additional convolutional layer
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the image to 1D
        layers.Flatten(),

        # Additional fully connected layer
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(15, activation='softmax')])

    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics='accuracy'
    )
    return model


model = make_model()
from keras.callbacks import EarlyStopping

# definisanje ranog zaustavljanja
stop_early = EarlyStopping(monitor='val_accuracy', patience=15)

history = model.fit(Xtrain,
                    epochs=200,
                    validation_data=Xval,
                    callbacks=[stop_early],
                    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

# matrica konfuzije za trening skup
labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

# matrica konfuzije za test skup

labels = np.array([])
pred = np.array([])
for img, lab in test_data:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

print("Tačnost na test skupu:", model.evaluate(test_data)[1] * 100, '%.')


model.save('./sacuvan_model')
