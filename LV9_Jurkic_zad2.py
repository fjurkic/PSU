import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix
import numpy as np


train_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset='training',
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

validation_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset='validation',
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

test_ds = image_dataset_from_directory(
    directory='gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48),
    shuffle=False
)


model = Sequential()


model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(43, activation='softmax'))



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir='logs'
)



history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=20,
    callbacks=[checkpoint, tensorboard]
)



loss, accuracy = model.evaluate(test_ds)

print(f'Tocnost na testnom skupu: {accuracy * 100:.2f}%')



true_labels = np.concatenate([
    np.argmax(y, axis=1) for x, y in test_ds
])

predictions = model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)

print(cm)





