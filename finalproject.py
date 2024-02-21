import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

data_dir_01 = pathlib.Path('/kaggle/input/final-project-ai-datasets/final_project_dataset/training_set')
image_count_01 = len(list(data_dir_01.glob('*/*.jpg')))
print(image_count_01)

data_dir_02 = pathlib.Path('/kaggle/input/final-project-ai-datasets/final_project_dataset/test_set')
image_count_02 = len(list(data_dir_02.glob('*/*.jpg')))
print(image_count_02)

batch_size = 64
img_height = 150
img_width = 150
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_01,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_01,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
print(class_names)

def conv_block(inputs, filters, kernel_size, strides):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def identity_block(inputs, filters, kernel_size):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, inputs])
    #x = layers.Activation("relu")(x)
    return x

def ResNet_like(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = conv_block(inputs, filters=8, kernel_size=3, strides=1)
    x = conv_block(x, filters=16, kernel_size=3, strides=1)

    x = identity_block(x, filters=16, kernel_size=3)
    x = conv_block(x, filters=32, kernel_size=3, strides=2)

    x = identity_block(x, filters=32, kernel_size=3)
    x = identity_block(x, filters=32, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3, strides=2)

    x = identity_block(x, filters=64, kernel_size=3)
    x = identity_block(x, filters=64, kernel_size=3)
    x = identity_block(x, filters=64, kernel_size=3)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(num_classes)(x)

    model = keras.Model(inputs=inputs, outputs=x, name="resnet_like")

    return model

num_classes = len(class_names)
input_shape = (img_height, img_width, 3)
model = ResNet_like(input_shape, num_classes)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()


#Define the callbacks

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto')

#Train the model

epochs = 20
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=epochs,
callbacks=[early_stop, reduce_lr]
)

#Plot the training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Evaluate the model on the test set
test_ds = tf.keras.utils.image_dataset_from_directory(
data_dir_02,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size
)

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test accuracy:", test_accuracy)

#Plot graph

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)