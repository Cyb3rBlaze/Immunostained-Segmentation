import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

from os import listdir
import numpy as np
from PIL import Image
from tqdm import tqdm

'''
#  Preprocessing functions for efficient training
'''

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

def load_one_image(data, label):
    input_image = tf.image.resize(data, (128, 128))
    input_mask = tf.image.resize(label, (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_train(data_dir, label_dir):
    images = listdir(data_dir)
    dataImages = []
    labelImages = []

    print("Filter phase 1...")

    for i in images:
        image_ = Image.open(data_dir + "/" + i).convert("RGB").resize((128, 128))
        label_ = Image.open(label_dir + "/" + i).convert("RGB").resize((128, 128))
        dataImages += [np.array(image_)]
        labelImages += [np.array(label_)]

    print("Filter phase 2...")
    
    tempLabels = np.zeros((np.array(labelImages).shape[0], np.array(labelImages).shape[1], np.array(labelImages).shape[2], 1)).tolist()
    print(np.array(tempLabels).shape)

    index = 0
    for labelImage in tqdm(labelImages):
        for i in range(len(labelImage)):
            for j in range(len(labelImage[i])):
                if labelImage[i][j][0] < 10:
                    tempLabels[index][i][j] = [2]
        index += 1
    
    print("Filter phase 3...")

    index = 0
    for labelImage in tqdm(labelImages):
        for i in range(len(labelImage)):
            for j in range(len(labelImage[i])):
                if labelImage[i][j][0] > 220 and labelImage[i][j][1] > 220 and labelImage[i][j][2] > 220:
                    tempLabels[index][i][j] = [1]
                elif labelImage[i][j][0] > 220 and labelImage[i][j][1] > 80 and labelImage[i][j][2] > 85:
                    tempLabels[index][i][j] = [0]
        index += 1
    
    finalData = []
    finalLabels = []

    for i, j in zip(dataImages, tempLabels):
        combined = load_one_image(i, j)
        finalData += [combined[0]]
        finalLabels += [combined[1]]
    
    return finalData, finalLabels

data, labels = load_train("./raw_data", "./raw_labels")


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("test.jpg")
    plt.show()

'''
#  Model decleration and other model stuff
'''

BATCH_SIZE = 4
BUFFER_SIZE = 12
OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu', 
    'block_6_expand_relu', 
    'block_13_expand_relu',
    'block_16_project',
]
layers = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)

        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    print(pred_mask)
    return pred_mask[0]

train = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

sample_image = None
sample_mask = None
for image, mask in train.take(1):
    sample_image, sample_mask = image[0], mask[0]
    break

def show_predictions(num=1):
    display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        for image, mask in zip(data, labels):
            sample_image, sample_mask = image, mask
            show_predictions()
            break
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

model_history = model.fit(train, epochs=200, steps_per_epoch=BUFFER_SIZE/BATCH_SIZE)

print("done")
show_predictions()
