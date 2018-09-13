import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Conv2D, Deconv2D, Activation, BatchNormalization, Add
from keras.callbacks import ModelCheckpoint

from datagen import gen_data

SEED = 1

EPOCHS = 40
BATCH_SIZE = 4
LOAD_WEIGHTS = False


def resnet_block(layer_input, filters=64, f_size=3):
    y = Conv2D(filters, (f_size, f_size), padding='same')(layer_input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, (f_size, f_size), padding='same')(y)
    y = BatchNormalization()(y)

    y = Add()([layer_input, y])

    return Activation('relu')(y)


IMG_HEIGHT, IMG_WIDTH = 128, 128

inputs = Input((None, None, 1))

x = Conv2D(64, 9, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = resnet_block(x)
x = resnet_block(x)
x = resnet_block(x)

outputs = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

if LOAD_WEIGHTS:
    model.load_weights('model4.h5')

model.compile(loss='MSE', optimizer='Adam')

checkpointer = ModelCheckpoint(filepath='model4.h5', verbose=1)


def _train_generator():
    rnd = np.random.RandomState(SEED)
    while True:
        yield gen_data(rnd, BATCH_SIZE)


def _val_generator():
    rnd = np.random.RandomState(SEED + 1)
    while True:
        yield gen_data(rnd, BATCH_SIZE)


train_generator = _train_generator()
val_generator = _val_generator()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=512 // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=32 // BATCH_SIZE,
    callbacks=[checkpointer]
)

model.save('model4_final.h5')
