'''
author : Ashwin de Silva
'''

# import the libraries
import os
from keras.layers import Activation, BatchNormalization, Conv2D, concatenate, Conv2DTranspose, Lambda
from keras.layers import MaxPooling2D, Dropout, Input
from keras.models import *
import keras.backend as K
K.set_image_data_format('channels_last')
import h5py
import numpy as np
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.utilities import *
from keras.preprocessing.image import ImageDataGenerator
import scikitplot as skplt


class UNET(object):
    def __init__(self, img_rows=256, img_cols=256, channel=3, n_filters=16, dropout=0.1, batchnorm=True ):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.model = None

    def conv2d_block(self, input_tensor, filters, kernel_size=3):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def unet(self):

        input_img = Input((self.img_rows, self.img_cols, self.channel))

        # Contracting Path
        c1 = self.conv2d_block(input_img, filters=self.n_filters * 1, kernel_size=3)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout)(p1)

        c2 = self.conv2d_block(p1, filters=self.n_filters * 2, kernel_size=3)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)

        c3 = self.conv2d_block(p2,filters=self.n_filters * 4, kernel_size=3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)

        c4 = self.conv2d_block(p3, filters=self.n_filters * 8, kernel_size=3)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)

        c5 = self.conv2d_block(p4, filters=self.n_filters * 16, kernel_size=3)

        # Expansive Path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = self.conv2d_block(u6, filters=self.n_filters * 8, kernel_size=3)

        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = self.conv2d_block(u7, filters=self.n_filters * 4, kernel_size=3)

        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = self.conv2d_block(u8, filters=self.n_filters * 2, kernel_size=3)

        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(self.dropout)(u9)
        c9 = self.conv2d_block(u9, filters=self.n_filters * 1, kernel_size=3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        self.model = Model(inputs=[input_img], outputs=[outputs])

        # self.model.summary()

        return self.model


class DRIUNET(object):
    def __init__(self, data_path, model_path):
        self.unet = UNET()
        self.data_path = data_path
        self.model_path = model_path
        self.model = self.unet.unet()

        (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test) = load_data(self.data_path)

        plot_samples(self.x_train[1], self.y_train[1])

        self.datagen_args = dict(
                     rescale=1./255,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2
        )

        self.compile_args = dict(
                    optimizer=Adam(lr=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
        )

        self.epochs = 50

        self.earlystopper = EarlyStopping(
                    patience=10,
                    verbose=1
        )

        self.model_checkpoint = ModelCheckpoint(
                    self.model_path,
                    monitor='loss',
                    verbose=1,
                    save_best_only=True
        )

    def get_generator(self, images, masks):
        image_datagen = ImageDataGenerator(**self.datagen_args)
        mask_datagen = ImageDataGenerator(**self.datagen_args)

        seed = 1

        image_generator = image_datagen.flow(images, seed=seed)
        mask_generator = mask_datagen.flow(masks, seed=seed)

        return zip(image_generator, mask_generator)

    def train(self):
        self.model.compile(**self.compile_args)
        train_generator = self.get_generator(self.x_train, self.y_train)
        val_generator = self.get_generator(self.x_val, self.y_val)
        self.model.fit_generator(
                    train_generator,
                    epochs=self.epochs,
                    steps_per_epoch=200,
                    verbose=1,
                    callbacks=[self.model_checkpoint, self.earlystopper],
                    validation_data=val_generator,
                    validation_steps=10
        )
        print('Training Complete')

    def predict(self):
        self.x_test /= 255
        self.y_test /= 255

        model = load_model(self.model_path)
        pred_test = model.predict(self.x_test)

        # plot the results
        id = np.random.randint(pred_test.shape[0])
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
        ax1, ax2, ax3 = ax.ravel()
        ax1.imshow(np.squeeze(self.x_test[id]), cmap='jet')
        ax1.set_title('Retinal Image')
        ax2.imshow(np.squeeze(self.y_test[id]), cmap='jet')
        ax2.set_title('Annontated Vessels')
        ax3.imshow(np.squeeze(pred_test[id]), cmap='jet')
        ax3.set_title('Predicted Vessels')
        plt.show

        # plot P-R curve
        skplt.metrics.plot_precision_recall_curve(self.y_test.flatten(), pred_test.flatten())












