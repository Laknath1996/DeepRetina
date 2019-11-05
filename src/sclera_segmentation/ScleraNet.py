"""
objective : Define the unet architecture for sclera segmentation
author(s) : Ashwin de Silva
date      : 
"""

import keras.backend as K
K.set_image_data_format('channels_last')
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from src.sclera_segmentation.utilities import *
from keras.preprocessing.image import ImageDataGenerator
import scikitplot as skplt
from src.unet_zoo.UNET import *
import numpy as np


class ScleraNet(object):
    """
    deploys the sclera net
    """
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test) = load_data(self.data_path)
        # plot_samples(self.x_train[1], self.y_train[1])

        print('train samples = %i' % self.x_train.shape[0])
        print('val samples = %i' % self.x_val.shape[0])
        print('test samples = %i' % self.x_test.shape[0])

        self.unet = UNET(img_rows=self.x_train.shape[1], img_cols=self.x_train.shape[2], channel=self.x_train.shape[3])
        self.model = self.unet.unet()

        self.datagen_args = dict(
                     # rescale=1./255,
                     rotation_range=0,
                     # width_shift_range=0.1,
                     # height_shift_range=0.1,
                     # zoom_range=0.2
        )

        self.compile_args = dict(
                    optimizer=Adam(lr=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
        )

        self.epochs = 50

        self.earlystopper = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    verbose=1
        )

        self.model_checkpoint = ModelCheckpoint(
                    self.model_path,
                    monitor='loss',
                    verbose=1,
                    save_best_only=True
        )

    def get_generator(self, images, masks):
        """
        define the data generators
        """
        image_datagen = ImageDataGenerator(**self.datagen_args)
        mask_datagen = ImageDataGenerator(**self.datagen_args)

        seed = 1

        image_generator = image_datagen.flow(images, seed=seed)
        mask_generator = mask_datagen.flow(masks, seed=seed)

        return zip(image_generator, mask_generator)

    def train(self):
        """
        train the network
        """
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

    def extract_scleral_mask(self, im):
        """
        extract the scleral mask from an image of an eye
        :param im: eye image (not normalized to [0, 1])
        :return: scleral mask
        """
        im /= 255.
        model = load_model(self.model_path)
        mask = model.predict(im)
        return mask

    def evaluate(self):
        """
        evaluate the trained model
        """
        model = load_model(self.model_path)
        pred_test = model.predict(self.x_test)

        # plot the results
        id = np.random.randint(pred_test.shape[0])
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        ax1, ax2, ax3 = ax.ravel()
        ax1.imshow(np.squeeze(self.x_test[id]), cmap='jet')
        ax1.set_title('Original Image')
        ax2.imshow(np.squeeze(self.y_test[id]))
        ax2.set_title('Annontated Mask')
        ax3.imshow(np.squeeze(pred_test[id]))
        ax3.set_title('Predicted Mask')
        plt.show()

        # plot P-R curves (?)
        print('plotting PR Curves...')
        probas_unet = np.vstack((1-pred_test.flatten(), pred_test.flatten())).T
        ax1 = skplt.metrics.plot_precision_recall(self.y_test.flatten(), probas_unet, plot_micro=False, classes_to_plot=1)

        probas_human = np.vstack((1-self.y_test_human.flatten(), self.y_test_human.flatten())).T
        skplt.metrics.plot_precision_recall(self.y_test.flatten(), probas_human, plot_micro=False, classes_to_plot=1, ax=ax1, cmap='jet')

        # plot ROC curves (?)
        print('plotting PR Curves...')
        ax2 = skplt.metrics.plot_roc(self.y_test.flatten(), probas_unet, plot_micro=False, plot_macro=False, classes_to_plot=1)
        skplt.metrics.plot_roc(self.y_test.flatten(), probas_human, plot_micro=False, plot_macro=False, classes_to_plot=1, ax=ax2, cmap='jet')

        # accuracy metrics
        pred_class = np.copy(pred_test)
        pred_class[pred_class < 0.5] = 0
        pred_class[pred_class >= 0.5] = 1
        print('F1-Score : ', get_f1_score(self.y_test, pred_class))
        print('Accuracy : ', get_accuracy(self.y_test, pred_class))
        print('ROC AUC  : ', get_roc_auc(self.y_test, pred_test))
