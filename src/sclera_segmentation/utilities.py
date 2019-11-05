'''
functions for file handling operations
'''

import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def save_to_hdf5(path, size):
    """
    save the images to an hdf5 dataset after diving them to train, val and test sets
    """
    folders = np.arange(1, 56, 1)

    train_folders = folders[:int(len(folders)*0.7)]
    val_folders = folders[int(len(folders)*0.7):int(len(folders)*0.9)]
    test_folders = folders[int(len(folders)*0.9):]

    train = np.zeros((1, size[0], size[1], 3))
    train_mask = np.zeros((1, size[0], size[1]))
    val = np.zeros((1, size[0], size[1], 3))
    val_mask = np.zeros((1, size[0], size[1]))
    test = np.zeros((1, size[0], size[1], 3))
    test_mask = np.zeros((1, size[0], size[1]))

    dataset = h5py.File(os.path.join(path, "scleral_dataset"), mode='w')

    print("Obtaining the train set...")

    idx = 0
    for folder in train_folders:
        image_files = os.listdir(os.path.join(path, str(folder)))
        original_images = []
        scleral_masks = []

        for image_file in image_files:
            if image_file.split('_')[-1] == 'sclera.png':
                scleral_masks.append(image_file)
        for image in scleral_masks:
            original_images.append(image[:-11] + '.JPG')

        for image_idx in range(len(original_images)):
            im = cv2.imread(os.path.join(path, str(folder), original_images[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, size)
            train = np.concatenate((train, im.reshape(1, size[0], size[1], 3)), axis=0)

            im = cv2.imread(os.path.join(path, str(folder), scleral_masks[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, size)
            train_mask = np.concatenate((train_mask, im.reshape(1, size[0], size[1])), axis=0)

            print(idx)
            idx += 1

    dataset.create_dataset('train', data=train[1:, ...])
    dataset.create_dataset('train_mask', data=train_mask[1:, ...])

    print("Obtaining the val set...")

    idx = 0
    for folder in val_folders:
        image_files = os.listdir(os.path.join(path, str(folder)))
        original_images = []
        scleral_masks = []

        for image_file in image_files:
            if image_file.split('_')[-1] == 'sclera.png':
                scleral_masks.append(image_file)
        for image in scleral_masks:
            original_images.append(image[:-11] + '.JPG')

        # print(scleral_masks)

        for image_idx in range(len(original_images)):
            im = cv2.imread(os.path.join(path, str(folder), original_images[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, size)
            val = np.concatenate((val, im.reshape(1, size[0], size[1], 3)), axis=0)

            im = cv2.imread(os.path.join(path, str(folder), scleral_masks[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, size)
            val_mask = np.concatenate((val_mask, im.reshape(1, size[0], size[1])), axis=0)

            print(idx)
            idx += 1

    dataset.create_dataset('val', data=val[1:, ...])
    dataset.create_dataset('val_mask', data=val_mask[1:, ...])

    print("Obtaining the test set...")

    idx = 0
    for folder in test_folders:
        image_files = os.listdir(os.path.join(path, str(folder)))
        original_images = []
        scleral_masks = []

        for image_file in image_files:
            if image_file.split('_')[-1] == 'sclera.png':
                scleral_masks.append(image_file)
        for image in scleral_masks:
            original_images.append(image[:-11] + '.JPG')

        # print(scleral_masks)

        for image_idx in range(len(original_images)):
            im = cv2.imread(os.path.join(path, str(folder), original_images[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, size)
            test = np.concatenate((test, im.reshape(1, size[0], size[1], 3)), axis=0)

            im = cv2.imread(os.path.join(path, str(folder), scleral_masks[image_idx]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, size)
            test_mask = np.concatenate((test_mask, im.reshape(1, size[0], size[1])), axis=0)

            print(idx)
            idx += 1

    dataset.create_dataset('test', data=test[1:, ...])
    dataset.create_dataset('test_mask', data=test_mask[1:, ...])

    dataset.close()


def load_data(data_path):
    """
    load the train, val and test data
    """
    data = h5py.File(data_path, mode='r')
    x_train = np.array(data["train"])
    y_train = np.array(data["train_mask"])
    x_val = np.array(data["val"])
    y_val = np.array(data["val_mask"])
    x_test = np.array(data["test"])
    y_test = np.array(data["test_mask"])

    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], y_val.shape[2], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

    x_train /= 255.
    x_val /= 255.
    x_test /= 255.

    # # just the red channel
    # x_train = x_train[..., 0].reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    # x_val= x_val[..., 0].reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    # x_test = x_test[..., 0].reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train[y_train >= 0.5] = 1
    y_val[y_val >= 0.5] = 1
    y_test[y_test >= 0.5] = 1

    y_train[y_train < 0.5] = 0
    y_val[y_val < 0.5] = 0
    y_test[y_test < 0.5] = 0

    data.close()
    return x_train, y_train, x_val, y_val, x_test, y_test


def plot_samples(X, Y):
    """
    plot the original image and its mask
    """
    X /= 255
    Y /= 255
    mask = np.zeros((Y.shape[0], Y.shape[1], 3))
    mask[..., 1] = Y

    dst = X + mask*2

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1, ax2, ax3 = ax.ravel()
    ax1.imshow(np.squeeze(X))
    ax1.set_title('Original Image')
    ax2.imshow(np.squeeze(Y), cmap='gray')
    ax2.set_title('Annotated Sclera Mask')
    ax3.imshow(np.squeeze(dst))
    ax3.set_title('Image + Mask')
    plt.show()


def get_f1_score(y_true, y_pred):
    """
    returns the f1 score
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return f1_score(y_true, y_pred, average='weighted')


def get_accuracy(y_true, y_pred):
    """
    returns the accuracy score
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return accuracy_score(y_true, y_pred)


def get_roc_auc(y_true, y_score):
    """
    returns the roc area under score
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    return roc_auc_score(y_true, y_score)
