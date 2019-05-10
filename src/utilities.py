'''
functions for file handling operations
'''

import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
% matplotlib inline

def access(path, dataset_file, data_name, target_size ):
    images = os.listdir(path)
    images.sort()
    id = 0
    for image in images:
        im = cv2.imread(os.path.join(path, image))
        if im is None:
            continue
        if data_name=='train_mask' or data_name=='val_mask' or data_name=='test_mask':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, target_size)
        dataset_file[data_name][id, ...] = im
        id += 1
    return dataset_file

def save_to_hdf5(path, filename):
    dataset_file = h5py.File(os.path.join(path, filename), mode='w')
    dataset_file.create_dataset("train", (17, 256, 256, 3), np.float32)
    dataset_file.create_dataset("train_mask", (17, 256, 256), np.float32)
    dataset_file.create_dataset("val", (3, 256, 256, 3), np.float32)
    dataset_file.create_dataset("val_mask",(3, 256, 256), np.float32)
    dataset_file.create_dataset("test", (20, 256, 256, 3), np.float32)
    dataset_file.create_dataset("test_mask", (20, 256, 256), np.float32)

    dataset_file = access(os.path.join(path, 'train/images'), dataset_file, "train", (256, 256))
    dataset_file = access(os.path.join(path, 'train/masks'), dataset_file, "train_mask", (256, 256))
    dataset_file = access(os.path.join(path, 'val/images'), dataset_file, "val", (256, 256))
    dataset_file = access(os.path.join(path, 'val/masks'), dataset_file, "val_mask", (256, 256))
    dataset_file = access(os.path.join(path, 'test/images'), dataset_file, "test", (256, 256))
    dataset_file = access(os.path.join(path, 'test/masks'), dataset_file, "test_mask", (256, 256))

    dataset_file.close()

def load_data(data_path):
    data = h5py.File(data_path, mode='r')
    x_train = np.array(data["train"])
    y_train = np.array(data["train_mask"])
    x_val = np.array(data["val"])
    y_val = np.array(data["val_mask"])
    x_test = np.array(data["test"])
    y_test = np.array(data["test_mask"])

    y_train = y_train.reshape(y_train.shape[0], 256, 256, 1)
    y_val = y_val.reshape(y_val.shape[0], 256, 256, 1)
    y_test = y_test.reshape(y_test.shape[0], 256, 256, 1)
    data.close()
    return x_train, y_train, x_val, y_val, x_test, y_test

def plot_samples(X, Y):
    X /= 255
    Y /= 255
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1, ax2 = ax.ravel()
    ax1.imshow(np.squeeze(X), cmap='jet')
    ax1.set_title('Retinal Image')
    ax2.imshow(np.squeeze(Y), cmap='gray')
    ax2.set_title('Vessel Image')
    plt.show()



