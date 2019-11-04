"""
objective : Execute
author(s) : Ashwin de Silva
date      : 
"""

from src.sclera_segmentation.utilities import save_to_hdf5
import matplotlib.pyplot as plt
import h5py
import numpy as np

#########################################
# save to hdf5
#########################################

imageset_path = '/Volumes/Seagate Expansion Drive/SBVP_with_masks'
save_to_hdf5(imageset_path, size=(256, 256))

#########################################
# visualize
#########################################

# data_path = '/Users/ashwin/DeepRetina/src/sclera_segmentation/data/scleral_dataset.hdf5'
# data = h5py.File(data_path, mode='r')
# x_train = np.array(data["val"])
# y_train = np.array(data["val_mask"])
#
# y_train[y_train >= 0.5] = 1
#
# idx = 10
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
# ax1, ax2 = ax.ravel()
# ax1.imshow(x_train[idx, ...]/255.)
# ax1.set_title('Original Image')
# ax2.imshow(np.squeeze(y_train[idx]))
# ax2.set_title('Annontated Sclera')
# plt.show()
