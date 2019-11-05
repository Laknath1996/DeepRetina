'''
execute DRIUNet

work to complete

Read the papers
Modify the architectures
Read on the accuracy metrics F1, sensitivity etc
Use the full scale images
Don't use a validation set
Look in to optical disk segmentation too

'''

"""
objective : Execute
author(s) : Ashwin de Silva
date      : 
"""

from src.sclera_segmentation.utilities import save_to_hdf5, plot_samples
import matplotlib.pyplot as plt
import h5py
import numpy as np
from src.sclera_segmentation.ScleraNet import *

#########################################
# save to hdf5
#########################################

# imageset_path = '/Volumes/Seagate Expansion Drive/SBVP_with_masks'
# save_to_hdf5(imageset_path, size=(256, 256))

#########################################
# visualize
#########################################

# data_path = '/Users/ashwin/DeepRetina/src/sclera_segmentation/data/scleral_dataset'
# data = h5py.File(data_path, mode='r')
# x_train = np.array(data["val"])
# y_train = np.array(data["val_mask"])
#
# y_train[y_train >= 0.5] = 255
# y_train[y_train < 0.5] = 0
#
# plot_samples(x_train[10], y_train[10])

#########################################
# train
#########################################

data_path = '/home/ashwind/DeepRetina/src/sclera_segmentation/data/scleral_dataset.hdf5'
model_path = '/home/ashwind/DeepRetina/src/sclera_segmentation/models/sclera_model.hdf5'

scn = ScleraNet(data_path, model_path)
scn.train()

# from src.DRIUNet import *
# from src.utilities import *
#
# cwd = os.getcwd()
# data_path = os.path.join(cwd, 'datasets/DRIVE_3.hdf5')
# model_path = os.path.join(cwd, 'models/model.hdf5')
#
# # save_to_hdf5(os.path.join(cwd, 'datasets'), 'DRIVE_3.hdf5')
#
# deepRetina = DRIUNET(data_path, model_path)
# # deepRetina.train()
# # deepRetina.predict()



