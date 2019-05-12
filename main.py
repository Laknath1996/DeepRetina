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

from src.DRIUNet import *
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from src.utilities import *
import scikitplot as skplt

cwd = os.getcwd()
data_path = os.path.join(cwd, 'datasets/DRIVE_3.hdf5')
model_path = os.path.join(cwd, 'models/model3.hdf5')

# save_to_hdf5(os.path.join(cwd, 'datasets'), 'DRIVE_3.hdf5')

deepRetina = DRIUNET(data_path, model_path)
deepRetina.train()
deepRetina.predict()



