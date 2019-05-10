'''
execute DRIUNet
'''

from src.DRIUNet import *
import matplotlib.pyplot as plt
import os
from src.utilities import *

cwd = os.getcwd()
data_path = os.path.join(cwd, 'datasets/DRIVE.hdf5')
model_path = os.path.join(cwd, 'models/model1.hdf5')

# save_to_hdf5(dataset_dir, 'DRIVE.hdf5')

deepRetina = DRIUNET(data_path, model_path)
deepRetina.train()
deepRetina.predict()




