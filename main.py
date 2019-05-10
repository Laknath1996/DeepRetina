'''
execute DRIUNet
'''

from src.DRIUNet import *
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from src.utilities import *
import scikitplot as skplt

cwd = os.getcwd()
data_path = os.path.join(cwd, 'datasets/DRIVE_2.hdf5')
model_path = os.path.join(cwd, 'models/model1.hdf5')

# save_to_hdf5(os.path.join(cwd, 'datasets'), 'DRIVE_2.hdf5')

deepRetina = DRIUNET(data_path, model_path)
# deepRetina.train()
deepRetina.predict()



