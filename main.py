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
from src.utilities import *

cwd = os.getcwd()
data_path = os.path.join(cwd, 'datasets/DRIVE_3.hdf5')
model_path = os.path.join(cwd, 'models/model.hdf5')

# save_to_hdf5(os.path.join(cwd, 'datasets'), 'DRIVE_3.hdf5')

deepRetina = DRIUNET(data_path, model_path)
# deepRetina.train()
# deepRetina.predict()



