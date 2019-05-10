'''
execute DRIUNet
'''

from src.DRIUNet import *
import matplotlib.pyplot as plt
import os
from src.utilities import *

cwd = os.getcwd()
dataset_dir = os.path.join(cwd, 'datasets')
print(dataset_dir)

# save_to_hdf5(dataset_dir, 'DRIVE.hdf5')

x_train, y_train, x_val, y_val, x_test, y_test = load_data(dataset_dir, 'DRIVE.hdf5')

x_train /= 255
y_train /= 255

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()
ax1.imshow(np.squeeze(x_train[1]), cmap='jet')
ax1.set_title('Retinal Image')
ax2.imshow(np.squeeze(y_train[1]), cmap='gray')
ax2.set_title('Vessel Image')
plt.show()

deepRetina = DRIUNET('model1.hdf5')
deepRetina.train()
deepRetina.predict()




