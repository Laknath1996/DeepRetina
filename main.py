'''
execute DRIUNet
'''

# from src.DRIUNet import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from src.utilities import *
import matplotlib


cwd = os.getcwd()
dataset_dir = os.path.join(cwd, 'datasets')
print(dataset_dir)

# save_to_hdf5(dataset_dir, 'DRIVE.hdf5')

x_train, y_train, x_val, y_val, x_test, y_test = load_data(dataset_dir, 'DRIVE.hdf5')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()
ax1.imshow(np.squeeze(x_train[1]))
ax1.set_title('Wrapped Image')
ax2.imshow(np.squeeze(y_train[1]))
ax2.set_title('Original Image')
plt.show()


