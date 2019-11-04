"""
objective : preprocess the scleral images
author(s) : Ashwin de Silva, Malsha Perera
date      : 
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import chan_vese


def resize(im, size):
    """
    resize the image
    :param im: image
    :param size: size tuple
    :return: resized image
    """
    return cv2.resize(im, size)

def extract_sclera_chanvese(im):
    """
    extract the sclera using chan-vese segmentation
    :param im: image
    :return: binary mask of the sclera
    """
    # resize the image
    im = resize(im, (256, 256))

    # extract the red channel
    im = im[..., 0]

    # intensity windowing
    im = np.array(im, dtype=np.float)
    im = im**0.3

    # scleral segmentation
    cv = chan_vese(im, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=500, dt=0.5, init_level_set="checkerboard", extended_output=True)
    im = cv[0]

    return im

def main():
    im = cv2.imread('/Users/ashwin/DeepRetina/src/scleral_test.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
  main()


