# Code Credit to:
# 1. Mulholl, Sander, Hamilton: https://www.kaggle.com/c/data-science-bowl-2017#tutorial
# 2. ArnavJain: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
# 3. Guido Zuidhof: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
# 4. Sentdex: https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d_old
import SimpleITK as sitk

# TODO: needs to be debugged
# 03/30/2017. Source: 2
def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    print "pass1"
    ax = fig.add_subplot(111, projection='3d')
    print "pass2"


    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# 03/20/2017. Source: 2
def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)

    plt.show()
    return
