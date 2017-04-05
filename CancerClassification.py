# Code Credit to:
# 1. Mulholl, Sander, Hamilton: https://www.kaggle.com/c/data-science-bowl-2017#tutorial
# 2. ArnavJain: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
# 3. Guido Zuidhof: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
# 4. Sentdex: https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
from scipy.misc import imresize
import copy
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
import scipy.misc
from skimage import data
from scipy import ndimage as ndi
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.segmentation import clear_border
from plot import plot_3d, plot_ct_scan
import SimpleITK as sitk
import time
import scipy.io
import math

# Constants
HU = True
HM_SLICES = 20
MIN_BOUND = -1000.0 # for normalization, see full-processing-tutorial
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25   # for zero centering, see full-processing-tutorial
LUNA_FOLDER = "/Volumes/brain_cleaner/LUNA_16/subset0/"
INPUT_FOLDER = "/Volumes/brain_cleaner/"        # where folder is located
IMAGE_FOLDER = INPUT_FOLDER + "sample_images/"
CSV_PATH = INPUT_FOLDER + "stage1_labels.csv"



"""Loading Data"""
# Load the test scans for a single dicom file in given folder path
# 03/20/17. Source: 3
def load_test(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices



"""Image Processing for a single patient"""

# get pixels from scan, HU = pixel shift left by 1024(-1024)
# 03/20/17. Source: 3
def get_pixels(slices, hu = True):
    image = np.stack([s.pixel_array for s in slices])
    np.save("raw",image)
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    if hu:
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

    img = np.array(image, dtype=np.int16)           #merges all of the layers of the image
    # img.flatten()
    # np.save("thresholded", img)
    return img


# Resampling
# 03/20/17. Source: 3
def resample(image, spacing, new_spacing=[1,1,1]):
    # # Determine current pixel spacing

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')


    return image, new_spacing


# segments the lung area for one slice of image(--LUNA 16 kernel)
# 03/21/17. Source: 2
def get_segmented_lung(old_im, plot=True, hu = False):
    im = copy.deepcopy(old_im)

    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.

    '''
    if hu:
        th = 604 - 1024
    else:
        th = 604    # 604 for pixel value
    binary = im < th
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    plt.show()

    return im

def addZeroPadding(image, xPad, yPad, zPad):
    xPad1 = xPad/2
    xPad2 = xPad - xPad1

    yPad1 = yPad/2
    yPad2 = yPad - yPad1

    zPad1 = zPad/2
    zPad2 = zPad - zPad1
    newimage = np.lib.pad(image, ((zPad1, zPad2), (xPad1, xPad2), (yPad1, yPad2)), 'constant', constant_values=(0, 0))
    return newimage

# segments all lung images from an entire scan
#  03/21/17. Source: 2
def segment_lungs_from_scan(scan_pixels, hu = False):
    num = len(scan_pixels)
    if num ==0:
        raise ValueError("eh-oh, you need at least one slice in your scan")
    return np.asarray([get_segmented_lung(slice,False, hu) for slice in scan_pixels])

# # resize images for one patient
def resize(slices, spacing, size = (150,150)):
    # slices = np.array([imresize(pixel_array, size) for pixel_array in slices])
    new_shape = np.round([20.0,150.0,150.0])
    real_resize_factor = new_shape / slices.shape
    print "\treal resize factor:",real_resize_factor
    new_spacing = spacing / real_resize_factor
    downed = scipy.ndimage.interpolation.zoom(slices, real_resize_factor, mode='nearest')
    print "\ttesting resize, shape of downed: ", downed.shape
    return downed, new_spacing

# original function for image_processing, w/ resampling
# processes all slices for a loaded scan
def image_processing(scan_pixels, spacing, train = False):
    """
    Train is True when processing LUNA16 data
    """
    # get pixels
    # scan_pixels = get_pixels(scan_pixels, HU)
    # resample
    print "\tresampling..."
    print "\tshape of scan_pixels:",scan_pixels.shape
    lung_img, new_spacing = resample(scan_pixels,spacing,[1,1,1])
    print "\tshape of resampled pixels:", lung_img.shape
    # lung_img, new_spacing = resize(scan_pixels,spacing)
    # np.save("resampled", lung_img)

    # segment lungs
    print "\tsegmenting lungs..."
    segmented = segment_lungs_from_scan(lung_img, HU)
    # np.save("segmented", segmented)

    # normalize and zero centering. 03/20/2017. Source: 3
    image = (segmented - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image - PIXEL_MEAN
    # np.save("normalized", image)

    #TODO: zero padding in 3d dimensions
    # image = resize(image)
    sz, sy,sx = image.shape
    z = 400 - sz
    y = 400 - sy
    x = 400 - sx
    image = addZeroPadding(image, x,y,z)
    print "\tshape after zero padding", image.shape

    # lung_img is the resampled image(ignore)
    # image is the resampled, segmented and normalized image
    return lung_img, image, new_spacing


"""Extracts training/testing data for a list of patients"""
def create_training(patients, label_path = CSV_PATH, data_path = IMAGE_FOLDER, hm_slices = HM_SLICES):
    """

    :param patients:  list of patient ids
    :param label_path: path of the label csv
    :param data_path: folder where images are stored
    :param hm_slices:  # of chunked slices
    :return: new_patients, new slices
    """

    # patients = os.listdir(data_path)
    print "Started Computing Training Data!"

    #read labels
    labels_df = pd.read_csv(label_path, index_col=0)

    # Method 1: using simple list
    labels = []
    X_train = []
    new_patients = []

    # # Method 2: using pickle
    # patient_info = {}

    # 03/22/17. Source: 4
    for patient in patients:
        print "patient id: ", patient
        start = time.time()
        print
        try:

            label = labels_df.get_value(patient, 'cancer')
            labels.append(label)
            new_patients.append(patient)

            #TODO: store data on pickle
            # this_patient = {}
            # this_patient["label"] = label

            patient_path = data_path + patient
            scans =load_test(patient_path)
            pixel_arrays = get_pixels(scans, HU)
            print "\tpixel_array:",pixel_arrays.shape
            spacing = np.array([scans[0].SliceThickness] + scans[0].PixelSpacing, dtype=np.float32)
            lung_img, data, new_spacing = image_processing(pixel_arrays, spacing)
            X_train.append(data)
            # plot_ct_scan(data)

            # store data
        except Exception as e:
            # again, some patients are not labeled, but JIC we still want the error if something
            # else is wrong with our code
            print(str(e))
        end = time.time()
        print "\t time used: ", end-start

    return np.array(new_patients), np.array(labels), np.array(X_train)




# Create training data from sample_files
start = time.time()
print("hello")
patients = os.listdir(IMAGE_FOLDER)
patients.sort()
print patients
p, l, x = create_training(patients)
print "printing sizes...."
print "Shape of X_train", x.shape

np.save("patients",x)
np.save("labels",l)
end = time.time()
print"total time used",(end - start)
# print "cancer:", l.count(1)
# print "cancerless:,",l.count(0)
print l
# np.save("sample_images",x)

patients = np.load("patients.npy")
labels = np.load("labels.npy")
# print "cancer"
# plt.imshow(y[2][70], cmap=plt.cm.gray)  # Show some slice
# plt.show()
# plot_ct_scan(y[1])
# print y.shape
print labels
