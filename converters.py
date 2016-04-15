import numpy as np


def distance(coord1, coord2):
    assert len(coord1) == len(coord2)
    arr1 = np.array(coord1)
    arr2 = np.array(coord2)
    return np.linalg.norm(arr1 - arr2)


def rgba_to_rgb(np_array, bgc=255):
    (height, width, depth) = np_array.shape
    assert depth is 4

    img = np_array.astype(np.float64)
    img[:,:,0] = ((1 - img[:,:,3]/255) * bgc) + (img[:,:,3]/255 * img[:,:,0])
    img[:,:,1] = ((1 - img[:,:,3]/255) * bgc) + (img[:,:,3]/255 * img[:,:,1])
    img[:,:,2] = ((1 - img[:,:,3]/255) * bgc) + (img[:,:,3]/255 * img[:,:,2])
    img = img[:,:,:-1]
    return img.astype(np.uint8)
