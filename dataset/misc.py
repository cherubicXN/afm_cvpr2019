import numpy as np

def AugmentationHorizontalFlip(image, lines):
    height, width = image.shape[:2]
    image_lr = image.copy()
    lines_lr    = lines.copy()
    image_lr = image_lr[:,::-1,:]
    lines_lr[:,0] = width - lines_lr[:,0]
    lines_lr[:,2] = width - lines_lr[:,2]

    return image_lr, lines_lr

def AugmentationVerticalFlip(image, lines):
    height, width = image.shape[:2]
    image_ud = image.copy()
    lines_ud = lines.copy()
    image_ud = image_ud[::-1,:,:]
    lines_ud[:,1] = height - lines_ud[:,1]
    lines_ud[:,3] = height - lines_ud[:,3]

    return image_ud, lines_ud
    