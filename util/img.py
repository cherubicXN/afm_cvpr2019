import numpy as np

def display_format(image):
    image_disp = image.transpose([1,2,0])

    image_disp[...,0] = image_disp[...,0]*0.229 + 0.485
    image_disp[...,1] = image_disp[...,1]*0.224 + 0.456
    image_disp[...,2] = image_disp[...,2]*0.225 + 0.406

    return image_disp