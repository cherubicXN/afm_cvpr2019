import os
import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from tqdm import tqdm
import json
import glob

this_dir = osp.abspath(osp.dirname(__file__))

data_root = osp.join(this_dir,'york_raw')
output_root = osp.join(this_dir,'york')
if osp.isdir(output_root) is not True:
    os.makedirs(output_root)

test_lst = glob.glob(osp.join(data_root,'*.mat'))
test_lst = [osp.basename(f).rstrip('.mat') for f in test_lst]
    
def load_datum(filename):
    image = cv2.imread(osp.join(data_root,filename)+'_rgb.png')
    lsgs = sio.loadmat(osp.join(data_root,filename)+'.mat')['line']

    return image, {'filename': filename+'.png', 
            'lines'   : lsgs.tolist(), 'height':image.shape[0], 'width': image.shape[1]}


if __name__ == "__main__":
    image_path = osp.join(output_root,'images')
    if osp.isdir(image_path) is not True:
        os.makedirs(image_path)

    test_hanlde = open(osp.join(output_root,'test.txt'),'w')

    test_annotations = []
    for filename in tqdm(test_lst):
        image, data = load_datum(filename)
        lines = data['lines']
        test_annotations += [data]
        if osp.isfile(osp.join(image_path,data['filename'])) is not True:
            cv2.imwrite(osp.join(image_path,data['filename']), image)
        test_hanlde.write(data['filename']+'\n')
    
    test_hanlde.close()
    
    with open(osp.join(output_root,'test.json'),'w') as json_file:
       json.dump(test_annotations, json_file)
