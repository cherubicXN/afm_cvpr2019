import os
import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from tqdm import tqdm
import json

this_dir = osp.abspath(osp.dirname(__file__))

data_root = osp.join(this_dir,'wireframe_raw')
output_root = osp.join(this_dir,'wireframe')
if osp.isdir(output_root) is not True:
    os.makedirs(output_root)

with open(osp.join(data_root, 'train.txt')) as handle:
    train_lst = [f.rstrip('.jpg\n') for f in handle.readlines()]

with open(osp.join(data_root, 'test.txt')) as handle:
    test_lst = [f.rstrip('.jpg\n') for f in handle.readlines()]
    
def load_datum(filename):
    with open(osp.join(data_root,'pointlines',filename+'.pkl'),'rb') as handle:
        d = pickle.load(handle, encoding='latin1')
        h, w = d['img'].shape[:2]
        points = d['points']
        lines = d['lines']
        lsgs = np.array([[points[i][0], points[i][1], points[j][0], points[j][1]] for i, j in lines],
                        dtype=np.float32)
        image = d['img']
    
    return image, {'filename': filename+'.png', 
            'lines'   : lsgs.tolist(), 'height':image.shape[0], 'width': image.shape[1]}


if __name__ == "__main__":
    image_path = osp.join(output_root,'images')
    if osp.isdir(image_path) is not True:
        os.makedirs(image_path)

    train_annotations = []
    for filename in tqdm(train_lst):
        image, data = load_datum(filename)
        lines = data['lines']
        train_annotations += [data]
        if osp.isfile(osp.join(image_path,data['filename'])) is not True:
            cv2.imwrite(osp.join(image_path,data['filename']), image)

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
    
    with open(osp.join(output_root,'train.json'),'w') as json_file:
       json.dump(train_annotations, json_file)

    with open(osp.join(output_root,'test.json'),'w') as json_file:
       json.dump(test_annotations, json_file)
