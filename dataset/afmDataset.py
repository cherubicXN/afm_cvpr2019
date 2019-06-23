import torch.utils.data as data
import numpy as np
import os.path as osp
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch
from .cache import AfmTrainCache
class AFMTrainDataset(data.Dataset):
    def __init__(self, data_root, img_res = [320,320], afm_res = [320,320]):
        '''
        Training dataset should have the following format:
            DATASET_NAME/images
            DATASET_NAME/annote.json
        
        The json file should have N items and each item should contains an image name and the line segment annotations.      
        '''
        self.data_root = data_root
        self.img_res = img_res
        self.afm_res = afm_res

        self.cache = AfmTrainCache(self.data_root, self.img_res, self.afm_res)
        
    def __len__(self):
        return self.cache.len()
    
    def __getitem__(self, idx):
        imgpath, afmpath, _ = self.cache.get_path(idx)

        image = cv2.imread(imgpath)
        afmap = np.load(afmpath)

        image = np.array(image,dtype=np.float32)/255.0
        image[...,0] = (image[...,0] - 0.485)/0.229
        image[...,1] = (image[...,1] - 0.456)/0.224
        image[...,2] = (image[...,2] - 0.406)/0.225
        image = np.transpose(image,(2,0,1))

        return image, afmap

class AFMTestDataset(data.Dataset):
    def __init__(self, data_root, img_res = [320,320]):        
        '''
        For testing dataset, the images should be placed in the DATASET_NAME/images

        If you have a list of testing images, the filenames should be saved in the test.txt  
        '''
        self.data_root = data_root
        if osp.isfile(osp.join(self.data_root,'test.json')) is True:
            with open(osp.join(self.data_root,'test.json'),'r') as handle:
                dataset = json.load(handle)
            for data in dataset:
                data['lines'] = np.array(data['lines'],dtype=np.float32)

        elif osp.isfile(osp.join(self.data_root,'test.txt')) is True:
            with open(osp.join(self.data_root,'test.txt'),'r') as handle:
                filename = [f.rstrip('\n') for f in handle.readlines()]
        
            dataset = [{'filename': f, 'lines': np.array([0,0,0,0],dtype=np.float32)} for f in filename]

        else:
            raise NotImplementedError()
        
        self.dataset = dataset
        self.img_res = img_res
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = osp.join(self.data_root, 'images',self.dataset[idx]['filename'])

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_res[1],self.img_res[0]))
        image = np.array(image,dtype=np.float32)/255.0              
        image[...,0] = (image[...,0] - 0.485)/0.229
        image[...,1] = (image[...,1] - 0.456)/0.224
        image[...,2] = (image[...,2] - 0.406)/0.225
        image = np.transpose(image,(2,0,1))

        lines = self.dataset[idx]['lines']

        fname = self.dataset[idx]['filename']

        return image, lines, fname

def collect_fn(data):
    images, lines, fnames = zip(*data)
    images = torch.stack([torch.from_numpy(img) for img in images],0)
    lines = [torch.from_numpy(ll) for ll in lines]

    batch_size = images.shape[0]
    start = np.array([ll.size()[0] for ll in lines])
    end   = np.cumsum(start)
    start = end-start

    shape_info = np.array([[start[i], end[i], images.shape[2], images.shape[3]] for i in range(batch_size) ])

    lines = torch.cat(lines, dim=0)

    shape_info = torch.IntTensor(shape_info)

    return images, lines, shape_info, fnames

