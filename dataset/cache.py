import os
import os.path as osp
import json
import cv2
from .misc import AugmentationHorizontalFlip, AugmentationVerticalFlip
from lib.afm_op import afm
import torch
import numpy as np
from tqdm import tqdm
class AfmTrainCache:
    def __init__(self, root, img_res, afm_res):
        self.root = root
        self.img_res = img_res
        self.afm_res = afm_res
        self.image_dir = osp.join(self.root,'images')
        self.anno_path = osp.join(self.root,'train.json')
        self.cached_root = osp.join(self.root, '.cache')
        self.cached_image_dir = osp.join(self.root, '.cache','img','{}x{}'.format(img_res[0],img_res[1]))
        self.cached_afmap_dir = osp.join(self.root, '.cache','afm','{}x{}'.format(afm_res[0],afm_res[1]))
        self.cached_label_dir = osp.join(self.root, '.cache','idx','{}x{}'.format(afm_res[0],afm_res[1]))
        self._check_directory()
        self.makeCache()

    def _check_directory(self):
        def makedir(path):
            if os.path.isdir(path) is not True:
                os.makedirs(path)
        makedir(self.cached_root)
        makedir(self.cached_image_dir)
        makedir(self.cached_afmap_dir)
        makedir(self.cached_label_dir)

    def len(self):
        return len(self.dataset)

    def get_path(self, idx):
        image_path = osp.join(self.cached_image_dir,self.dataset[idx]+'.png')
        afmap_path = osp.join(self.cached_afmap_dir,
        self.dataset[idx]+'.npy')
        label_path = osp.join(self.cached_label_dir,
        self.dataset[idx]+'.npy')
        return image_path, afmap_path, label_path

    def makeCache(self):
        with open(self.anno_path, 'r') as handle:
            dataset = json.load(handle)

        lst_path = osp.join(self.cached_root, 'list.txt')
        if osp.isfile(lst_path) is True:
            with open(lst_path,'r') as stream:
                datanames = [f.rstrip('\n') for f in stream.readlines()]
            self.dataset = datanames
            return True

        stream = open(osp.join(self.cached_root,'list.txt'),'w')
        
        datanames = []
        for data in tqdm(dataset):
            names, imgs, afms, idxs = self._makeData(data)
            datanames += names
            for n, im, af, idx in zip(names, imgs, afms, idxs):    
                stream.write(n+'\n')
                cv2.imwrite(osp.join(self.cached_image_dir,n+'.png'),im)
                np.save(osp.join(self.cached_afmap_dir,n+'.npy'),af)
                np.save(osp.join(self.cached_label_dir,n+'.npy'),
                idx)
        stream.close()

        self.dataset = datanames
        
        return True



    def _makeData(self, data):
        datanames = [data['filename'].rstrip('.png'),
                        data['filename'].rstrip('.png')+'_lr',
                        data['filename'].rstrip('.png')+'_ud']

        image = cv2.imread(osp.join(self.image_dir,data['filename']))
        lines = np.array(data['lines'],dtype=np.float32)        
        image_lr, lines_lr = AugmentationHorizontalFlip(image,lines)
        image_ud, lines_ud = AugmentationVerticalFlip(image,lines)


        height, width = image.shape[:2]
        resize = lambda image: cv2.resize(image,(self.img_res[1], self.img_res[0]))
        image, image_lr, image_ud = list(map(resize,[image,image_lr,image_ud]))

        num_lines = lines.shape[0]
        shape_info = torch.IntTensor([[0, num_lines, height, width],
                               [num_lines, 2*num_lines, height, width],
                               [2*num_lines, 3*num_lines, height, width]])
        lines = torch.from_numpy(np.vstack([lines,lines_lr,lines_ud]))

        
        afmap, label = afm(lines.cuda(),shape_info.cuda(), self.afm_res[0],self.afm_res[1])

        afmap = afmap.data.cpu().numpy()
        label = label.data.cpu().numpy()
        
        return datanames, [image, image_lr, image_ud], [afmap[0],afmap[1],afmap[2]], [label[0],label[1],label[2]]


if __name__ == "__main__":
    
    cache = AfmTrainCache('data/wireframe',[320,320],[320,320])
    import pdb
    pdb.set_trace()

# def make_cache(root, in_res, out_res):
#     image_dir = 