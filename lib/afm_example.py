import pickle
import numpy as np
import time
from afm.gpu_afm import afm_transform_gpu as afm_trans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm

data_root = '../data/wireframe_raw/pointlines'

files = [f for f in os.listdir(data_root) if f.endswith('.pkl')]

def afm_data(file):
    with open(os.path.join(data_root,file),'rb') as handle:
        d = pickle.load(handle, encoding='latin1')
        height, width = d['img'].shape[:2]
        points = d['points']
        lines = d['lines']
        lsgs = np.array([[points[i][0], points[i][1], points[j][0], points[j][1]] for i, j in lines],
                        dtype=np.float32)
        lsgs[:,0] *= 320.0/width
        lsgs[:,1] *= 320.0/height
        lsgs[:,2] *= 320.0/width
        lsgs[:,3] *= 320.0/height
        # afmap, aflabel = afm_trans(lsgs, height, width, 0)
        afmap, aflabel = afm_trans(lsgs, 320, 320, 0)
        bbox = np.zeros((lsgs.shape[0],4),dtype=np.float32)        

        for label in range(lsgs.shape[0]):
            y,x = np.where(aflabel == label)
            xmin = x.min()
            ymin = y.min()
            xmax = x.max()
            ymax = y.max()
            # patch = aflabel[ymin:ymax+1,xmin:xmax+1]
            # import pdb
            # pdb.set_trace()
            bbox[label] = [xmin,ymin,xmax,ymax]
    
    return afmap, aflabel, bbox


afmap, aflabel, afbox = afm_data('00030254.pkl')
plt.imshow(afmap[0])
plt.show()


start_time = time.time()

shapes = {}
for f in tqdm(files):
    afmap, aflabel, afbox = afm_data(f)
    shape_data = aflabel.shape
    if shape_data in shapes.keys():
        shapes[shape_data] += 1
    else:
        shapes.update({shape_data:1})
    
    
    # plt.show()
    
    fig,ax = plt.subplots(1)
    ax.imshow(aflabel)     
    for lb, bbox in enumerate(afbox):
        xmin,ymin,xmax,ymax = bbox   
        
        # ax.imshow(aflabel==lb)
        ax.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],'r-')
        # rect = patches.Rectangle((bbox[0],bbox[1]), xmax-xmin, ymax-ymin,edgecolor='r',fill=False)
        # ax.add_patch(rect)
    # plt.xlim([-10,600])
    # plt.ylim([-10,600])
    plt.show()

# xx,yy = np.meshgrid(range(width),range(height))
# xx = np.array(xx,dtype=np.float32)
# yy = np.array(yy,dtype=np.float32)

# xx += afmap[0]
# yy += afmap[1]

# xx,yy = xx.flatten(), yy.flatten()

# plt.imshow(d['img'])
# plt.plot(xx,yy,'r.')
# plt.show()
end_time = time.time() - start_time

print(end_time)
import pdb
pdb.set_trace()