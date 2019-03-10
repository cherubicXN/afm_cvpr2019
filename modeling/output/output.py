import modeling.registry as registry
from modeling.registry import OUTPUT_METHODS
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
import scipy.io as sio

@OUTPUT_METHODS.register("display")
def display(data_dict, cfg):
    image = data_dict['image'] 
    # image_resized = data_dict['image_resized']

    # image = np.transpose(image, [1,2,0])
    # image[...,0] = (image[...,0]*0.229 + 0.485)
    # image[...,1] = (image[...,1]*0.224 + 0.456)
    # image[...,2] = (image[...,2]*0.225 + 0.406)
    # image = np.array(image*255,dtype=np.uint8)
    # import pdb
    # pdb.set_trace()
    height, width = image.shape[:2]
    h0, w0 = data_dict['afmap_pred'].shape[1:]

    scale_factor = np.array([width/w0,height/h0,width/w0,height/h0],dtype=np.float32)


    lines = data_dict['lines_pred_resized']
    lines[:,:4] *= scale_factor

    lengths = np.sqrt((lines[:,2]-lines[:,0])*(lines[:,2]-lines[:,0]) + (lines[:,3]-lines[:,1])*(lines[:,3]-lines[:,1]))
    ratio = lines[:,4]/lengths    

    threshold = cfg.TEST.DISPLAY.THRESHOLD 
    idx = np.where(ratio<=threshold)[0]    
    lines = lines[idx]

    plt.imshow(image)
    plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],'r-')
    plt.xlim([0,width])
    plt.ylim([height,0])
    plt.axis('off')
    plt.show()


@OUTPUT_METHODS.register("save")
def save(data_dict, cfg):
    fname = data_dict['fname'].rstrip('.png')
    image = data_dict['image'] 
    image_resized = data_dict['image_resized']

    # image = np.transpose(image, [1,2,0])
    # image[...,0] = (image[...,0]*0.229 + 0.485)
    # image[...,1] = (image[...,1]*0.224 + 0.456)
    # image[...,2] = (image[...,2]*0.225 + 0.406)
    # image = np.array(image*255,dtype=np.uint8)
    # import pdb
    # pdb.set_trace()
    height, width = image.shape[:2]
    h0, w0 = image_resized.shape[1:]

    scale_factor = np.array([width/w0,height/h0,width/w0,height/h0],dtype=np.float32)    


    lines = data_dict['lines_pred_resized']
    lines[:,:4] *=scale_factor


    output_dir = data_dict['output_dir']
    if osp.isdir(output_dir) is not True:
        os.makedirs(output_dir)

    output_path = osp.join(output_dir, fname+'.mat')
    
    sio.savemat(output_path, mdict={
        'height': height,
        'width': width,
        'gt': data_dict['lines_gt'],
        'pred': lines,
    })

    

    
    # lines_pred = data_dict['lines_pred']

def build_output_method(cfg):
    assert cfg.TEST.OUTPUT_MODE in registry.OUTPUT_METHODS

    return registry.OUTPUT_METHODS[cfg.TEST.OUTPUT_MODE]

