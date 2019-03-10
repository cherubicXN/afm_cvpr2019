import os
import os.path as osp
from .afmDataset import AFMTrainDataset, AFMTestDataset, collect_fn
import torch.utils.data as data

def build_train_dataset(config):
    
    root_list = [osp.abspath(osp.join(osp.dirname(__file__),'..','data', f)) for f in config.DATASETS.TRAIN]

    IN_RES = [config.INPUT.IN_RES]*2
    OUT_RES= [config.INPUT.OUT_RES]*2

    get_dataset = lambda path: AFMTrainDataset(path, img_res=IN_RES, afm_res=OUT_RES)
    
    dataset = data.ConcatDataset(list(map(get_dataset,root_list)))    

    dataset = data.DataLoader(dataset, batch_size=config.SOLVER.BATCH_SIZE,shuffle=True,num_workers=config.DATALOADER.NUM_WORKERS, pin_memory=True)

    return dataset

def build_test_dataset(config):
    root_list = [osp.abspath(osp.join(osp.dirname(__file__),'..','data', f)) for f in config.DATASETS.TEST]

    if root_list == []:
        return None
    IN_RES = [config.INPUT.IN_RES]*2
    get_dataset = lambda path: data.DataLoader(AFMTestDataset(path,IN_RES),batch_size=1, shuffle=False, collate_fn=collect_fn, num_workers=config.DATALOADER.NUM_WORKERS)

    datasets  = list(map(get_dataset,root_list))

    return datasets